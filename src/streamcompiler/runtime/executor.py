"""Event-driven runtime components.

The whole-machine scheduler stays separate from per-backend kernel compilers.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from streamcompiler.compile.pipeline import SpecializedArtifact

from streamcompiler.errors import RuntimePlanError
from streamcompiler.ir.resource_graph import MemoryClass, ResourceGraph
from streamcompiler.planner.maximal import ExecutionPlan


@dataclass
class ResidentCopy:
    memory: str
    version: int
    nbytes: int
    handle: Any = None


@dataclass
class TensorRecord:
    tensor_id: str
    home: str
    copies: list[ResidentCopy] = field(default_factory=list)
    version: int = 0
    immutable: bool = True


class TensorDirectory:
    def __init__(self) -> None:
        self._records: dict[str, TensorRecord] = {}
        self._lock = threading.RLock()

    def register(self, record: TensorRecord) -> None:
        with self._lock:
            self._records[record.tensor_id] = record

    def get(self, tensor_id: str) -> TensorRecord:
        with self._lock:
            if tensor_id not in self._records:
                raise RuntimePlanError(f"Unknown tensor {tensor_id}")
            return self._records[tensor_id]

    def invalidate_stale(self, tensor_id: str, version: int) -> None:
        with self._lock:
            rec = self._records[tensor_id]
            rec.copies = [c for c in rec.copies if c.version == version]
            rec.version = version


class TieredAllocator:
    """Tracks outstanding bytes per memory resource; does not invent capacity."""

    def __init__(self, machine: ResourceGraph) -> None:
        self.machine = machine
        self._used: dict[str, int] = {name: 0 for name in machine.memory}
        self._lock = threading.RLock()

    def allocate(self, memory: str, nbytes: int) -> None:
        with self._lock:
            mem = self.machine.memory.get(memory)
            if mem is None:
                raise RuntimePlanError(f"Unknown memory resource {memory}")
            if mem.allocatable_bytes > 0 and self._used[memory] + nbytes > mem.allocatable_bytes:
                raise RuntimePlanError(
                    f"Allocator would exceed {memory}: {self._used[memory] + nbytes} > {mem.allocatable_bytes}"
                )
            self._used[memory] += nbytes

    def release(self, memory: str, nbytes: int) -> None:
        with self._lock:
            self._used[memory] = max(0, self._used[memory] - nbytes)

    def used(self) -> dict[str, int]:
        with self._lock:
            return dict(self._used)


class EventPool:
    def __init__(self) -> None:
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def record(self, name: str) -> threading.Event:
        with self._lock:
            ev = threading.Event()
            ev.set()
            self._events[name] = ev
            return ev

    def wait(self, name: str, timeout: float | None = None) -> bool:
        with self._lock:
            ev = self._events.get(name)
        if ev is None:
            raise RuntimePlanError(f"Unknown event {name}")
        return ev.wait(timeout)


class CpuExecutor:
    def __init__(self, workers: int = 4) -> None:
        self.workers = workers

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)


class GpuExecutor:
    """Dispatches to the owning backend; unavailable backends fail explicitly."""

    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id

    def submit(self, executable: Any, dependencies: list[Any] | None = None) -> Any:
        from streamcompiler.backends import backend_by_id

        backend = backend_by_id(self.backend_id)
        if backend is None or not backend.available():
            raise RuntimePlanError(f"GPU backend {self.backend_id} unavailable")
        return backend.execute(executable, dependencies or [])


class IoExecutor:
    def prefetch(self, path: str, nbytes: int) -> dict[str, Any]:
        # Milestone: record intent; concrete io_uring/libaio paths are optional.
        return {"path": path, "nbytes": nbytes, "status": "planned"}


class CollectiveExecutor:
    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id

    def allreduce(self, tensors: Any, devices: tuple[str, ...]) -> Any:
        from streamcompiler.communication import select_communication_backend

        backend = select_communication_backend(devices)
        return backend.allreduce(tensors, devices)


class TelemetryCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, kind: str, **payload: Any) -> None:
        self.events.append({"kind": kind, "t": time.time(), **payload})


class PlanExecutor:
    def __init__(self, plan: ExecutionPlan, machine: ResourceGraph) -> None:
        self.plan = plan
        self.machine = machine
        self.directory = TensorDirectory()
        self.allocator = TieredAllocator(machine)
        self.events = EventPool()
        self.cpu = CpuExecutor()
        self.io = IoExecutor()
        self.collectives = CollectiveExecutor(plan.communication_backend)
        self.telemetry = TelemetryCollector()
        self._gpu_executors = {
            d: GpuExecutor(next(p.backend_id for p in plan.placements if p.device == d))
            for d in plan.devices_used
            if any(p.device == d and p.backend_id != "cpu" for p in plan.placements)
        }

    def run(self) -> dict[str, Any]:
        """Execute the specialized plan asynchronously per device where possible."""
        results: dict[str, Any] = {}
        threads: list[threading.Thread] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def run_placement(region_id: str, device: str, backend_id: str) -> None:
            try:
                self.telemetry.emit("start", region=region_id, device=device)
                if backend_id == "cpu":
                    out = self.cpu.submit(lambda: {"region": region_id, "device": device})
                else:
                    exe = self._gpu_executors.get(device)
                    if exe is None:
                        raise RuntimePlanError(f"No GPU executor for {device}")
                    # CompiledRegion stub for milestone dispatch.
                    from streamcompiler.backends.base import CompiledRegion

                    out = exe.submit(
                        CompiledRegion(
                            region_id=region_id,
                            device=device,
                            backend_id=backend_id,
                            executable={"region": region_id},
                            dtype="float32",
                        )
                    )
                self.events.record(f"done:{region_id}")
                with lock:
                    results[region_id] = out
                self.telemetry.emit("end", region=region_id, device=device)
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        # Launch regions with empty depends_on concurrently; serialize the rest.
        pending = list(self.plan.placements)
        completed: set[str] = set()
        while pending:
            launchable = [
                p
                for p in pending
                if set(p.depends_on).issubset(completed) or (not p.depends_on and len(self.plan.devices_used) > 1)
            ]
            if not launchable:
                # Fall back to sequential to avoid deadlock on incomplete metadata.
                launchable = [pending[0]]
            batch = []
            used_devices: set[str] = set()
            for p in launchable:
                if p.device in used_devices:
                    continue
                used_devices.add(p.device)
                batch.append(p)
            threads = []
            for p in batch:
                t = threading.Thread(target=run_placement, args=(p.region_id, p.device, p.backend_id))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise RuntimePlanError(str(errors[0])) from errors[0]
            for p in batch:
                completed.add(p.region_id)
                pending.remove(p)

        return {
            "results": results,
            "telemetry": self.telemetry.events,
            "allocator_used": self.allocator.used(),
            "storage_tiers": [
                m.id.name
                for m in self.machine.memory.values()
                if m.memory_class in (MemoryClass.NVME, MemoryClass.DISK_CACHE)
            ],
        }


class Coordinator:
    def __init__(self, specialized: SpecializedArtifact, machine: ResourceGraph) -> None:
        self.specialized = specialized
        self.machine = machine
        self.executor = PlanExecutor(specialized.plan, machine)

    def execute(self) -> dict[str, Any]:
        if specialized_fingerprint_mismatch(self.specialized, self.machine):
            raise RuntimePlanError(
                "Specialized artifact fingerprint does not match this machine; run `streamcompiler autotune` again."
            )
        return self.executor.run()


def specialized_fingerprint_mismatch(artifact: SpecializedArtifact, machine: ResourceGraph) -> bool:
    return bool(artifact.fingerprint and machine.fingerprint and artifact.fingerprint != machine.fingerprint)
