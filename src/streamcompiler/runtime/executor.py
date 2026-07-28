"""Shared runtime services.

Region scheduling lives in :mod:`streamcompiler.runtime.graph_executor`. This
module holds the machine-level services that scheduler builds on: the tensor
directory, the tiered allocator, event pools, block I/O and collectives.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from streamcompiler.compile.pipeline import SpecializedArtifact

from streamcompiler.errors import RuntimePlanError
from streamcompiler.ir.resource_graph import ResourceGraph


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
    """Performs real block reads from a model pack or any backing file."""

    def read_block(self, path: str, offset: int, nbytes: int) -> bytes:
        fd = os.open(path, os.O_RDONLY)
        try:
            data = os.pread(fd, nbytes, offset)
        finally:
            os.close(fd)
        if len(data) != nbytes:
            raise RuntimePlanError(f"Short read from {path}: requested {nbytes} bytes at {offset}, got {len(data)}")
        return data

    def prefetch(self, path: str, offset: int, nbytes: int) -> dict[str, Any]:
        """Read a block now and report what was actually read."""
        data = self.read_block(path, offset, nbytes)
        return {"path": path, "offset": offset, "nbytes": len(data), "status": "read"}


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


def specialized_fingerprint_mismatch(artifact: SpecializedArtifact, machine: ResourceGraph) -> bool:
    """True when a cached artifact was specialized for a different machine."""
    return bool(artifact.fingerprint and machine.fingerprint and artifact.fingerprint != machine.fingerprint)
