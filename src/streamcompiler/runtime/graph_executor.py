"""Schedule-driven region executor.

``ExecutableSchedule`` is the exclusive runtime program. ``GraphExecutor`` owns
bindings, parameter store, and process workers, then delegates every run to
:class:`~streamcompiler.runtime.schedule_executor.ScheduleExecutor`.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from streamcompiler.backends.torch_device import coerce_region_result, unwrap_region_callable
from streamcompiler.codegen.regions import RegionBinding, RegionProgram
from streamcompiler.errors import RuntimePlanError
from streamcompiler.runtime.allocation_pool import ActivationAllocator
from streamcompiler.runtime.schedule import ExecutableSchedule
from streamcompiler.runtime.tensor_directory import TensorDirectory
from streamcompiler.runtime.tensor_store import ParameterStore

# Fork workers inherit this table; keyed by executor instance id.
_FORK_REGION_CALLABLES: dict[int, dict[str, Any]] = {}
_FORK_EXECUTOR_IDS = itertools.count(1)


def _fork_run_region(
    registry_id: int,
    region_id: str,
    device: str,
    backend_id: str,
    args: tuple[Any, ...],
) -> tuple[RegionEvent, tuple[Any, ...]]:
    start = time.perf_counter()
    call = _FORK_REGION_CALLABLES[registry_id][region_id]
    result = call(*args)
    outputs = coerce_region_result(result)
    end = time.perf_counter()
    return (
        RegionEvent(
            region_id=region_id,
            device=device,
            backend_id=backend_id,
            start_s=start,
            end_s=end,
            worker=f"proc-{os.getpid()}",
        ),
        outputs,
    )


@dataclass
class RegionEvent:
    region_id: str
    device: str
    backend_id: str
    start_s: float
    end_s: float
    worker: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class ExecutionReport:
    wall_time_s: float
    events: list[RegionEvent] = field(default_factory=list)
    peak_activation_bytes: int = 0
    released_values: int = 0
    parallel_overlaps: int = 0
    max_concurrent_regions: int = 1
    parameter_store: dict[str, Any] = field(default_factory=dict)
    copy_snapshot: dict[str, Any] = field(default_factory=dict)
    instruction_ids: list[str] = field(default_factory=list)

    def overlapping_pairs(self) -> list[tuple[str, str]]:
        """Region pairs whose execution intervals genuinely overlapped in time."""
        pairs: list[tuple[str, str]] = []
        ordered = sorted(self.events, key=lambda e: e.start_s)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                if second.start_s >= first.end_s:
                    break
                pairs.append((first.region_id, second.region_id))
        return pairs

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_time_s": self.wall_time_s,
            "region_count": len(self.events),
            "peak_activation_bytes": self.peak_activation_bytes,
            "released_values": self.released_values,
            "parallel_overlaps": self.parallel_overlaps,
            "max_concurrent_regions": self.max_concurrent_regions,
            "parameter_store": self.parameter_store,
            "copy_snapshot": self.copy_snapshot,
            "instruction_ids": list(self.instruction_ids),
            "regions": [
                {
                    "region_id": e.region_id,
                    "device": e.device,
                    "backend_id": e.backend_id,
                    "duration_s": e.duration_s,
                    "worker": e.worker,
                }
                for e in self.events
            ],
        }


class GraphExecutor:
    """Executes a :class:`RegionProgram` exclusively through ``ExecutableSchedule``."""

    def __init__(
        self,
        program: RegionProgram,
        bindings: dict[str, RegionBinding],
        *,
        parameter_store: ParameterStore,
        max_workers: int = 1,
        prefetch_distance: int = 1,
        intraop_threads: int = 0,
        activation_budget_bytes: int | None = None,
        schedule: ExecutableSchedule | None = None,
        tensor_directory: TensorDirectory | None = None,
        buffer_reuse_assignment: dict[str, int] | None = None,
        allow_activation_spill: bool = True,
        activation_overflow_policy: str = "spill",
        process_workers: int = 0,
    ) -> None:
        missing = [r.region_id for r in program.regions if r.region_id not in bindings]
        if missing:
            raise RuntimePlanError(f"No compiled executable for regions: {missing}")
        self.program = program
        self.bindings = bindings
        self.parameter_store = parameter_store
        self.max_workers = max(1, int(max_workers))
        self.prefetch_distance = max(0, int(prefetch_distance))
        self.intraop_threads = max(0, int(intraop_threads))
        self.activation_budget_bytes = activation_budget_bytes
        # Kept for API/telemetry compatibility; schedule path owns residency.
        self.tensor_directory = tensor_directory if tensor_directory is not None else TensorDirectory()
        self._reuse_assignment = dict(buffer_reuse_assignment or {})
        self._allow_activation_spill = bool(allow_activation_spill) and activation_budget_bytes is not None
        self._activation_overflow_policy = (
            activation_overflow_policy if activation_overflow_policy in {"spill", "recompute"} else "spill"
        )
        self._spill_events: list[dict[str, Any]] = []
        self._process_pool: Any = None
        self._fork_registry_id: int | None = None
        self._last_schedule_report: Any = None
        self._transfer_events: list[dict[str, Any]] = []
        self._cancel_requested = False
        self._run_lock = threading.Lock()
        self._prefetch_enabled = self.prefetch_distance > 0 and parameter_store.needs_prefetch
        self._callables = self._resolve_callables()
        self._allocator = ActivationAllocator() if self._reuse_assignment and self.max_workers == 1 else None

        if schedule is None:
            from streamcompiler.runtime.schedule import schedule_from_bindings

            streaming = bool(getattr(parameter_store, "needs_prefetch", False))
            schedule = schedule_from_bindings(
                program,
                bindings,
                streaming=streaming,
                prefetch_distance=self.prefetch_distance if streaming else 0,
            )
        self.schedule = schedule
        from streamcompiler.runtime.schedule import ScheduleValidationError, validate_schedule

        violations = validate_schedule(schedule)
        if violations:
            raise RuntimePlanError(
                f"ExecutableSchedule {schedule.graph_name!r} failed validation: {violations}"
            ) from ScheduleValidationError(str(violations))
        self._schedule_driven = True
        self._static_order = tuple(program.regions)  # introspection only; deps decide order
        self._fast = None
        self._static_resident = None

        self._init_process_workers(int(process_workers))
        from streamcompiler.runtime.schedule_executor import ScheduleExecutor

        # Streaming budgets cannot pin every region's state at once — limit inflight
        # so Load/Compute/Evict double-buffer instead of stampeding the pack cache.
        inflight = 2 if getattr(parameter_store, "needs_prefetch", False) else max(8, self.max_workers * 2)
        self._schedule_executor: ScheduleExecutor | None = ScheduleExecutor(
            program,
            bindings,
            schedule,
            parameter_store=parameter_store,
            max_inflight=inflight,
            process_pool=self._process_pool,
            fork_registry_id=self._fork_registry_id,
            callables=self._callables,
            allocator=self._allocator,
            activation_budget_bytes=self.activation_budget_bytes,
            allow_activation_spill=self._allow_activation_spill,
            spill_events=self._spill_events,
            reuse_assignment=self._reuse_assignment,
        )

    def _init_process_workers(self, process_workers: int) -> None:
        """Attach a fork process pool when requested (Linux) for concurrent regions."""
        if process_workers <= 0 or self.max_workers <= 1:
            return
        if sys.platform != "linux":
            return
        from streamcompiler.runtime.process_workers import ProcessWorkerPool

        self._fork_registry_id = next(_FORK_EXECUTOR_IDS)
        _FORK_REGION_CALLABLES[self._fork_registry_id] = dict(self._callables)
        self._process_pool = ProcessWorkerPool(
            max_workers=min(process_workers, self.max_workers),
            start_method="fork",
            warm_up=True,
        )

    def close(self) -> None:
        sched = self._schedule_executor
        self._schedule_executor = None
        if sched is not None:
            sched.close()
        pool = self._process_pool
        self._process_pool = None
        if pool is not None:
            pool.shutdown(wait=True)
        if self._fork_registry_id is not None:
            _FORK_REGION_CALLABLES.pop(self._fork_registry_id, None)
            self._fork_registry_id = None

    @property
    def closed(self) -> bool:
        return self._schedule_executor is None

    @property
    def uses_fast_path(self) -> bool:
        """Always False: fast path removed; schedule is exclusive."""
        return False

    @property
    def uses_schedule_path(self) -> bool:
        """True when ``ExecutableSchedule`` is the exclusive runtime program."""
        return self._schedule_executor is not None

    @property
    def uses_static_resident(self) -> bool:
        """Always False: static-resident walk removed; schedule is exclusive."""
        return False

    def _resolve_callables(self) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for region_id, binding in self.bindings.items():
            exe = getattr(binding.compiled, "executable", binding.compiled)
            resolved[region_id] = unwrap_region_callable(exe)
        return resolved

    def request_cancel(self) -> None:
        self._cancel_requested = True
        if self._schedule_executor is not None:
            self._schedule_executor.request_cancel()

    def run(self, flat_inputs: list[Any]) -> tuple[list[Any], ExecutionReport]:
        if self._schedule_executor is None:
            raise RuntimePlanError("GraphExecutor is closed")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimePlanError(
                "GraphExecutor.run is not reentrant; serialize callers or compile one module per thread"
            )
        try:
            restore_threads: int | None = None
            if self.intraop_threads > 0:
                restore_threads = torch.get_num_threads()
                torch.set_num_threads(self.intraop_threads)
            try:
                return self._run_via_schedule(flat_inputs)
            finally:
                if restore_threads is not None:
                    torch.set_num_threads(restore_threads)
        finally:
            self._run_lock.release()

    def _run_via_schedule(self, flat_inputs: list[Any]) -> tuple[list[Any], ExecutionReport]:
        """Execute exclusively through the instruction-DAG ScheduleExecutor."""
        assert self._schedule_executor is not None
        self._cancel_requested = False
        self._transfer_events.clear()
        self._spill_events.clear()
        outputs, sreport = self._schedule_executor.run(flat_inputs)
        self._last_schedule_report = sreport
        region_events: list[RegionEvent] = []
        for ev in sreport.events:
            if ev.opcode != "Compute":
                continue
            region_id = ev.name.removeprefix("compute::")
            binding = self.bindings.get(region_id)
            region_events.append(
                RegionEvent(
                    region_id=region_id,
                    device=ev.resource,
                    backend_id=binding.backend_id if binding is not None else "",
                    start_s=ev.start_s,
                    end_s=ev.end_s,
                    worker="schedule",
                )
            )
        for ev in sreport.events:
            if ev.opcode in {"Transfer", "Prefetch", "Load", "RecordEvent", "WaitEvent"}:
                self._transfer_events.append(
                    {
                        "event": ev.opcode.lower(),
                        "name": ev.name,
                        "resource": ev.resource,
                        "duration_s": ev.duration_s,
                        "nbytes": ev.nbytes,
                        "notes": ev.notes,
                        "enqueue_start_s": ev.enqueue_start_s,
                        "enqueue_end_s": ev.enqueue_end_s,
                        "complete_s": ev.complete_s,
                        "exposed_stall_s": ev.exposed_stall_s,
                        "prefetch_hit": ev.prefetch_hit,
                        "simulated": ev.simulated,
                    }
                )
        stats = dict(sreport.parameter_store) if isinstance(sreport.parameter_store, dict) else {}
        stats["schedule_driven"] = True
        stats["schedule_report"] = sreport.as_dict()
        if getattr(sreport, "multi_copy_peaks", None):
            stats["multi_copy_peaks"] = list(sreport.multi_copy_peaks)
        return outputs, ExecutionReport(
            wall_time_s=sreport.wall_time_s,
            events=region_events,
            peak_activation_bytes=int(getattr(sreport, "peak_activation_bytes", 0) or 0),
            released_values=sum(1 for e in sreport.events if e.opcode == "Release"),
            parallel_overlaps=sreport.parallel_overlaps,
            max_concurrent_regions=sreport.max_concurrent,
            parameter_store=stats,
            copy_snapshot=dict(getattr(sreport, "copy_snapshot", {}) or {}),
            instruction_ids=[e.name for e in sreport.events],
        )
