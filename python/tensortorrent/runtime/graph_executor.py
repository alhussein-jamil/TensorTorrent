"""Region program executor with schedule and optional direct fast paths.

``ExecutableSchedule`` is the production program for residency, streaming,
training, and cancellation. When the plan is eligible, ``direct_path`` bypasses
dispatch for resident single-region or hetero dataflow cases.
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

from tensortorrent.backends.torch_device import coerce_region_result, unwrap_region_callable
from tensortorrent.compile.regions import RegionBinding, RegionProgram
from tensortorrent.errors import RuntimePlanError
from tensortorrent.runtime.allocation_pool import ActivationAllocator
from tensortorrent.runtime.schedule import ExecutableSchedule
from tensortorrent.runtime.tensor_store import ParameterStore

# Fork workers inherit this table; keyed by executor instance id.
_FORK_REGION_CALLABLES: dict[int, dict[str, Any]] = {}
_FORK_EXECUTOR_IDS = itertools.count(1)


def _direct_path_wanted(config: Any | None) -> bool:
    """Whether to attempt the eligible direct call.

    ``TT_DIRECT_PATH=0`` forces the schedule path; ``=1`` forces attempting
    direct. Otherwise ``CompileConfig.prefer_direct_path`` (default True).
    """
    env = os.environ.get("TT_DIRECT_PATH")
    if env == "0":
        return False
    if env == "1":
        return True
    if config is None:
        return True
    return bool(getattr(config, "prefer_direct_path", True))


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
    allocation_peak_bytes: int = 0
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
            "allocation_peak_bytes": self.allocation_peak_bytes,
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
    """Executes a :class:`RegionProgram` via schedule, or direct when eligible."""

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
        buffer_reuse_assignment: dict[str, int] | None = None,
        process_workers: int = 0,
        machine: Any | None = None,
        device_workers: Any | None = None,
        config: Any | None = None,
        enable_dataflow_direct_path: bool = False,
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
        # Capture ambient CPU thread budget before any per-run intra-op pinch so
        # overlapped CPU regions keep a full worker (see _run_dataflow_direct).
        self._region_pool_threads = max(1, torch.get_num_threads())
        self.activation_budget_bytes = activation_budget_bytes
        self._reuse_assignment = dict(buffer_reuse_assignment or {})
        self._dataflow_direct_path_enabled = bool(enable_dataflow_direct_path)
        # Spill/stall settings from CompileConfig — forwarded to native_bridge.
        self._config_spill_dir: str | None = getattr(config, "spill_dir", None)
        self._config_max_total_spill_bytes: int | None = getattr(config, "max_total_spill_bytes", None)
        self._config_stall_timeout_s: float = float(getattr(config, "stall_timeout_s", 300.0) or 300.0)
        from pathlib import Path as _Path

        _cache_dir = getattr(config, "cache_dir", None)
        self._config_cache_dir: Any = _Path(_cache_dir).expanduser() if _cache_dir is not None else None
        self._spill_events: list[dict[str, Any]] = []
        self._process_pool: Any = None
        self._fork_registry_id: int | None = None
        self._last_schedule_report: Any = None
        self._transfer_events: list[dict[str, Any]] = []
        self._cancel_requested = False
        self._closed = False
        from tensortorrent.runtime.inflight import InFlightGate

        self._gate = InFlightGate()
        self._report_lock = threading.Lock()
        self._thread_lock = threading.Lock()
        self._thread_owners = 0
        self._saved_threads: int | None = None
        self._prefetch_enabled = self.prefetch_distance > 0 and parameter_store.needs_prefetch
        self._callables = self._resolve_callables()
        self._allocator = ActivationAllocator() if self._reuse_assignment and self.max_workers == 1 else None

        if schedule is None:
            from tensortorrent.runtime.schedule import schedule_from_bindings

            streaming = bool(getattr(parameter_store, "needs_prefetch", False))
            schedule = schedule_from_bindings(
                program,
                bindings,
                streaming=streaming,
                prefetch_distance=self.prefetch_distance if streaming else 0,
            )
        self.schedule = schedule
        self._static_order = tuple(program.regions)  # introspection only; deps decide order

        self._init_process_workers(int(process_workers))
        from tensortorrent.runtime.schedule_executor import ScheduleExecutor

        # Streaming budgets cannot pin every region's state at once — limit inflight
        # so Load/Compute/Evict double-buffer instead of stampeding the pack cache.
        inflight = 2 if getattr(parameter_store, "needs_prefetch", False) else max(8, self.max_workers * 2)
        self.machine = machine
        # ScheduleExecutor validates + normalizes streams once (shared authority).
        self._schedule_executor: ScheduleExecutor | None = ScheduleExecutor(
            program,
            bindings,
            schedule,
            parameter_store=parameter_store,
            max_inflight=inflight,
            max_workers=self.max_workers,
            process_pool=self._process_pool,
            fork_registry_id=self._fork_registry_id,
            callables=self._callables,
            allocator=self._allocator,
            activation_budget_bytes=self.activation_budget_bytes,
            spill_events=self._spill_events,
            reuse_assignment=self._reuse_assignment,
            machine=machine,
            device_workers=device_workers,
            hoist_resident_parameters=not bool(getattr(config, "allow_training", False)),
        )
        self.schedule = self._schedule_executor.schedule

        # Eligible plans skip dispatch: single-region DirectPlan, or (when
        # enabled) resident multi-region DataflowDirectPlan. None → schedule.
        # TT_DIRECT_PATH=0/1 overrides CompileConfig.prefer_direct_path.
        from tensortorrent.runtime.direct_path import build_direct_plan

        self._direct_plan = build_direct_plan(self) if _direct_path_wanted(config) else None

    @property
    def direct_plan(self) -> Any:
        """The zero-overhead plan in use, or ``None`` when scheduling is required."""
        return self._direct_plan

    def _init_process_workers(self, process_workers: int) -> None:
        """Attach a CPU-only fork pool when requested (Linux)."""
        if process_workers <= 0 or self.max_workers <= 1:
            return
        if sys.platform != "linux":
            return
        accelerator_bindings = sorted(
            {
                f"{binding.backend_id}/{binding.device}"
                for binding in self.bindings.values()
                if binding.backend_id not in {"cpu", "cpu_numa"}
            }
        )
        if accelerator_bindings:
            raise RuntimePlanError(
                "process_workers uses Linux fork and is CPU-only; accelerator bindings are unsafe after fork: "
                + ", ".join(accelerator_bindings)
            )
        from tensortorrent.runtime.process_workers import ProcessWorkerPool

        self._fork_registry_id = next(_FORK_EXECUTOR_IDS)
        _FORK_REGION_CALLABLES[self._fork_registry_id] = dict(self._callables)
        self._process_pool = ProcessWorkerPool(
            max_workers=min(process_workers, self.max_workers),
            start_method="fork",
            warm_up=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        # Drain concurrent forwards before tearing down pools.
        self._gate.mark_closed_and_wait()
        if self._closed:
            return
        self._closed = True
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
        return self._closed or self._schedule_executor is None

    @property
    def uses_schedule_path(self) -> bool:
        """True when inference forwards use ExecutableSchedule (no direct plan)."""
        return self._direct_plan is None and self._schedule_executor is not None

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

    def run(
        self,
        flat_inputs: list[Any],
        *,
        cancel_token: Any | None = None,
        enable_grad: bool = False,
    ) -> tuple[list[Any], ExecutionReport]:
        if self._closed or self._schedule_executor is None:
            raise RuntimePlanError("GraphExecutor is closed")
        try:
            self._gate.enter()
        except RuntimeError as exc:
            raise RuntimePlanError("GraphExecutor is closed") from exc
        try:
            restore_threads = False
            # Autograd needs the resident schedule, and a cancel token means
            # the caller expects mid-forward cancellation the direct call
            # cannot offer. A prior request_cancel() likewise forces the
            # schedule path so ExecutionCancelled still fires.
            use_direct = (
                self._direct_plan is not None
                and not enable_grad
                and cancel_token is None
                and not self._cancel_requested
            )
            from tensortorrent.runtime.direct_path import DataflowDirectPlan

            # Dataflow overlap runs real CPU work on a pool thread. Do not pinch
            # intra-op threads for that path — the pinch exists for fair shared
            # microbenchmarks of the schedule executor, not for the fast path.
            pinch_intraop = self.intraop_threads > 0 and not (
                use_direct and isinstance(self._direct_plan, DataflowDirectPlan)
            )
            if pinch_intraop:
                with self._thread_lock:
                    if self._thread_owners == 0:
                        self._saved_threads = torch.get_num_threads()
                        torch.set_num_threads(self.intraop_threads)
                    self._thread_owners += 1
                    restore_threads = True
            try:
                if use_direct:
                    return self._run_direct(flat_inputs)
                return self._run_via_schedule(flat_inputs, cancel_token=cancel_token, enable_grad=enable_grad)
            finally:
                if restore_threads:
                    with self._thread_lock:
                        self._thread_owners = max(0, self._thread_owners - 1)
                        if self._thread_owners == 0 and self._saved_threads is not None:
                            torch.set_num_threads(self._saved_threads)
                            self._saved_threads = None
        finally:
            self._gate.leave()

    def _run_direct(self, flat_inputs: list[Any]) -> tuple[list[Any], ExecutionReport]:
        """Call the single region directly, bypassing scheduler dispatch.

        Semantics match the scheduled path for this plan shape: the same
        callable on the same device with the same arguments. What is skipped is
        bookkeeping that only has meaning when there is more than one thing to
        order.
        """
        plan = self._direct_plan
        assert plan is not None
        from tensortorrent.runtime.direct_path import DataflowDirectPlan

        if isinstance(plan, DataflowDirectPlan):
            return self._run_dataflow_direct(plan, flat_inputs)
        start = time.perf_counter()
        outputs = plan.call(*plan.build_args(flat_inputs))
        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)
        wall = time.perf_counter() - start
        # Clear only after a completed direct call; pending cancel must survive
        # until the schedule path consumes it.
        self._cancel_requested = False

        by_name = dict(zip(plan.output_names, outputs, strict=False))
        flat_outputs = [by_name[ref[1]] for ref in self.program.output_refs]

        # The scheduled path derives these from the residency session. With one
        # region the same quantities are exactly the tensors involved, so they
        # are computed directly rather than approximated.
        def _nbytes(values: Any) -> int:
            total = 0
            for v in values:
                if hasattr(v, "numel") and hasattr(v, "element_size"):
                    total += int(v.numel() * v.element_size())
            return total

        activation_bytes = _nbytes(outputs)
        allocation_peak = plan.param_bytes + _nbytes(flat_inputs) + activation_bytes

        store_stats = dict(self.parameter_store.stats())
        store_stats["execution_path"] = "direct"
        store_stats["schedule_driven"] = False
        store_stats["peak_activation_bytes"] = activation_bytes

        report = ExecutionReport(
            wall_time_s=wall,
            events=[
                RegionEvent(
                    region_id=plan.region_id,
                    device=plan.device,
                    backend_id=self.bindings[plan.region_id].backend_id,
                    start_s=start,
                    end_s=start + wall,
                    worker="direct",
                )
            ],
            max_concurrent_regions=1,
            peak_activation_bytes=activation_bytes,
            allocation_peak_bytes=allocation_peak,
            parameter_store=store_stats,
            instruction_ids=[f"compute::{plan.region_id}"],
        )
        self._last_schedule_report = None
        return flat_outputs, report

    def _run_dataflow_direct(self, plan: Any, flat_inputs: list[Any]) -> tuple[list[Any], ExecutionReport]:
        """Execute precomputed resident dependency waves with minimal dispatch."""
        from concurrent.futures import wait

        from tensortorrent.backends.torch_device import coerce_region_result

        values = dict(zip(plan.user_inputs, flat_inputs, strict=True))
        region_events: list[RegionEvent] = []
        wall_start = time.perf_counter()
        schedule_executor = self._schedule_executor
        assert schedule_executor is not None
        plan.refresh_parameters()

        def run_region(region: Any, worker: str) -> tuple[dict[str, Any], RegionEvent]:
            args: list[Any] = []
            for is_value, slot in region.arg_plan:
                value = values[slot] if is_value else getattr(slot, "value", slot)
                if is_value and isinstance(value, torch.Tensor) and region.torch_device is not None:
                    target = torch.device(region.torch_device)
                    if value.device != target:
                        value = value.to(target)
                args.append(value)
            start = time.perf_counter()
            with torch.inference_mode():
                result = region.call(*args)
            outputs = coerce_region_result(result)
            if len(outputs) != len(region.output_names):
                raise RuntimePlanError(
                    f"Region {region.region_id} produced {len(outputs)} values, expected {len(region.output_names)}"
                )
            end = time.perf_counter()
            return (
                dict(zip(region.output_names, outputs, strict=True)),
                RegionEvent(
                    region_id=region.region_id,
                    device=region.device,
                    backend_id=self.bindings[region.region_id].backend_id,
                    start_s=start,
                    end_s=end,
                    worker=worker,
                ),
            )

        for wave_idx, wave in enumerate(plan.waves):
            if len(wave) == 1:
                produced, event = run_region(wave[0], "direct")
                values.update(produced)
                region_events.append(event)
            else:
                # Keep accelerator launch on caller thread (CUDA contexts and eager
                # baselines do the same); start CPU siblings first so work overlaps.
                inline_index = next(
                    (
                        index
                        for index, region in enumerate(wave)
                        if any(token in region.device.lower() for token in ("cuda", "rocm", "gpu", "accel"))
                    ),
                    0,
                )
                pool = schedule_executor._ensure_region_pool(
                    min(len(wave) - 1, self.max_workers),
                    threads=self._region_pool_threads,
                )
                futures = [
                    pool.submit(run_region, region, "direct-pool")
                    for index, region in enumerate(wave)
                    if index != inline_index
                ]
                try:
                    produced, event = run_region(wave[inline_index], "direct")
                    values.update(produced)
                    region_events.append(event)
                    for future in futures:
                        produced, event = future.result()
                        values.update(produced)
                        region_events.append(event)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    wait(futures)
                    raise
            if wave_idx < len(plan.release_after_wave):
                for name in plan.release_after_wave[wave_idx]:
                    values.pop(name, None)

        wall = time.perf_counter() - wall_start
        self._cancel_requested = False
        flat_outputs = [values[str(ref)] for kind, ref in plan.output_refs if kind == "value"]

        def _nbytes(items: Any) -> int:
            return sum(
                int(value.numel() * value.element_size())
                for value in items
                if hasattr(value, "numel") and hasattr(value, "element_size")
            )

        activation_bytes = _nbytes(flat_outputs)
        store_stats = dict(self.parameter_store.stats())
        store_stats.update(
            {
                "execution_path": "direct_dataflow",
                "schedule_driven": False,
                "peak_activation_bytes": activation_bytes,
            }
        )
        report = ExecutionReport(
            wall_time_s=wall,
            events=region_events,
            max_concurrent_regions=max(len(wave) for wave in plan.waves),
            peak_activation_bytes=activation_bytes,
            allocation_peak_bytes=plan.param_bytes + _nbytes(flat_inputs) + activation_bytes,
            parameter_store=store_stats,
            instruction_ids=[f"compute::{event.region_id}" for event in region_events],
        )
        self._last_schedule_report = None
        return flat_outputs, report

    def _run_via_schedule(
        self,
        flat_inputs: list[Any],
        *,
        cancel_token: Any | None = None,
        enable_grad: bool = False,
    ) -> tuple[list[Any], ExecutionReport]:
        """Execute exclusively through the instruction-DAG ScheduleExecutor."""
        assert self._schedule_executor is not None
        try:
            outputs, sreport = self._schedule_executor.run(
                flat_inputs, cancel_token=cancel_token, enable_grad=enable_grad
            )
        finally:
            # Whether the schedule raised ExecutionCancelled or completed, the
            # pending GraphExecutor cancel has been observed — clear for next run.
            self._cancel_requested = False
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
        transfer_events: list[dict[str, Any]] = []
        for ev in sreport.events:
            if ev.opcode in {"Transfer", "Prefetch", "Load", "RecordEvent", "WaitEvent"}:
                transfer_events.append(
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
        spill_events = list(getattr(sreport, "spill_events", None) or [])
        with self._report_lock:
            self._last_schedule_report = sreport
            self._transfer_events = transfer_events
            self._spill_events = spill_events
        stats = dict(sreport.parameter_store) if isinstance(sreport.parameter_store, dict) else {}
        stats["schedule_driven"] = True
        stats["schedule_report"] = sreport.as_dict()
        if getattr(sreport, "multi_copy_peaks", None):
            stats["multi_copy_peaks"] = list(sreport.multi_copy_peaks)
        return outputs, ExecutionReport(
            wall_time_s=sreport.wall_time_s,
            events=region_events,
            peak_activation_bytes=int(getattr(sreport, "peak_activation_bytes", 0) or 0),
            allocation_peak_bytes=int(getattr(sreport, "allocation_peak_bytes", 0) or 0),
            released_values=sum(1 for e in sreport.events if e.opcode == "Release"),
            parallel_overlaps=sreport.parallel_overlaps,
            max_concurrent_regions=sreport.max_concurrent,
            parameter_store=stats,
            copy_snapshot=dict(getattr(sreport, "copy_snapshot", {}) or {}),
            instruction_ids=[e.name for e in sreport.events],
        )
