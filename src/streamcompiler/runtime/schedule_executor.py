"""Instruction-DAG executor: ExecutableSchedule is the exclusive runtime program.

Every Prefetch / Load / Transfer / RecordEvent / WaitEvent / Compute / Evict /
Release op is dispatched when its ``depends_on`` instructions have completed.
Independent instructions may overlap; compute order need not match region order.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import torch

from streamcompiler.backends.torch_device import coerce_region_result
from streamcompiler.codegen.regions import RegionBinding, RegionProgram
from streamcompiler.errors import ExecutionCancelled, MemoryCapacityError, RuntimePlanError, StorageError
from streamcompiler.ir.graph import OpCode
from streamcompiler.runtime.copies import CopyStore
from streamcompiler.runtime.execution_context import ExecutionContext
from streamcompiler.runtime.schedule import ExecutableSchedule, PlanInstruction
from streamcompiler.runtime.streams import DeviceStreams, StreamEvent
from streamcompiler.runtime.tensor_store import ParameterStore
from streamcompiler.runtime.transfers import select_transfer_backend


@dataclass
class InstructionEvent:
    name: str
    opcode: str
    resource: str
    submitted_s: float
    start_s: float
    end_s: float
    nbytes: int = 0
    notes: str = ""
    prefetch_hit: bool | None = None
    exposed_stall_s: float = 0.0
    enqueue_start_s: float = 0.0
    enqueue_end_s: float = 0.0
    complete_s: float = 0.0
    consumer_wait_s: float = 0.0
    simulated: bool = False
    region_id: str | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def max_concurrency_from_intervals(intervals: list[tuple[float, float]]) -> int:
    """Peak concurrency via sweep over half-open ``[start, end)`` intervals."""
    points: list[tuple[float, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        points.append((start, 1))
        points.append((end, -1))
    # Ends (-1) before starts (+1) at the same timestamp.
    points.sort(key=lambda p: (p[0], p[1]))
    cur = peak = 0
    for _, delta in points:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


@dataclass
class ScheduleReport:
    wall_time_s: float
    events: list[InstructionEvent] = field(default_factory=list)
    parallel_overlaps: int = 0
    max_concurrent: int = 1
    copy_snapshot: dict[str, Any] = field(default_factory=dict)
    parameter_store: dict[str, Any] = field(default_factory=dict)
    multi_copy_peaks: list[dict[str, Any]] = field(default_factory=list)
    peak_activation_bytes: int = 0
    activation_bytes_written: int = 0
    activation_bytes_read: int = 0
    spill_latency_s: float = 0.0
    reload_latency_s: float = 0.0
    allocation_peak_bytes: int = 0

    def overlapping_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        ordered = sorted(self.events, key=lambda e: e.start_s)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                if second.start_s >= first.end_s:
                    break
                if (
                    first.opcode == "Compute"
                    and second.opcode == "Compute"
                    or {first.opcode, second.opcode} & {"Transfer", "Compute", "Prefetch", "Load"}
                ):
                    pairs.append((first.name, second.name))
        return pairs

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_time_s": self.wall_time_s,
            "instruction_count": len(self.events),
            "parallel_overlaps": self.parallel_overlaps,
            "max_concurrent": self.max_concurrent,
            "copy_snapshot": self.copy_snapshot,
            "multi_copy_peaks": self.multi_copy_peaks,
            "peak_activation_bytes": self.peak_activation_bytes,
            "activation_bytes_written": self.activation_bytes_written,
            "activation_bytes_read": self.activation_bytes_read,
            "spill_latency_s": self.spill_latency_s,
            "reload_latency_s": self.reload_latency_s,
            "allocation_peak_bytes": self.allocation_peak_bytes,
            "parameter_store": self.parameter_store,
            "instructions": [
                {
                    "name": e.name,
                    "opcode": e.opcode,
                    "resource": e.resource,
                    "duration_s": e.duration_s,
                    "nbytes": e.nbytes,
                    "prefetch_hit": e.prefetch_hit,
                    "exposed_stall_s": e.exposed_stall_s,
                    "notes": e.notes,
                }
                for e in self.events
            ],
        }


class ScheduleExecutor:
    """Runs an :class:`ExecutableSchedule` as an instruction dependency DAG."""

    def __init__(
        self,
        program: RegionProgram,
        bindings: dict[str, RegionBinding],
        schedule: ExecutableSchedule,
        *,
        parameter_store: ParameterStore,
        streams: DeviceStreams | None = None,
        max_inflight: int = 8,
        process_pool: Any | None = None,
        fork_registry_id: int | None = None,
        callables: dict[str, Any] | None = None,
        allocator: Any | None = None,
        activation_budget_bytes: int | None = None,
        spill_events: list[dict[str, Any]] | None = None,
        reuse_assignment: dict[str, int] | None = None,
    ) -> None:
        from streamcompiler.runtime.schedule import ScheduleValidationError, ensure_explicit_streams, validate_schedule

        schedule = ensure_explicit_streams(schedule)
        violations = validate_schedule(schedule)
        if violations:
            raise RuntimePlanError(
                f"ExecutableSchedule {schedule.graph_name!r} failed validation: {violations}"
            ) from ScheduleValidationError(str(violations))
        self.program = program
        self.bindings = bindings
        self.schedule = schedule
        self.parameter_store = parameter_store
        self.streams = streams if streams is not None else DeviceStreams()
        self.max_inflight = max(1, int(max_inflight))
        self.process_pool = process_pool
        self.fork_registry_id = fork_registry_id
        self.allocator = allocator
        self.activation_budget_bytes = activation_budget_bytes
        self._spill_events = spill_events if spill_events is not None else []
        self._reuse_assignment = dict(reuse_assignment or {})
        # Last-run residency snapshot only; live copies live on ExecutionContext.
        self.copies = CopyStore()
        self._by_name = {i.name: i for i in schedule.instructions}
        self._dependents: dict[str, list[str]] = defaultdict(list)
        for inst in schedule.instructions:
            for dep in inst.depends_on:
                self._dependents[dep].append(inst.name)
        if callables is not None:
            self._callables = callables
        else:
            self._callables = {
                rid: getattr(binding.compiled, "executable", binding.compiled) for rid, binding in bindings.items()
            }
        self._run_lock = threading.Lock()
        self._cancel = False
        self._closed = False
        self._transfer_lock = threading.Lock()
        self._sync_pool = ThreadPoolExecutor(
            max_workers=max(4, self.max_inflight),
            thread_name_prefix="schedule-sync",
        )
        self._native_artifact: Any | None = None
        self._native_cancel: Any | None = None
        self._install_native_artifact(schedule)

    def _install_native_artifact(self, schedule: ExecutableSchedule) -> None:
        from streamcompiler.native import native_available, require_native

        if not native_available():
            self._native_artifact = None
            self._native_cancel = None
            return
        native = require_native()
        self._native_artifact = native.NativeCompiledArtifact.from_schedule(schedule)
        self._native_cancel = native.NativeCancelToken()

    def close(self) -> None:
        if self._closed:
            return
        with self._run_lock:
            if self._closed:
                return
            self._closed = True
            self._cancel = True
            self._sync_pool.shutdown(wait=True, cancel_futures=True)
            self.streams.shutdown(wait=True)

    def replace_schedule(self, schedule: ExecutableSchedule) -> None:
        """Install a new immutable schedule (e.g. attribute annotations for tests)."""
        from streamcompiler.runtime.schedule import ScheduleValidationError, ensure_explicit_streams, validate_schedule

        with self._run_lock:
            if self._closed:
                raise RuntimePlanError("ScheduleExecutor is closed")
            schedule = ensure_explicit_streams(schedule)
            violations = validate_schedule(schedule)
            if violations:
                raise RuntimePlanError(
                    f"ExecutableSchedule {schedule.graph_name!r} failed validation: {violations}"
                ) from ScheduleValidationError(str(violations))
            self.schedule = schedule
            self._by_name = {i.name: i for i in schedule.instructions}
            self._dependents = defaultdict(list)
            for inst in schedule.instructions:
                for dep in inst.depends_on:
                    self._dependents[dep].append(inst.name)
            self._install_native_artifact(schedule)

    def request_cancel(self) -> None:
        self._cancel = True
        if self._native_cancel is not None:
            self._native_cancel.cancel()

    def run(self, flat_inputs: list[Any]) -> tuple[list[Any], ScheduleReport]:
        if self._closed:
            raise RuntimePlanError("ScheduleExecutor is closed")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimePlanError("ScheduleExecutor.run is not reentrant")
        try:
            from streamcompiler.runtime.native_bridge import run_schedule_native, should_use_native_runtime

            if should_use_native_runtime():
                return run_schedule_native(self, flat_inputs)
            from streamcompiler.native import allow_python_runtime

            if not allow_python_runtime():
                from streamcompiler.native import require_native

                require_native()
            try:
                from streamcompiler.native import require_native as _rn

                _rn().record_python_fallback_enter()
            except Exception:
                pass
            return self._run_unlocked(flat_inputs)
        finally:
            self._run_lock.release()

    def _run_unlocked(self, flat_inputs: list[Any]) -> tuple[list[Any], ScheduleReport]:
        if self._closed:
            raise RuntimePlanError("ScheduleExecutor is closed")
        if self._cancel:
            self._cancel = False
            raise ExecutionCancelled("Schedule execution cancelled")
        self._cancel = False
        ctx = ExecutionContext(host_resource=self._default_host_resource())
        if self._cancel:
            ctx.cancellation.request()
        report = ScheduleReport(wall_time_s=0.0)
        events_by_name: dict[str, InstructionEvent] = {}
        completed: set[str] = set()
        remaining_deps: dict[str, set[str]] = {inst.name: set(inst.depends_on) for inst in self.schedule.instructions}
        ready: deque[str] = deque(name for name, deps in remaining_deps.items() if not deps)
        running: dict[Future[Any], str] = {}
        host = ctx.host_resource
        if len(flat_inputs) != len(self.program.user_inputs):
            raise RuntimePlanError(f"Expected {len(self.program.user_inputs)} inputs, got {len(flat_inputs)}")
        for name, value in zip(self.program.user_inputs, flat_inputs, strict=True):
            ctx.copies.put(name, host, value, tier="system_ram", authoritative=True, ownership="input")
            if host != "cpu":
                ctx.copies.alias(name, host, "cpu")
            if host != "host":
                ctx.copies.alias(name, host, "host")

        self.parameter_store.begin_execution()
        wall0 = time.perf_counter()

        def _finish(name: str, event: InstructionEvent) -> None:
            events_by_name[name] = event
            report.events.append(event)
            st = ctx.state_for(name)
            st.start_s = event.start_s
            st.completion_s = event.end_s
            st.result = event
            completed.add(name)
            for child in self._dependents.get(name, ()):
                remaining_deps[child].discard(name)
                if (
                    not remaining_deps[child]
                    and child not in completed
                    and child not in running.values()
                    and child not in ready
                ):
                    ready.append(child)

        while ready or running:
            if (self._cancel or ctx.cancellation.cancelled) and not running:
                self._cancel = False
                ctx.cancellation.clear()
                raise ExecutionCancelled("Schedule execution cancelled")
            while ready and len(running) < self.max_inflight and not (self._cancel or ctx.cancellation.cancelled):
                name = ready.popleft()
                if name in completed:
                    continue
                inst = self._by_name[name]
                submitted = time.perf_counter()
                ctx.state_for(name).submitted_s = submitted
                fut = self._dispatch(inst, ctx, submitted)
                running[fut] = name
                report.max_concurrent = max(report.max_concurrent, len(running))
            if not running:
                if self._cancel or ctx.cancellation.cancelled:
                    self._cancel = False
                    ctx.cancellation.clear()
                    raise ExecutionCancelled("Schedule execution cancelled")
                if ready:
                    continue
                break
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                name = running.pop(fut)
                try:
                    event = fut.result()
                except BaseException as exc:
                    ctx.state_for(name).exception = exc
                    raise
                _finish(name, event)
                self._assert_activation_budget(ctx, completed)
            if (self._cancel or ctx.cancellation.cancelled) and not running:
                self._cancel = False
                ctx.cancellation.clear()
                raise ExecutionCancelled("Schedule execution cancelled")

        missing = [i.name for i in self.schedule.instructions if i.name not in completed]
        if missing:
            raise RuntimePlanError(f"Schedule left unfinished instructions: {missing}")

        self._assert_activation_budget(ctx, completed)
        report.wall_time_s = time.perf_counter() - wall0
        report.parallel_overlaps = len(report.overlapping_pairs())
        report.copy_snapshot = ctx.copies.snapshot()
        report.multi_copy_peaks = list(ctx.telemetry.multi_copy_peaks)
        report.peak_activation_bytes = max(ctx.activation_peak_bytes, ctx.copies.activation_live_bytes())
        report.activation_bytes_written = ctx.telemetry.activation_bytes_written
        report.activation_bytes_read = ctx.telemetry.activation_bytes_read
        report.spill_latency_s = ctx.telemetry.spill_latency_s
        report.reload_latency_s = ctx.telemetry.reload_latency_s
        report.max_concurrent = max(
            report.max_concurrent,
            max_concurrency_from_intervals([(e.start_s, e.end_s) for e in report.events]),
        )
        report.allocation_peak_bytes = ctx.allocations.peak_bytes()
        if report.allocation_peak_bytes == 0 and report.peak_activation_bytes > 0:
            report.allocation_peak_bytes = report.peak_activation_bytes
        self.copies = ctx.copies
        self._spill_events.extend(ctx.telemetry.spill_events)
        compute_intervals = [(e.start_s, e.end_s) for e in report.events if e.opcode == "Compute"]
        if hasattr(self.parameter_store, "record_compute_intervals"):
            self.parameter_store.record_compute_intervals(compute_intervals)
        stats = self.parameter_store.stats()
        if isinstance(stats, dict):
            stats = dict(stats)
            stats["schedule_instruction_events"] = len(report.events)
            stats["schedule_driven"] = True
            stats["peak_activation_bytes"] = report.peak_activation_bytes
            stats["activation_bytes_written"] = report.activation_bytes_written
            stats["activation_bytes_read"] = report.activation_bytes_read
        report.parameter_store = stats if isinstance(stats, dict) else {}
        return self._collect_outputs(ctx), report

    def _default_host_resource(self) -> str:
        for binding in self.bindings.values():
            if "cpu" in binding.device or "numa" in binding.device:
                return binding.device
        return "cpu"

    def _protected_budget_tensors(self) -> set[str]:
        """Tensors the planner refuses to spill (inputs + graph outputs)."""
        protected: set[str] = set(self.program.user_inputs)
        for kind, ref in getattr(self.program, "output_refs", ()):
            if kind == "value":
                protected.add(str(ref))
        protected.update(getattr(self.program, "user_outputs", ()) or ())
        return protected

    def _pending_spill_tensors(self, completed: set[str]) -> set[str]:
        """Activation tensors still waiting on a scheduled spill Evict."""
        pending: set[str] = set()
        for inst in self.schedule.instructions:
            if inst.name in completed:
                continue
            if inst.opcode == OpCode.EVICT and inst.attributes.get("kind") == "activation_spill":
                pending.update(inst.inputs)
        return pending

    def _assert_activation_budget(self, ctx: ExecutionContext, completed: set[str]) -> None:
        """Fail when durable activation residency exceeds the configured budget.

        Matches planner semantics: disk spills and protected tensors may leave
        live bytes above budget only while a pending spill still covers every
        spillable resident activation. Transient overage between Compute and its
        dependent Evict is allowed; leftover spillable residency is not.
        """
        budget = self.activation_budget_bytes
        live = ctx.copies.activation_live_bytes()
        ctx.note_activation_live(live)
        if budget is None:
            return
        if live <= int(budget):
            return
        protected = self._protected_budget_tensors()
        pending = self._pending_spill_tensors(completed)
        spillable = sorted(
            tid for tid in ctx.copies.activation_tensor_ids() if tid not in protected and tid not in pending
        )
        if not spillable:
            return
        raise RuntimePlanError(f"activation budget {int(budget)} bytes exceeded: live={live} spillable={spillable}")

    def _dispatch(
        self,
        inst: PlanInstruction,
        ctx: ExecutionContext,
        submitted: float,
    ) -> Future[Any]:
        opcode = inst.opcode
        if opcode == OpCode.PREFETCH:
            return self._submit_sync(lambda: self._exec_prefetch(inst, ctx, submitted))
        if opcode == OpCode.LOAD:
            return self._submit_sync(lambda: self._exec_load(inst, ctx, submitted))
        if opcode == OpCode.TRANSFER:
            return self._submit_transfer(inst, ctx, submitted)
        if opcode == OpCode.RECORD_EVENT:
            return self._submit_sync(lambda: self._exec_record(inst, ctx, submitted))
        if opcode == OpCode.WAIT_EVENT:
            return self._submit_sync(lambda: self._exec_wait(inst, ctx, submitted))
        if opcode == OpCode.COMPUTE:
            return self._submit_compute(inst, ctx, submitted)
        if opcode == OpCode.RELEASE:
            return self._submit_sync(lambda: self._exec_release(inst, ctx, submitted))
        if opcode == OpCode.EVICT:
            return self._submit_sync(lambda: self._exec_evict(inst, ctx, submitted))
        raise RuntimePlanError(f"Unsupported schedule opcode {opcode}")

    def _submit_sync(self, fn: Any) -> Future[Any]:
        if self._closed:
            fut: Future[Any] = Future()
            fut.set_exception(RuntimePlanError("ScheduleExecutor is closed"))
            return fut
        return self._sync_pool.submit(fn)

    def _exec_prefetch(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        tensor_id = inst.inputs[0] if inst.inputs else ""
        hit = False
        nbytes = inst.nbytes
        if tensor_id and getattr(self.parameter_store, "needs_prefetch", False):
            names = self._state_env_names(inst)
            try:
                self.parameter_store.prefetch(tuple(names))
            except (StorageError, MemoryCapacityError, RuntimePlanError) as exc:
                end = time.perf_counter()
                return InstructionEvent(
                    name=inst.name,
                    opcode=inst.opcode.value,
                    resource=str(inst.resource),
                    submitted_s=submitted,
                    start_s=start,
                    end_s=end,
                    nbytes=nbytes,
                    prefetch_hit=False,
                    notes=f"schedule Prefetch skipped: {exc}",
                )
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            nbytes=nbytes,
            prefetch_hit=hit,
            notes="schedule Prefetch",
        )

    def _exec_load(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        kind = str(inst.attributes.get("kind") or "")
        if kind == "activation_reload":
            return self._exec_activation_reload(inst, ctx, submitted)

        stall0 = time.perf_counter()
        names = self._state_env_names(inst)
        nbytes = 0
        # Load always materializes into host-accessible RAM — never device VRAM.
        dest = str(inst.destination or inst.resource)
        if _tier_is_device(dest):
            dest = ctx.host_resource
        before: dict[str, Any] = {}
        store_stats = getattr(self.parameter_store, "stats", None)
        if callable(store_stats):
            before = dict(store_stats())
        tier = _copy_tier(inst.memory_tier)
        for env_name in names:
            tensor = self.parameter_store.acquire(env_name)
            if tier == "pinned_ram":
                tensor = _ensure_pinned(tensor)
            target = self.program.state_bindings.get(env_name, env_name)
            ctx.copies.put(env_name, dest, tensor, tier=tier, ownership="parameter")
            if dest != "cpu":
                ctx.copies.alias(env_name, dest, "cpu")
            if target != env_name:
                ctx.copies.put(target, dest, tensor, tier=tier, ownership="parameter")
                if dest != "cpu":
                    ctx.copies.alias(target, dest, "cpu")
            if isinstance(tensor, torch.Tensor):
                nbytes += int(tensor.numel() * tensor.element_size())
        stall = time.perf_counter() - stall0
        prefetch_hit: bool | None = None
        if before and callable(store_stats):
            after = dict(store_stats())
            hits_delta = int(after.get("prefetch_hits", 0) or 0) - int(before.get("prefetch_hits", 0) or 0)
            miss_delta = int(after.get("cache_misses", 0) or 0) - int(before.get("cache_misses", 0) or 0)
            cache_hits = int(after.get("cache_hits", 0) or 0) - int(before.get("cache_hits", 0) or 0)
            if hits_delta > 0 and miss_delta == 0:
                prefetch_hit = True
            elif miss_delta > 0:
                prefetch_hit = False
            elif cache_hits > 0 and stall < 1e-4:
                prefetch_hit = True
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            nbytes=nbytes or inst.nbytes,
            exposed_stall_s=stall,
            prefetch_hit=prefetch_hit,
            notes="schedule Load disk→host",
        )

    def _exec_activation_reload(
        self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float
    ) -> InstructionEvent:
        from streamcompiler.runtime.activation_spill import is_spilled, reload_spilled

        start = time.perf_counter()
        tensor_id = inst.inputs[0] if inst.inputs else ""
        dest = str(inst.destination or inst.resource)
        if _tier_is_device(dest):
            dest = ctx.host_resource
        copy = ctx.copies.require(tensor_id, "disk")
        if not is_spilled(copy.value):
            raise RuntimePlanError(f"activation_reload {inst.name}: disk copy of {tensor_id!r} is not a spilled handle")
        # Keep disk copy until Release so parallel consumers can share one spill.
        tensor = reload_spilled(copy.value, delete=False)
        tier = _copy_tier(inst.memory_tier)
        if tier == "pinned_ram":
            tensor = _ensure_pinned(tensor)
        nbytes = int(tensor.numel() * tensor.element_size()) if isinstance(tensor, torch.Tensor) else copy.nbytes
        if ctx.copies.has(tensor_id, dest, valid_only=True):
            ctx.copies.replace_handle(tensor_id, dest, tensor, tier=tier)
        else:
            ctx.copies.replicate(
                tensor_id,
                dest,
                tensor,
                tier=tier,
                ownership="activation",
                source_resource="disk",
            )
        if dest != "cpu" and not ctx.copies.has(tensor_id, "cpu", valid_only=True):
            ctx.copies.alias(tensor_id, dest, "cpu")
        latency = time.perf_counter() - start
        ctx.telemetry.record_reload(name=tensor_id, nbytes=nbytes, latency_s=latency, instruction=inst.name)
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=dest,
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            nbytes=nbytes,
            exposed_stall_s=latency,
            notes="schedule Load activation disk→host",
        )

    def _state_env_names(self, inst: PlanInstruction) -> list[str]:
        region_id = str(inst.attributes.get("region_id") or "")
        if region_id and region_id in self.bindings:
            return list(self.bindings[region_id].region.state_inputs)
        # Fall back: inputs named state::region or env names already.
        out: list[str] = []
        for raw in inst.inputs:
            if raw.startswith("state::"):
                rid = raw.split("::", 1)[1]
                if rid in self.bindings:
                    out.extend(self.bindings[rid].region.state_inputs)
            elif raw in self.program.state_bindings:
                out.append(raw)
        return out

    def _submit_transfer(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Future[Any]:
        tensor_id = inst.inputs[0] if inst.inputs else ""
        dest = str(inst.destination or inst.resource)
        src = str(inst.source or ctx.host_resource)
        key = (tensor_id, dest)
        st = ctx.state_for(inst.name)

        def _body() -> InstructionEvent:
            enqueue_start = time.perf_counter()
            existing_dest = ctx.copies.try_get(tensor_id, dest)
            if existing_dest is not None and not existing_dest.stale:
                ready = existing_dest.ready_event
                incomplete = False
                if ready is not None:
                    if hasattr(ready, "is_complete"):
                        incomplete = not bool(ready.is_complete())
                    elif hasattr(ready, "done"):
                        incomplete = not bool(ready.done())
                if not incomplete:
                    end = time.perf_counter()
                    return InstructionEvent(
                        name=inst.name,
                        opcode=inst.opcode.value,
                        resource=str(inst.resource),
                        submitted_s=submitted,
                        start_s=enqueue_start,
                        end_s=end,
                        nbytes=0,
                        notes="elided duplicate transfer (dest already resident)",
                        enqueue_start_s=enqueue_start,
                        enqueue_end_s=end,
                    )
            with self._transfer_lock:
                existing = ctx.pending_transfers.get(key)
            if existing is not None and not existing.done():
                st.async_future = existing
                st.enqueue_start_s = enqueue_start
                st.enqueue_end_s = time.perf_counter()
                end = time.perf_counter()
                return InstructionEvent(
                    name=inst.name,
                    opcode=inst.opcode.value,
                    resource=str(inst.resource),
                    submitted_s=submitted,
                    start_s=enqueue_start,
                    end_s=end,
                    nbytes=0,
                    notes="joined in-progress transfer (async)",
                    enqueue_start_s=enqueue_start,
                    enqueue_end_s=end,
                )

            src_resource = src
            try:
                src_copy = ctx.copies.require(tensor_id, src_resource)
            except RuntimePlanError as exc:
                raise RuntimePlanError(
                    f"Transfer {inst.name}: required source copy missing/stale "
                    f"tensor={tensor_id!r} source={src_resource!r}"
                ) from exc

            backend = select_transfer_backend(inst.transfer_backend, destination=dest)
            delay = float(inst.attributes.get("mock_transfer_delay_s", 0.0))
            is_mock = "mock" in dest.lower() or delay > 0 or backend.backend_id == "simulated_device"
            stream = self.streams.copy_stream(dest, delay_s=delay if is_mock else 0.0)
            tier = "device" if _tier_is_device(dest) else "system_ram"
            pending_event = StreamEvent(name=f"ready::{inst.name}", device=dest)
            ctx.copies.replicate(
                tensor_id,
                dest,
                src_copy.value,
                tier=tier,
                ownership=src_copy.ownership,
                source_resource=src_resource,
                ready_event=pending_event,
            )

            def _xfer() -> Any:
                out, result = backend.transfer(
                    src_copy.value,
                    source=src_resource,
                    destination=dest,
                    nbytes=inst.nbytes or src_copy.nbytes,
                )
                ctx.copies.replace_handle(tensor_id, dest, out, tier=tier, ready_event=None)
                resources = ctx.copies.resources_for(tensor_id, valid_only=True)
                if len(resources) > 1:
                    ctx.telemetry.multi_copy_peaks.append(
                        {
                            "tensor_id": tensor_id,
                            "resources": list(resources),
                            "at": "transfer_complete",
                        }
                    )
                with self._transfer_lock:
                    ctx.pending_transfers.pop(key, None)
                return result

            fut = stream.submit(_xfer, delay_s=delay if is_mock else 0.0)
            enqueue_end = time.perf_counter()
            pending_event.bind_future(
                fut,
                enqueue_start_s=enqueue_start,
                enqueue_end_s=enqueue_end,
            )
            with self._transfer_lock:
                ctx.pending_transfers[key] = fut
            st.async_future = fut
            st.enqueue_start_s = enqueue_start
            st.enqueue_end_s = enqueue_end
            st.completion_event = pending_event
            return InstructionEvent(
                name=inst.name,
                opcode=inst.opcode.value,
                resource=str(inst.resource),
                submitted_s=submitted,
                start_s=enqueue_start,
                end_s=enqueue_end,
                nbytes=inst.nbytes or src_copy.nbytes,
                notes="transfer enqueued (async)",
                simulated=is_mock,
                enqueue_start_s=enqueue_start,
                enqueue_end_s=enqueue_end,
            )

        return self._submit_sync(_body)

    def _exec_record(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        waits = str(inst.attributes.get("pairs_with_wait") or "")
        transfer_name = inst.depends_on[0] if inst.depends_on else ""
        event = StreamEvent(name=inst.name, device=str(inst.resource))
        if transfer_name:
            transfer_state = ctx.state_for(transfer_name)
            fut = transfer_state.async_future
            if isinstance(fut, Future):
                event.bind_future(
                    fut,
                    enqueue_start_s=float(transfer_state.enqueue_start_s or start),
                    enqueue_end_s=float(transfer_state.enqueue_end_s or start),
                )
            else:
                event.record()
        else:
            event.record()
        ctx.events.store(inst.name, event)
        ctx.state_for(inst.name).completion_event = event
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            notes=f"RecordEvent pairs_with={waits}",
            enqueue_start_s=event.enqueue_start_s,
            enqueue_end_s=event.enqueue_end_s,
            complete_s=event.complete_s or end,
        )

    def _exec_wait(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        waits_for = str(inst.attributes.get("waits_for") or (inst.depends_on[0] if inst.depends_on else ""))
        event = ctx.events.get(waits_for)
        wait0 = time.perf_counter()
        event.wait()
        wait_s = time.perf_counter() - wait0
        ctx.state_for(inst.name).wait_duration_s = wait_s
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            consumer_wait_s=wait_s,
            exposed_stall_s=wait_s,
            notes=f"WaitEvent on {waits_for}",
            complete_s=event.complete_s or end,
        )

    def _submit_compute(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Future[Any]:
        delay = float(inst.attributes.get("mock_compute_delay_s", 0.0))
        binding = self.bindings[str(inst.executable_ref or "")]
        resource = binding.device
        if delay <= 0 and "mock" in resource:
            delay = (
                float(binding.compiled.attributes.get("mock_delay_s", 0.05))
                if hasattr(binding.compiled, "attributes")
                else 0.05
            )
        # Async only when mock delay or process workers need a real stream/pool.
        if delay <= 0 and self.process_pool is None:
            out: Future[Any] = Future()
            try:
                out.set_result(self._exec_compute(inst, ctx, submitted))
            except BaseException as exc:
                out.set_exception(exc)
            return out

        stream = self.streams.compute_stream(
            resource,
            delay_s=delay if delay > 0 else 0.0,
            workers=max(1, self.max_inflight),
        )
        fut = stream.submit(lambda: self._exec_compute(inst, ctx, submitted), delay_s=delay if delay > 0 else 0.0)
        out2: Future[Any] = Future()

        def _done(f: Future[Any]) -> None:
            try:
                out2.set_result(f.result())
            except Exception as exc:
                out2.set_exception(exc)

        fut.add_done_callback(_done)
        return out2

    def _exec_compute(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
        region_id = str(inst.executable_ref or "")
        binding = self.bindings[region_id]
        region = binding.region
        resource = binding.device

        from streamcompiler.runtime.activation_spill import is_spilled
        from streamcompiler.runtime.virtual_tensor import unwrap_for_compute, wrap_virtual

        args: list[Any] = []
        leased: list[tuple[str, str]] = []
        for name in region.inputs:
            try:
                copy = ctx.copies.require(name, resource)
            except RuntimePlanError as exc:
                raise RuntimePlanError(
                    f"Compute {region_id} on {resource}: required copy of {name!r} missing/stale "
                    f"(schedule must Load/Transfer before Compute; no hidden materialization)"
                ) from exc
            if is_spilled(copy.value):
                raise RuntimePlanError(
                    f"Compute {region_id}: {name!r} still spilled on {resource!r}; "
                    f"schedule must emit activation_reload Load before Compute"
                )
            ctx.native_require(name, resource)
            ctx.copies.add_consumer(name, resource)
            leased.append((name, resource))
            args.append(unwrap_for_compute(copy.value, resource=resource))

        call = self._callables[region_id]

        if self.process_pool is not None and self.fork_registry_id is not None and "mock" not in resource:
            from streamcompiler.runtime.graph_executor import _fork_run_region

            def _detach_arg(value: Any) -> Any:
                if isinstance(value, torch.Tensor):
                    return value.detach()
                if isinstance(value, (tuple, list)):
                    return type(value)(_detach_arg(v) for v in value)
                if isinstance(value, dict):
                    return {k: _detach_arg(v) for k, v in value.items()}
                return value

            region_event, outputs = self.process_pool.submit(
                _fork_run_region,
                self.fork_registry_id,
                region_id,
                resource,
                binding.backend_id,
                tuple(_detach_arg(a) for a in args),
            ).result()
            try:
                for out_name, value in zip(region.outputs, outputs, strict=True):
                    ctx.copies.put(out_name, resource, value, ownership="activation")
                    ctx.mirror_native_put(out_name, resource, value)
                return InstructionEvent(
                    name=inst.name,
                    opcode=inst.opcode.value,
                    resource=resource,
                    submitted_s=submitted,
                    start_s=region_event.start_s,
                    end_s=region_event.end_s,
                    notes=f"Compute {region_id} (process)",
                    region_id=region_id,
                )
            finally:
                for tname, tres in leased:
                    with contextlib.suppress(Exception):
                        ctx.copies.release_consumer(tname, tres)

        start = time.perf_counter()
        try:
            if torch.is_inference_mode_enabled():
                result = call(*args)
            else:
                with torch.inference_mode():
                    result = call(*args)
            outputs = coerce_region_result(result)
            if len(outputs) != len(region.outputs):
                raise RuntimePlanError(
                    f"Region {region_id} produced {len(outputs)} values, expected {len(region.outputs)}"
                )
            for out_name, value in zip(region.outputs, outputs, strict=True):
                if self.allocator is not None and isinstance(value, torch.Tensor):
                    slot = self._reuse_assignment.get(out_name)
                    if slot is not None:
                        value = self.allocator.acquire(slot, out_name, value)
                if "mock" in resource:
                    value = wrap_virtual(value, resource)
                ctx.copies.put(out_name, resource, value, ownership="activation")
                ctx.mirror_native_put(out_name, resource, value)
            end = time.perf_counter()
            return InstructionEvent(
                name=inst.name,
                opcode=inst.opcode.value,
                resource=resource,
                submitted_s=submitted,
                start_s=start,
                end_s=end,
                notes=f"Compute {region_id}",
                region_id=region_id,
            )
        finally:
            for tname, tres in leased:
                with contextlib.suppress(Exception):
                    ctx.copies.release_consumer(tname, tres)

    def _exec_release(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        freed = 0
        for tensor_id in inst.inputs:
            resource = str(inst.attributes.get("release_resource") or inst.resource)
            if not ctx.copies.has(tensor_id, resource):
                # After schedule spill without reload, the live copy may be on disk.
                if ctx.copies.has(tensor_id, "disk"):
                    resource = "disk"
                else:
                    raise RuntimePlanError(
                        f"Release of unknown copy: tensor={tensor_id!r} resource={resource!r} "
                        f"(instruction={inst.name!r}; no silent drop-all fallback)"
                    )
            copy = ctx.copies.try_get(tensor_id, resource)
            if copy is not None and copy.active_consumers > 0:
                raise RuntimePlanError(
                    f"Release while active leases remain: tensor={tensor_id!r} "
                    f"resource={resource!r} leases={copy.active_consumers}"
                )
            if ctx.native_residency is not None and ctx.native_residency.session.has(tensor_id, resource):
                ctx.native_residency.release(tensor_id, resource)
            freed += ctx.copies.drop(tensor_id, resource)
            if tensor_id in self.program.state_bindings or tensor_id in self.program.state_bindings.values():
                self.parameter_store.release((tensor_id,))
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            nbytes=freed,
            notes="schedule Release",
        )

    def _exec_evict(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
        start = time.perf_counter()
        kind = str(inst.attributes.get("kind") or "")
        if kind == "activation_spill":
            return self._exec_activation_spill(inst, ctx, submitted)
        freed = 0
        for tensor_id in inst.inputs:
            resource = str(inst.destination or inst.resource)
            if not ctx.copies.has(tensor_id, resource):
                continue
            freed += ctx.copies.drop(tensor_id, resource)
            if ctx.native_residency is not None and ctx.native_residency.session.has(tensor_id, resource):
                ctx.native_residency.release(tensor_id, resource)
            if tensor_id in self.program.state_bindings:
                self.parameter_store.release((tensor_id,))
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            nbytes=freed,
            notes="schedule Evict",
        )

    def _exec_activation_spill(
        self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float
    ) -> InstructionEvent:
        from streamcompiler.runtime.activation_spill import is_spilled, spill_tensor

        start = time.perf_counter()
        freed = 0
        for tensor_id in inst.inputs:
            # Reload keeps the disk handle (shared consumers). A later spill must
            # still vacate host RAM — "disk already spilled" is not enough.
            resource = str(inst.attributes.get("spill_resource") or inst.source or inst.resource)
            copy = None
            for alt in (
                resource,
                ctx.host_resource,
                "cpu",
                "host",
                *ctx.copies.resources_for(tensor_id, valid_only=True),
            ):
                if alt == "disk":
                    continue
                cand = ctx.copies.try_get(tensor_id, alt)
                if cand is None or cand.stale:
                    continue
                from streamcompiler.runtime.virtual_tensor import VirtualDeviceTensor

                if isinstance(cand.value, (torch.Tensor, VirtualDeviceTensor)):
                    copy = cand
                    resource = alt
                    break
            if copy is None:
                disk = ctx.copies.try_get(tensor_id, "disk")
                if disk is not None and not disk.stale and is_spilled(disk.value):
                    # Fully spilled: no host/device tensor residency remains.
                    freed += int(disk.nbytes)
                    continue
                raise RuntimePlanError(
                    f"activation_spill {inst.name}: required copy of {tensor_id!r} missing on {resource!r}"
                )
            from streamcompiler.runtime.virtual_tensor import VirtualDeviceTensor

            spill_src = copy.value.to_host() if isinstance(copy.value, VirtualDeviceTensor) else copy.value
            spilled = spill_tensor(spill_src)
            with ctx.copies._lock:  # noqa: SLF001 — atomic relocate under residency lock
                src = ctx.copies._copies.get((tensor_id, resource))  # noqa: SLF001
                if (
                    src is None
                    or src.stale
                    or not isinstance(getattr(src, "value", None), (torch.Tensor, VirtualDeviceTensor))
                ):
                    # Find any remaining host/device tensor under the lock.
                    found: str | None = None
                    for (tid, rid), cand in list(ctx.copies._copies.items()):  # noqa: SLF001
                        if tid != tensor_id or rid == "disk" or cand.stale:
                            continue
                        if isinstance(cand.value, (torch.Tensor, VirtualDeviceTensor)):
                            found = rid
                            break
                    if found is None:
                        again = ctx.copies._copies.get((tensor_id, "disk"))  # noqa: SLF001
                        with contextlib.suppress(OSError):
                            spilled.path.unlink(missing_ok=True)
                        if again is not None and not again.stale and is_spilled(again.value):
                            freed += int(again.nbytes)
                            continue
                        raise RuntimePlanError(
                            f"activation_spill {inst.name}: lost copy of {tensor_id!r} during spill race"
                        )
                    resource = found
                    src = ctx.copies._copies.get((tensor_id, resource))  # noqa: SLF001
                # Drop every non-disk residency (aliases + multi-resource copies).
                for tid, rid in list(ctx.copies._copies.keys()):  # noqa: SLF001
                    if tid != tensor_id or rid == "disk":
                        continue
                    if rid == resource:
                        continue
                    alloc_id = ctx.copies._alloc_by_copy.pop((tid, rid), None)  # noqa: SLF001
                    ctx.copies._copies.pop((tid, rid), None)  # noqa: SLF001
                    if alloc_id is not None and ctx.copies._allocations is not None:  # noqa: SLF001
                        ctx.copies._allocations.release(alloc_id)  # noqa: SLF001
                src = ctx.copies._copies.pop((tensor_id, resource))  # noqa: SLF001
                src_alloc = ctx.copies._alloc_by_copy.pop((tensor_id, resource), src.allocation_id)  # noqa: SLF001
                if src_alloc is not None and ctx.copies._allocations is not None:  # noqa: SLF001
                    ctx.copies._allocations.release(src_alloc)  # noqa: SLF001
                # Replace any prior disk spill handle from an earlier spill+reload cycle.
                old_disk = ctx.copies._copies.pop((tensor_id, "disk"), None)  # noqa: SLF001
                old_disk_alloc = ctx.copies._alloc_by_copy.pop((tensor_id, "disk"), None)  # noqa: SLF001
                if old_disk_alloc is not None and ctx.copies._allocations is not None:  # noqa: SLF001
                    ctx.copies._allocations.release(old_disk_alloc)  # noqa: SLF001
                if old_disk is not None and is_spilled(old_disk.value):
                    with contextlib.suppress(OSError):
                        old_disk.value.path.unlink(missing_ok=True)
                ctx.copies._install(  # noqa: SLF001
                    tensor_id,
                    "disk",
                    spilled,
                    nbytes=spilled.nbytes,
                    tier="disk",
                    version=src.version,
                    authoritative=src.authoritative,
                    ownership=src.ownership if src.ownership == "activation" else "activation",
                    ready_event=None,
                    stale=False,
                )
            freed += spilled.nbytes
            latency = time.perf_counter() - start
            ctx.telemetry.record_spill(
                name=tensor_id,
                nbytes=spilled.nbytes,
                latency_s=latency,
                instruction=inst.name,
                path=str(spilled.path),
            )
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=str(inst.resource),
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            nbytes=freed,
            notes="schedule Evict activation RAM→disk",
        )

    def _collect_outputs(self, ctx: ExecutionContext) -> list[Any]:
        host = ctx.host_resource
        flat: list[Any] = []
        for kind, ref in self.program.output_refs:
            if kind != "value":
                flat.append(ref)
                continue
            name = str(ref)
            copy = ctx.copies.try_get(name, host)
            if copy is None or copy.stale:
                resources = ctx.copies.resources_for(name, valid_only=True)
                if not resources:
                    raise RuntimePlanError(f"Missing output {name}")
                copy = ctx.copies.get(name, resources[0])
            value = copy.value
            from streamcompiler.runtime.activation_spill import is_spilled
            from streamcompiler.runtime.virtual_tensor import VirtualDeviceTensor

            if is_spilled(value):
                raise RuntimePlanError(f"Output {name!r} still spilled; schedule must reload before collect")
            if isinstance(value, VirtualDeviceTensor):
                value = value.to_host()
            flat.append(value)
        return flat


def _tier_is_device(resource: str) -> bool:
    name = resource.lower()
    return any(tok in name for tok in ("mock", "cuda", "rocm", "gpu", "xpu", "mps", "vram"))


def _copy_tier(memory_tier: Any) -> str:
    value = getattr(memory_tier, "value", memory_tier)
    name = str(value or "system_ram").lower()
    if "pinned" in name:
        return "pinned_ram"
    if "disk" in name:
        return "disk"
    if "device" in name:
        return "device"
    return "system_ram"


def _ensure_pinned(value: Any) -> Any:
    """Page-lock a host tensor when CUDA pinning is available."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.is_pinned():
        return value
    if not torch.cuda.is_available():
        return value
    return value.pin_memory()
