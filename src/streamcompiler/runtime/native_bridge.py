"""Bridge: Rust schedules instructions; Python executes tensor-bearing ops."""

from __future__ import annotations

import contextlib
import time
from typing import Any

from streamcompiler.errors import ExecutionCancelled, RuntimePlanError
from streamcompiler.ir.graph import OpCode
from streamcompiler.native import allow_python_runtime, native_available, require_native
from streamcompiler.runtime.execution_context import ExecutionContext
from streamcompiler.runtime.schedule import PlanInstruction
from streamcompiler.runtime.schedule_executor import InstructionEvent, ScheduleReport, max_concurrency_from_intervals


def _exec_inline(executor: Any, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
    """Run one instruction body without an extra Python thread-pool hop."""
    opcode = inst.opcode
    if opcode == OpCode.PREFETCH:
        event = executor._exec_prefetch(inst, ctx, submitted)
    elif opcode == OpCode.LOAD:
        event = executor._exec_load(inst, ctx, submitted)
    elif opcode == OpCode.TRANSFER:
        event = executor._submit_transfer(inst, ctx, submitted).result()
    elif opcode == OpCode.RECORD_EVENT:
        event = executor._exec_record(inst, ctx, submitted)
    elif opcode == OpCode.WAIT_EVENT:
        event = executor._exec_wait(inst, ctx, submitted)
    elif opcode == OpCode.COMPUTE:
        # Sync compute for CPU; stream path preserves mock-delay overlap semantics.
        delay = float(inst.attributes.get("mock_compute_delay_s", 0.0))
        if delay > 0.0 or "mock" in str(inst.resource):
            event = executor._submit_compute(inst, ctx, submitted).result()
        else:
            event = executor._exec_compute(inst, ctx, submitted)
    elif opcode == OpCode.RELEASE:
        event = executor._exec_release(inst, ctx, submitted)
    elif opcode == OpCode.EVICT:
        event = executor._exec_evict(inst, ctx, submitted)
    else:
        raise RuntimePlanError(f"Unsupported schedule opcode {opcode}")
    assert isinstance(event, InstructionEvent)
    return event


def _schedule_allows_native_data_plane(executor: Any) -> bool:
    """True when non-compute ops need no Python tensor bodies mid-schedule.

    Loads are prematerialized into CopyStore before the native run; Compute is
    the only Python re-entry. Streaming stores, transfers, activation spill,
    and mock-delay overlap still use the full instruction-callback path.
    """
    # Prematerializing every Load breaks bounded RAM streaming stores.
    store = executor.parameter_store
    if getattr(store, "needs_prefetch", False):
        return False
    if getattr(store, "ram_budget_bytes", None) not in (None, 0):
        return False
    for inst in executor.schedule.instructions:
        op = inst.opcode
        if op == OpCode.TRANSFER:
            return False
        if op == OpCode.PREFETCH:
            return False
        if op in (OpCode.LOAD, OpCode.EVICT):
            kind = str(inst.attributes.get("kind") or "")
            if kind in {"activation_reload", "activation_spill"}:
                return False
        if op == OpCode.COMPUTE:
            if float(inst.attributes.get("mock_compute_delay_s", 0.0)) > 0.0:
                return False
            if "mock" in str(inst.resource):
                return False
    return True


def _prematerialize_loads(executor: Any, ctx: ExecutionContext) -> list[InstructionEvent]:
    events: list[InstructionEvent] = []
    for inst in executor.schedule.instructions:
        if inst.opcode != OpCode.LOAD:
            continue
        submitted = time.perf_counter()
        event = executor._exec_load(inst, ctx, submitted)
        # Mirror parameter residency into Rust (authoritative metadata).
        dest = str(inst.destination or inst.resource)
        from streamcompiler.runtime.schedule_executor import _tier_is_device

        if _tier_is_device(dest):
            dest = ctx.host_resource
        for env_name in executor._state_env_names(inst):
            copy = ctx.copies.try_get(env_name, dest)
            if copy is not None:
                ctx.mirror_native_put(env_name, dest, copy.value, nbytes=int(copy.nbytes))
                for compute in executor.schedule.instructions:
                    if compute.opcode == OpCode.COMPUTE:
                        res = str(compute.resource)
                        if res != dest and ctx.native_residency is not None:
                            ctx.native_residency.mirror_alias(env_name, dest, res)
            target = executor.program.state_bindings.get(env_name, env_name)
            if target != env_name:
                tcopy = ctx.copies.try_get(target, dest)
                if tcopy is not None:
                    ctx.mirror_native_put(target, dest, tcopy.value, nbytes=int(tcopy.nbytes))
                    for compute in executor.schedule.instructions:
                        if compute.opcode == OpCode.COMPUTE:
                            res = str(compute.resource)
                            if res != dest and ctx.native_residency is not None:
                                ctx.native_residency.mirror_alias(target, dest, res)
        events.append(event)
    return events


def _sync_python_lifetime_ops(executor: Any, ctx: ExecutionContext, completed: set[str]) -> None:
    """Apply Release/Evict drops to Python CopyStore after a native region-path run.

    Rust already updated its residency metadata; Python still holds tensor values.
    """
    for inst in executor.schedule.instructions:
        if inst.name not in completed:
            continue
        if inst.opcode == OpCode.RELEASE:
            for tensor_id in inst.inputs:
                resource = str(inst.attributes.get("release_resource") or inst.resource)
                if not ctx.copies.has(tensor_id, resource):
                    if ctx.copies.has(tensor_id, "disk"):
                        resource = "disk"
                    else:
                        continue
                if ctx.native_residency is not None and ctx.native_residency.session.has(
                    tensor_id, resource
                ):
                    with contextlib.suppress(Exception):
                        ctx.native_residency.release(tensor_id, resource)
                with contextlib.suppress(Exception):
                    ctx.copies.drop(tensor_id, resource)
        elif inst.opcode == OpCode.EVICT:
            kind = str(inst.attributes.get("kind") or "")
            if kind == "activation_spill":
                continue  # spill needs full Python body; not on region path
            for tensor_id in inst.inputs:
                resource = str(inst.destination or inst.resource)
                if not ctx.copies.has(tensor_id, resource):
                    continue
                if ctx.native_residency is not None and ctx.native_residency.session.has(
                    tensor_id, resource
                ):
                    with contextlib.suppress(Exception):
                        ctx.native_residency.release(tensor_id, resource)
                with contextlib.suppress(Exception):
                    ctx.copies.drop(tensor_id, resource)


def _reraise_pending(executor: Any, pending_exc: list[BaseException], exc: Exception | None = None) -> None:
    if pending_exc:
        err = pending_exc[0]
        if isinstance(err, ExecutionCancelled):
            executor._cancel = False
            if executor._native_cancel is not None:
                executor._native_cancel.reset()
        if exc is not None:
            raise err from exc
        raise err
    if exc is None:
        return
    if isinstance(exc, (ExecutionCancelled, RuntimePlanError)):
        if isinstance(exc, ExecutionCancelled):
            executor._cancel = False
            if executor._native_cancel is not None:
                executor._native_cancel.reset()
        raise exc
    msg = str(exc)
    if "ExecutionCancelled" in msg or "cancelled" in msg.lower():
        executor._cancel = False
        if executor._native_cancel is not None:
            executor._native_cancel.reset()
        raise ExecutionCancelled("Schedule execution cancelled") from exc
    raise RuntimePlanError(f"native schedule execution failed: {exc}") from exc


def run_schedule_native(executor: Any, flat_inputs: list[Any]) -> tuple[list[Any], ScheduleReport]:
    """Run ``executor.schedule`` under the Rust dispatcher.

    Prefers a persistent :class:`NativeCompiledArtifact` on the executor so the
    Rust schedule is not rebuilt on every forward. When the schedule permits,
    Rust owns Load/Release/Record/Wait residency bookkeeping and Python is
    entered only for PyTorch region compute.
    """
    native = require_native()
    if executor._closed:
        raise RuntimePlanError("ScheduleExecutor is closed")
    if executor._cancel:
        executor._cancel = False
        if executor._native_cancel is not None:
            executor._native_cancel.reset()
        raise ExecutionCancelled("Schedule execution cancelled")
    executor._cancel = False
    if executor._native_cancel is not None:
        executor._native_cancel.reset()

    ctx = ExecutionContext(host_resource=executor._default_host_resource())
    report = ScheduleReport(wall_time_s=0.0)
    host = ctx.host_resource
    if len(flat_inputs) != len(executor.program.user_inputs):
        raise RuntimePlanError(f"Expected {len(executor.program.user_inputs)} inputs, got {len(flat_inputs)}")
    for name, value in zip(executor.program.user_inputs, flat_inputs, strict=True):
        ctx.copies.put(name, host, value, tier="system_ram", authoritative=True, ownership="input")
        if host != "cpu":
            ctx.copies.alias(name, host, "cpu")
        if host != "host":
            ctx.copies.alias(name, host, "host")

    executor.parameter_store.begin_execution()
    wall0 = time.perf_counter()
    completed: set[str] = set()
    events_by_name: dict[str, InstructionEvent] = {}
    pending_exc: list[BaseException] = []
    use_region_path = _schedule_allows_native_data_plane(executor)
    native_data_plane = False
    native_artifact_reused = False
    native_artifact_id: int | None = None
    native_report: dict[str, Any] = {}
    artifact = getattr(executor, "_native_artifact", None)
    cancel = getattr(executor, "_native_cancel", None)
    if use_region_path:
        from streamcompiler.runtime.handles import NativeResidencyBridge

        ctx.native_residency = NativeResidencyBridge.create()
        for name, value in zip(executor.program.user_inputs, flat_inputs, strict=True):
            ctx.mirror_native_put(name, host, value)
            if host != "cpu":
                ctx.native_residency.mirror_alias(name, host, "cpu")
            if host != "host":
                ctx.native_residency.mirror_alias(name, host, "host")
            # NUMA/compute pool ids used by the schedule must see the same handle.
            for inst in executor.schedule.instructions:
                if inst.opcode != OpCode.COMPUTE:
                    continue
                res = str(inst.resource)
                if res not in {host, "cpu", "host"}:
                    ctx.native_residency.mirror_alias(name, host, res)

        for event in _prematerialize_loads(executor, ctx):
            events_by_name[event.name] = event
            report.events.append(event)
            completed.add(event.name)

        def region_handler(region_id: str, _inputs: list[str], _outputs: list[str]) -> None:
            if executor._cancel or ctx.cancellation.cancelled:
                if executor._native_cancel is not None:
                    executor._native_cancel.cancel()
                raise ExecutionCancelled("Schedule execution cancelled")
            inst = None
            for candidate in executor.schedule.instructions:
                if candidate.opcode == OpCode.COMPUTE and str(candidate.executable_ref or "") == region_id:
                    inst = candidate
                    break
            if inst is None:
                raise RuntimePlanError(f"no Compute instruction for region {region_id!r}")
            submitted = time.perf_counter()
            ctx.state_for(inst.name).submitted_s = submitted
            try:
                event = executor._exec_compute(inst, ctx, submitted)
            except BaseException as exc:
                pending_exc.append(exc)
                raise
            events_by_name[inst.name] = event
            report.events.append(event)
            st = ctx.state_for(inst.name)
            st.start_s = event.start_s
            st.completion_s = event.end_s
            st.result = event
            completed.add(inst.name)
            executor._assert_activation_budget(ctx, completed)

        try:
            if artifact is not None:
                native_report = artifact.execute(
                    region_callback=region_handler,
                    dry_run=False,
                    cpu_workers=max(4, int(executor.max_inflight)),
                    cancel_token=cancel,
                )
                native_artifact_reused = True
                native_artifact_id = int(artifact.artifact_id)
            else:
                native_report = native.execute_schedule(
                    executor.schedule,
                    region_callback=region_handler,
                    dry_run=False,
                    cpu_workers=max(4, int(executor.max_inflight)),
                )
            native_data_plane = True
            for ev in native_report.get("events") or []:
                name = str(ev.get("name") or "")
                if name in events_by_name:
                    continue
                opcode = str(ev.get("opcode") or "")
                if opcode == "Compute":
                    continue
                event = InstructionEvent(
                    name=name,
                    opcode=opcode,
                    resource=str(ev.get("resource") or ""),
                    submitted_s=float(ev.get("submitted_s") or 0.0),
                    start_s=float(ev.get("start_s") or 0.0),
                    end_s=float(ev.get("end_s") or 0.0),
                    nbytes=int(ev.get("nbytes") or 0),
                    notes=str(ev.get("notes") or "native_data_plane"),
                    simulated=bool(ev.get("simulated")),
                )
                events_by_name[name] = event
                report.events.append(event)
                completed.add(name)
        except Exception as exc:
            if pending_exc:
                _reraise_pending(executor, pending_exc, exc)
            # Never silently restart after cancel — that re-runs Computes.
            if (
                executor._cancel
                or ctx.cancellation.cancelled
                or "cancel" in str(exc).lower()
                or type(exc).__name__ == "ExecutionCancelled"
            ):
                _reraise_pending(
                    executor,
                    [ExecutionCancelled("Schedule execution cancelled")],
                    exc,
                )
            # Fall back to full instruction-callback path for other region-path failures.
            use_region_path = False
            native_data_plane = False
            completed.clear()
            events_by_name.clear()
            report.events.clear()
            pending_exc.clear()
            ctx.native_residency = None
        else:
            # Region path: Rust Release/Evict only touch Rust residency — sync Python bags.
            _sync_python_lifetime_ops(executor, ctx, completed)

    if not native_data_plane:

        def handler(name: str) -> dict[str, Any]:
            if executor._cancel or ctx.cancellation.cancelled:
                if executor._native_cancel is not None:
                    executor._native_cancel.cancel()
                raise ExecutionCancelled("Schedule execution cancelled")
            inst = executor._by_name[name]
            submitted = time.perf_counter()
            ctx.state_for(name).submitted_s = submitted
            try:
                event = _exec_inline(executor, inst, ctx, submitted)
            except BaseException as exc:
                pending_exc.append(exc)
                raise
            events_by_name[name] = event
            report.events.append(event)
            st = ctx.state_for(name)
            st.start_s = event.start_s
            st.completion_s = event.end_s
            st.result = event
            completed.add(name)
            executor._assert_activation_budget(ctx, completed)
            return {
                "nbytes": int(event.nbytes),
                "simulated": bool(event.simulated),
                "notes": str(event.notes or ""),
            }

        try:
            if artifact is not None:
                native_report = artifact.execute(
                    instruction_handler=handler,
                    dry_run=False,
                    cpu_workers=max(4, int(executor.max_inflight)),
                    cancel_token=cancel,
                )
                native_artifact_reused = True
                native_artifact_id = int(artifact.artifact_id)
            else:
                native_report = native.execute_schedule(
                    executor.schedule,
                    instruction_handler=handler,
                    dry_run=False,
                    cpu_workers=max(4, int(executor.max_inflight)),
                )
                native_artifact_reused = False
                native_artifact_id = None
        except Exception as exc:
            _reraise_pending(executor, pending_exc, exc)

    if pending_exc:
        _reraise_pending(executor, pending_exc)

    missing = [i.name for i in executor.schedule.instructions if i.name not in completed]
    if missing:
        raise RuntimePlanError(f"Schedule left unfinished instructions: {missing}")

    executor._assert_activation_budget(ctx, completed)
    report.wall_time_s = float(native_report.get("wall_time_s") or (time.perf_counter() - wall0))
    report.parallel_overlaps = len(report.overlapping_pairs())
    report.copy_snapshot = ctx.copies.snapshot()
    report.multi_copy_peaks = list(ctx.telemetry.multi_copy_peaks)
    report.peak_activation_bytes = max(ctx.activation_peak_bytes, ctx.copies.activation_live_bytes())
    report.activation_bytes_written = ctx.telemetry.activation_bytes_written
    report.activation_bytes_read = ctx.telemetry.activation_bytes_read
    report.spill_latency_s = ctx.telemetry.spill_latency_s
    report.reload_latency_s = ctx.telemetry.reload_latency_s
    report.allocation_peak_bytes = ctx.allocations.peak_bytes()
    if report.allocation_peak_bytes == 0 and report.peak_activation_bytes > 0:
        report.allocation_peak_bytes = report.peak_activation_bytes
    executor.copies = ctx.copies
    executor._spill_events.extend(ctx.telemetry.spill_events)
    compute_intervals = [(e.start_s, e.end_s) for e in report.events if e.opcode == "Compute"]
    if hasattr(executor.parameter_store, "record_compute_intervals"):
        executor.parameter_store.record_compute_intervals(compute_intervals)
    stats = executor.parameter_store.stats()
    if isinstance(stats, dict):
        stats = dict(stats)
        stats["schedule_instruction_events"] = len(report.events)
        stats["schedule_driven"] = True
        stats["native_runtime"] = True
        stats["native_artifact_reused"] = native_artifact_reused
        stats["native_artifact_id"] = native_artifact_id
        stats["native_data_plane"] = native_data_plane
        if ctx.native_residency is not None:
            native_residency_stats = ctx.native_residency.stats()
            stats["native_residency"] = True
            stats["native_residency_stats"] = native_residency_stats
        else:
            stats["native_residency"] = False
        stats["peak_activation_bytes"] = report.peak_activation_bytes
        stats["activation_bytes_written"] = report.activation_bytes_written
        stats["activation_bytes_read"] = report.activation_bytes_read
    report.parameter_store = stats if isinstance(stats, dict) else {}
    report.max_concurrent = max(
        1,
        max_concurrency_from_intervals([(e.start_s, e.end_s) for e in report.events]),
    )
    if artifact is not None and not artifact.is_unmutated():
        raise RuntimePlanError("native compiled artifact mutated during execution")
    return executor._collect_outputs(ctx), report


def should_use_native_runtime() -> bool:
    if native_available():
        return True
    if allow_python_runtime():
        return False
    require_native()  # raises
    return False
