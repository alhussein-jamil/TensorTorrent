"""Bridge: Rust schedules instructions; Python executes tensor-bearing ops."""

from __future__ import annotations

import contextlib
import time
from typing import Any

from streamcompiler.errors import ExecutionCancelled, RuntimePlanError
from streamcompiler.ir.graph import OpCode
from streamcompiler.native import native_available, require_native
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
    """True when Compute is region-callback and I/O uses a narrow handler.

    Streaming Prefetch/Load and activation spill/reload run through ``io_handler``
    on the hybrid native path. Mock-delay Compute still needs the full
    instruction-callback stream path.
    """
    for inst in executor.schedule.instructions:
        if inst.opcode != OpCode.COMPUTE:
            continue
        if float(inst.attributes.get("mock_compute_delay_s", 0.0)) > 0.0:
            return False
        if "mock" in str(inst.resource) and float(inst.attributes.get("mock_compute_delay_s", 0.0)) > 0.0:
            return False
    return True


def _prematerialize_loads(executor: Any, ctx: ExecutionContext) -> list[InstructionEvent]:
    """Prematerialize resident (non-streaming) Loads only.

    Streaming Loads stay on the I/O handler so the RAM budget stays honest.
    """
    if getattr(executor.parameter_store, "needs_prefetch", False):
        return []
    events: list[InstructionEvent] = []
    for inst in executor.schedule.instructions:
        if inst.opcode != OpCode.LOAD:
            continue
        kind = str(inst.attributes.get("kind") or "")
        if kind in {"activation_reload", "activation_spill"}:
            continue
        submitted = time.perf_counter()
        event = executor._exec_load(inst, ctx, submitted)
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
                            if "mock" in res.lower() or _tier_is_device_name(res):
                                continue
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
                                if "mock" in res.lower() or _tier_is_device_name(res):
                                    continue
                                ctx.native_residency.mirror_alias(target, dest, res)
        events.append(event)
    return events


def _exec_io_inline(executor: Any, inst: Any, ctx: ExecutionContext, submitted: float) -> Any:
    """Python body for Prefetch/Load/Release/Evict on the hybrid native path."""
    opcode = inst.opcode
    if opcode == OpCode.PREFETCH:
        return executor._exec_prefetch(inst, ctx, submitted)
    if opcode == OpCode.LOAD:
        return executor._exec_load(inst, ctx, submitted)
    if opcode == OpCode.RELEASE:
        return executor._exec_release(inst, ctx, submitted)
    if opcode == OpCode.EVICT:
        return executor._exec_evict(inst, ctx, submitted)
    raise RuntimePlanError(f"io_handler unsupported opcode {opcode}")


def _sync_python_lifetime_ops(executor: Any, ctx: ExecutionContext, completed: set[str]) -> None:
    """Apply Release/Evict drops to Python CopyStore after a native region-path run.

    Rust already updated its residency metadata; Python still holds tensor values
    and streaming pin counts.
    """
    for inst in executor.schedule.instructions:
        if inst.name not in completed:
            continue
        if inst.opcode == OpCode.RELEASE:
            released_names: list[str] = []
            for tensor_id in inst.inputs:
                resource = str(inst.attributes.get("release_resource") or inst.resource)
                if not ctx.copies.has(tensor_id, resource):
                    if ctx.copies.has(tensor_id, "disk"):
                        resource = "disk"
                    else:
                        released_names.append(str(tensor_id))
                        continue
                if ctx.native_residency is not None and ctx.native_residency.session.has(
                    tensor_id, resource
                ):
                    ctx.native_residency.release(tensor_id, resource)
                ctx.copies.drop(tensor_id, resource)
                released_names.append(str(tensor_id))
            if released_names and hasattr(executor.parameter_store, "release"):
                executor.parameter_store.release(tuple(released_names))
        elif inst.opcode == OpCode.EVICT:
            kind = str(inst.attributes.get("kind") or "")
            if kind == "activation_spill":
                continue  # spill needs full Python body; handled by io_handler
            for tensor_id in inst.inputs:
                resource = str(inst.destination or inst.resource)
                if not ctx.copies.has(tensor_id, resource):
                    continue
                if ctx.native_residency is not None and ctx.native_residency.session.has(
                    tensor_id, resource
                ):
                    ctx.native_residency.release(tensor_id, resource)
                ctx.copies.drop(tensor_id, resource)


def _reraise_pending(executor: Any, pending_exc: list[BaseException], exc: Exception | None = None) -> None:
    def _clear_sticky() -> None:
        executor._cancel = False

    if pending_exc:
        err = pending_exc[0]
        if isinstance(err, ExecutionCancelled):
            _clear_sticky()
        if exc is not None:
            raise err from exc
        raise err
    if exc is None:
        return
    if isinstance(exc, ExecutionCancelled):
        _clear_sticky()
        raise exc
    if isinstance(exc, RuntimePlanError):
        raise exc
    msg = str(exc)
    if "ExecutionCancelled" in msg or "cancelled" in msg.lower():
        _clear_sticky()
        raise ExecutionCancelled("Schedule execution cancelled") from exc
    raise RuntimePlanError(f"native schedule execution failed: {exc}") from exc


def run_schedule_native(executor: Any, flat_inputs: list[Any]) -> tuple[list[Any], ScheduleReport]:
    """Run ``executor.schedule`` under the Rust dispatcher.

    Prefers a persistent :class:`NativeCompiledArtifact` on the executor so the
    Rust schedule is not rebuilt on every forward. When the schedule permits,
    Rust owns Load/Release/Record/Wait residency bookkeeping and Python is
    entered only for PyTorch region compute.

    Runtime path is selected once before any work. Mid-forward restart through
    another path is forbidden.
    """
    native = require_native()
    if executor._closed:
        raise RuntimePlanError("ScheduleExecutor is closed")
    # Sticky module cancel: next forward consumes it once (idle cancel).
    # In-flight siblings use per-forward tokens only — never clear shared mid-run.
    if executor._cancel:
        executor._cancel = False
        raise ExecutionCancelled("Schedule execution cancelled")

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
    artifact = getattr(executor, "_native_artifact", None)
    # Per-forward cancel token — concurrent forwards must not share one flag.
    run_cancel = native.NativeCancelToken()
    cancel_lock = getattr(executor, "_cancel_lock", None)
    if cancel_lock is not None:
        with cancel_lock:
            executor._active_cancels.append(run_cancel)
    elif hasattr(executor, "_active_cancels"):
        executor._active_cancels.append(run_cancel)
    try:
        return _run_schedule_native_body(
            executor,
            flat_inputs,
            ctx,
            report,
            host,
            wall0,
            completed,
            events_by_name,
            pending_exc,
            use_region_path,
            artifact,
            run_cancel,
            native,
        )
    finally:
        if cancel_lock is not None:
            with cancel_lock:
                with contextlib.suppress(ValueError):
                    executor._active_cancels.remove(run_cancel)
        elif hasattr(executor, "_active_cancels"):
            with contextlib.suppress(ValueError):
                executor._active_cancels.remove(run_cancel)


def _run_schedule_native_body(
    executor: Any,
    flat_inputs: list[Any],
    ctx: ExecutionContext,
    report: ScheduleReport,
    host: str,
    wall0: float,
    completed: set[str],
    events_by_name: dict[str, InstructionEvent],
    pending_exc: list[BaseException],
    use_region_path: bool,
    artifact: Any,
    run_cancel: Any,
    native: Any,
) -> tuple[list[Any], ScheduleReport]:
    native_data_plane = False
    native_artifact_reused = False
    native_artifact_id: int | None = None
    native_report: dict[str, Any] = {}
    shared_execution_id: int | None = None
    if use_region_path:
        from streamcompiler.runtime.handles import NativeResidencyBridge

        # One NativeExecutionContext per forward: residency session + Rust
        # dispatcher share the same store / event table / allocations.
        native_ctx = native.NativeExecutionContext(cancel_token=run_cancel)
        shared_execution_id = int(native_ctx.execution_id)
        ctx.native_residency = NativeResidencyBridge.create_from_context(native_ctx)
        for name, value in zip(executor.program.user_inputs, flat_inputs, strict=True):
            ctx.mirror_native_put(name, host, value)
            if host != "cpu":
                ctx.native_residency.mirror_alias(name, host, "cpu")
            if host != "host":
                ctx.native_residency.mirror_alias(name, host, "host")
            # Host compute-pool aliases only — mock/device needs explicit Transfer.
            for inst in executor.schedule.instructions:
                if inst.opcode != OpCode.COMPUTE:
                    continue
                res = str(inst.resource)
                if res in {host, "cpu", "host"}:
                    continue
                if "mock" in res.lower() or _tier_is_device_name(res):
                    continue
                ctx.native_residency.mirror_alias(name, host, res)

        for event in _prematerialize_loads(executor, ctx):
            events_by_name[event.name] = event
            report.events.append(event)
            completed.add(event.name)

        compute_by_region = {
            str(inst.executable_ref or ""): inst
            for inst in executor.schedule.instructions
            if inst.opcode == OpCode.COMPUTE
        }

        def region_handler(batch: list[tuple[str, list[str], list[str]]]) -> None:
            if ctx.cancellation.cancelled or run_cancel.is_cancelled():
                run_cancel.cancel()
                raise ExecutionCancelled("Schedule execution cancelled")
            for region_id, _inputs, _outputs in batch:
                inst = compute_by_region.get(region_id)
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

        def io_handler(name: str) -> dict[str, Any]:
            if ctx.cancellation.cancelled or run_cancel.is_cancelled():
                run_cancel.cancel()
                raise ExecutionCancelled("Schedule execution cancelled")
            # Prematerialized Loads already ran; do not double-append events.
            if name in completed:
                prior = events_by_name.get(name)
                return {
                    "nbytes": int(prior.nbytes) if prior is not None else 0,
                    "simulated": bool(prior.simulated) if prior is not None else False,
                    "notes": "prematerialized",
                }
            inst = executor._by_name[name]
            submitted = time.perf_counter()
            ctx.state_for(name).submitted_s = submitted
            try:
                event = _exec_io_inline(executor, inst, ctx, submitted)
            except BaseException as exc:
                pending_exc.append(exc)
                raise
            # Mirror Load results into Rust residency for later Transfer/Release.
            if inst.opcode == OpCode.LOAD:
                dest = str(inst.destination or inst.resource)
                from streamcompiler.runtime.schedule_executor import _tier_is_device

                if _tier_is_device(dest):
                    dest = ctx.host_resource
                for env_name in executor._state_env_names(inst):
                    copy = ctx.copies.try_get(env_name, dest)
                    if copy is not None:
                        ctx.mirror_native_put(env_name, dest, copy.value, nbytes=int(copy.nbytes))
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
                "notes": str(event.notes or "native_io_handler"),
            }

        needs_io = any(
            inst.opcode in (OpCode.PREFETCH, OpCode.LOAD, OpCode.RELEASE, OpCode.EVICT)
            for inst in executor.schedule.instructions
        )

        try:
            if artifact is not None:
                native_report = artifact.execute(
                    region_callback=region_handler,
                    instruction_handler=io_handler if needs_io else None,
                    dry_run=False,
                    cpu_workers=max(4, int(executor.max_inflight)),
                    cancel_token=run_cancel,
                    execution_context=native_ctx,
                )
                native_artifact_reused = True
                native_artifact_id = int(artifact.artifact_id)
            else:
                native_report = native.execute_schedule(
                    executor.schedule,
                    region_callback=region_handler,
                    instruction_handler=io_handler if needs_io else None,
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
            # Static path selection: never restart through instruction-callback
            # after region-path work (including prematerialized Loads / Computes).
            _reraise_pending(executor, pending_exc, exc)
        else:
            # Region path: Rust Release/Evict only touch Rust residency — sync Python bags.
            ops = {inst.opcode for inst in executor.schedule.instructions}
            if OpCode.RELEASE in ops or OpCode.EVICT in ops:
                _sync_python_lifetime_ops(executor, ctx, completed)
            if OpCode.TRANSFER in ops:
                _sync_python_copies_after_native_transfers(executor, ctx, completed)

    if not native_data_plane:

        def handler(name: str) -> dict[str, Any]:
            if ctx.cancellation.cancelled or run_cancel.is_cancelled():
                run_cancel.cancel()
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
                    cancel_token=run_cancel,
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
    # Prefer per-report spill list — avoid shared list races under concurrent forwards.
    report.spill_events = list(ctx.telemetry.spill_events)
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
        if shared_execution_id is not None:
            stats["native_execution_id"] = shared_execution_id
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


def _tier_is_device_name(resource: str) -> bool:
    name = resource.lower()
    return any(tok in name for tok in ("mock", "cuda", "rocm", "gpu", "xpu", "mps", "vram"))


def _sync_python_copies_after_native_transfers(executor: Any, ctx: ExecutionContext, completed: set[str]) -> None:
    """Materialize Transfer destinations into CopyStore from native handle table.

    Rust Transfer only updates residency metadata; Compute may already have
    wrapped values. This pass keeps collect/Release Python bags coherent.
    """
    if ctx.native_residency is None:
        return
    from streamcompiler.runtime.virtual_tensor import VirtualDeviceTensor, wrap_virtual

    for inst in executor.schedule.instructions:
        if inst.name not in completed or inst.opcode != OpCode.TRANSFER:
            continue
        src = str(inst.source or ctx.host_resource)
        dst = str(inst.destination or inst.resource)
        for tensor_id in list(inst.inputs) + list(inst.outputs):
            if ctx.copies.has(tensor_id, dst):
                continue
            if not ctx.native_residency.session.has(tensor_id, dst):
                continue
            try:
                value = ctx.native_residency.require_value(tensor_id, src if ctx.native_residency.session.has(tensor_id, src) else dst)
            except Exception:
                try:
                    value = ctx.native_residency.require_value(tensor_id, dst)
                except Exception:
                    continue
            if _tier_is_device_name(dst) and not isinstance(value, VirtualDeviceTensor):
                value = wrap_virtual(value, dst)
            elif not _tier_is_device_name(dst) and isinstance(value, VirtualDeviceTensor):
                value = value.to_host()
            src_res = src if ctx.copies.has(tensor_id, src) else None
            ctx.copies.replicate(
                tensor_id,
                dst,
                value,
                ownership="transfer",
                source_resource=src_res,
            )
            with ctx.native_residency._lock:
                handle = ctx.native_residency.require_handle(tensor_id, dst)
                ctx.native_residency._index[(str(tensor_id), str(dst))] = handle

