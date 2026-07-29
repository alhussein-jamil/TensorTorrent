"""Bridge: Rust schedules instructions; Python executes tensor-bearing ops."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from streamcompiler.errors import ExecutionCancelled, RuntimePlanError
from streamcompiler.ir.graph import OpCode
from streamcompiler.native import require_native
from streamcompiler.runtime.execution_context import ExecutionContext
from streamcompiler.runtime.schedule_executor import (
    InstructionEvent,
    ScheduleReport,
    max_concurrency_from_intervals,
)
from streamcompiler.runtime.schedule_executor import (
    _tier_is_device as _tier_is_device_name,
)


def _schedule_needs_spill_callbacks(executor: Any) -> bool:
    for inst in executor.schedule.instructions:
        kind = str(inst.attributes.get("kind") or "")
        if inst.opcode == OpCode.EVICT and kind == "activation_spill":
            return True
        if inst.opcode == OpCode.LOAD and kind == "activation_reload":
            return True
    return False


def _schedule_needs_parameter_load(executor: Any) -> bool:
    if not bool(getattr(executor.parameter_store, "needs_prefetch", False)):
        return False
    for inst in executor.schedule.instructions:
        if inst.opcode == OpCode.LOAD and str(inst.attributes.get("kind") or "") == "parameter_materialize":
            return True
    return False


def _register_persistent_residency(executor: Any, ctx: ExecutionContext) -> None:
    """Register already-resident parameters into value bag + native residency.

    Does **not** run schedule Load and does **not** call ``_exec_load``.
    Rust executes ``Load`` at the real schedule position as residency verify.
    """
    if getattr(executor.parameter_store, "needs_prefetch", False):
        return
    from streamcompiler.runtime.schedule_executor import _copy_tier, _ensure_pinned, _tier_is_device

    for inst in executor.schedule.instructions:
        if inst.opcode != OpCode.LOAD:
            continue
        kind = str(inst.attributes.get("kind") or "")
        if kind in {"activation_reload", "activation_spill"}:
            continue
        # Resident `parameter_materialize` still pre-registers values (not a schedule Load body).
        dest = str(inst.destination or inst.resource)
        if _tier_is_device(dest):
            dest = ctx.host_resource
        tier = _copy_tier(inst.memory_tier)
        for env_name in executor._state_env_names(inst):
            tensor = executor.parameter_store.acquire(env_name)
            if tier == "pinned_ram":
                tensor = _ensure_pinned(tensor)
            nbytes = int(getattr(tensor, "nbytes", 0) or 0)
            if not ctx.copies.has(env_name, dest, valid_only=True):
                ctx.copies.put(env_name, dest, tensor, tier=tier, ownership="parameter")
            else:
                ctx.copies.replace_handle(env_name, dest, tensor, tier=tier)
            ctx.mirror_native_put(env_name, dest, tensor, nbytes=nbytes)
            _alias_host_compute_resources(executor, ctx, env_name, dest)
            target = executor.program.state_bindings.get(env_name, env_name)
            if target != env_name:
                if not ctx.copies.has(target, dest, valid_only=True):
                    ctx.copies.put(target, dest, tensor, tier=tier, ownership="parameter")
                else:
                    ctx.copies.replace_handle(target, dest, tensor, tier=tier)
                ctx.mirror_native_put(target, dest, tensor, nbytes=nbytes)
                _alias_host_compute_resources(executor, ctx, target, dest)


def _alias_host_compute_resources(executor: Any, ctx: ExecutionContext, tensor_id: str, dest: str) -> None:
    if ctx.native_residency is None:
        return
    for compute in executor.schedule.instructions:
        if compute.opcode != OpCode.COMPUTE:
            continue
        res = str(compute.resource)
        if res == dest:
            continue
        if "mock" in res.lower() or _tier_is_device_name(res):
            continue
        ctx.native_residency.mirror_alias(tensor_id, dest, res)


def _drop_python_values_after_native_lifetime(executor: Any, ctx: ExecutionContext, completed: set[str]) -> None:
    """Drop Python tensor values after Rust already released residency metadata.

    Never re-enters native release/evict — Rust is authoritative. Only clears the
    Python value bag and streaming pin counts.
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
                ctx.copies.drop(tensor_id, resource)
                released_names.append(str(tensor_id))
            if released_names and hasattr(executor.parameter_store, "release"):
                executor.parameter_store.release(tuple(released_names))
        elif inst.opcode == OpCode.EVICT:
            kind = str(inst.attributes.get("kind") or "")
            if kind == "activation_spill":
                continue  # spill body owns Python values via io_handler
            for tensor_id in inst.inputs:
                resource = str(inst.destination or inst.resource)
                if ctx.copies.has(tensor_id, resource):
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
            artifact,
            run_cancel,
            native,
        )
    finally:
        if cancel_lock is not None:
            with cancel_lock, contextlib.suppress(ValueError):
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
    artifact: Any,
    run_cancel: Any,
    native: Any,
) -> tuple[list[Any], ScheduleReport]:
    native_data_plane = False
    native_artifact_reused = False
    native_artifact_id: int | None = None
    native_report: dict[str, Any] = {}
    shared_execution_id: int | None = None
    from streamcompiler.runtime.handles import NativeResidencyBridge

    # One NativeExecutionContext per forward: residency + events + allocs.
    native_ctx = native.NativeExecutionContext(cancel_token=run_cancel)
    shared_execution_id = int(native_ctx.execution_id)
    ctx.native_residency = NativeResidencyBridge.create_from_context(native_ctx)
    for name, value in zip(executor.program.user_inputs, flat_inputs, strict=True):
        ctx.mirror_native_put(name, host, value)
        if host != "cpu":
            ctx.native_residency.mirror_alias(name, host, "cpu")
        if host != "host":
            ctx.native_residency.mirror_alias(name, host, "host")
        for inst in executor.schedule.instructions:
            if inst.opcode != OpCode.COMPUTE:
                continue
            res = str(inst.resource)
            if res in {host, "cpu", "host"}:
                continue
            if "mock" in res.lower() or _tier_is_device_name(res):
                continue
            ctx.native_residency.mirror_alias(name, host, res)

    _register_persistent_residency(executor, ctx)

    compute_by_region = {
        str(inst.executable_ref or ""): inst for inst in executor.schedule.instructions if inst.opcode == OpCode.COMPUTE
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

    needs_spill = _schedule_needs_spill_callbacks(executor)
    needs_param_load = _schedule_needs_parameter_load(executor)

    native_store = getattr(executor.parameter_store, "_native_store", None)
    bindings = getattr(executor.parameter_store, "_env_to_key", None)
    native_io_origin = time.perf_counter()
    if native_store is not None and isinstance(bindings, dict):
        try:
            native_ctx.set_streaming_store(native_store, dict(bindings))
            executor.parameter_store._native_io_origin = native_io_origin
        except TypeError:
            pass

    dematerialize_cb = None
    materialize_cb = None
    parameter_load_cb = None
    parameter_release_cb = None
    spill_dir: Path | None = None
    if needs_spill:
        from streamcompiler.runtime.activation_spill import (
            spill_bytes_to_tensor,
            tensor_to_spill_bytes,
        )

        spill_dir = Path(tempfile.mkdtemp(prefix="sc_native_spill_"))
        native_ctx.set_spill_dir(str(spill_dir))

        def dematerialize_cb(tensor_id: str) -> dict[str, Any]:
            resource = ctx.host_resource
            copy = ctx.copies.try_get(tensor_id, resource)
            if copy is None:
                for rid in ctx.copies.resources_for(tensor_id):
                    if rid == "disk":
                        continue
                    copy = ctx.copies.try_get(tensor_id, rid)
                    if copy is not None:
                        resource = rid
                        break
            if copy is None or not isinstance(copy.value, torch.Tensor):
                raise RuntimePlanError(f"dematerialize missing tensor {tensor_id!r}")
            t0 = time.perf_counter()
            dtype, shape, raw = tensor_to_spill_bytes(copy.value)
            ctx.copies.drop(tensor_id, resource)
            ctx.telemetry.record_spill(
                name=tensor_id,
                nbytes=len(raw),
                latency_s=time.perf_counter() - t0,
                instruction="native_activation_spill",
            )
            return {"dtype": dtype, "shape": shape, "bytes": raw}

        def materialize_cb(tensor_id: str, dtype: str, shape: list[int], raw: bytes) -> None:
            t0 = time.perf_counter()
            tensor = spill_bytes_to_tensor(dtype, list(shape), bytes(raw))
            dest = ctx.host_resource
            if ctx.copies.has(tensor_id, dest, valid_only=True):
                ctx.copies.replace_handle(tensor_id, dest, tensor, tier="system_ram")
            else:
                ctx.copies.replicate(
                    tensor_id,
                    dest,
                    tensor,
                    tier="system_ram",
                    ownership="activation",
                    source_resource="disk",
                )
            ctx.mirror_native_put(tensor_id, dest, tensor, nbytes=int(tensor.nbytes))
            ctx.telemetry.record_reload(
                name=tensor_id,
                nbytes=int(tensor.nbytes),
                latency_s=time.perf_counter() - t0,
                instruction="native_activation_reload",
            )

    if needs_param_load:

        def parameter_load_cb(tensor_id: str) -> int:
            tensor = executor.parameter_store.acquire(tensor_id)
            dest = ctx.host_resource
            nbytes = int(getattr(tensor, "nbytes", 0) or 0)
            if ctx.copies.has(tensor_id, dest, valid_only=True):
                ctx.copies.replace_handle(tensor_id, dest, tensor, tier="system_ram")
            else:
                ctx.copies.put(
                    tensor_id,
                    dest,
                    tensor,
                    tier="system_ram",
                    authoritative=True,
                    ownership="parameter",
                )
            ctx.mirror_native_put(tensor_id, dest, tensor, nbytes=nbytes)
            _alias_host_compute_resources(executor, ctx, tensor_id, dest)
            return nbytes

        def parameter_release_cb(names: list[str]) -> None:
            for tensor_id in names:
                for rid in list(ctx.copies.resources_for(tensor_id)):
                    if rid == "disk":
                        continue
                    if ctx.copies.has(tensor_id, rid):
                        ctx.copies.drop(tensor_id, rid)
            if hasattr(executor.parameter_store, "release"):
                executor.parameter_store.release(tuple(names))

    try:
        if needs_spill and artifact is None:
            raise RuntimePlanError(
                "activation spill/reload requires NativeCompiledArtifact (dematerialize/materialize callbacks)"
            )
        exec_kwargs: dict[str, Any] = {
            "region_callback": region_handler,
            "instruction_handler": None,
            "dry_run": False,
            "cpu_workers": max(4, int(executor.max_inflight)),
            "cancel_token": run_cancel,
            "execution_context": native_ctx,
        }
        if needs_spill:
            exec_kwargs["dematerialize_callback"] = dematerialize_cb
            exec_kwargs["materialize_callback"] = materialize_cb
        if needs_param_load:
            exec_kwargs["parameter_load_callback"] = parameter_load_cb
            exec_kwargs["parameter_release_callback"] = parameter_release_cb
        if artifact is not None:
            native_report = artifact.execute(**exec_kwargs)
            native_artifact_reused = True
            native_artifact_id = int(artifact.artifact_id)
        else:
            native_report = native.execute_schedule(
                executor.schedule,
                region_callback=region_handler,
                instruction_handler=None,
                dry_run=False,
                cpu_workers=max(4, int(executor.max_inflight)),
            )
        native_data_plane = True
        for ev in native_report.get("events") or []:
            name = str(ev.get("name") or "")
            opcode = str(ev.get("opcode") or "")
            if opcode == "Compute" and name in events_by_name:
                # Region handler owns timings; Rust owns simulated label (VirtualBackend).
                if bool(ev.get("simulated")):
                    events_by_name[name].simulated = True
                completed.add(name)
                continue
            if name in events_by_name:
                continue
            if opcode == "Compute":
                completed.add(name)
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
        # Never restart through another runtime after native work began.
        _reraise_pending(executor, pending_exc, exc)
    else:
        ops = {inst.opcode for inst in executor.schedule.instructions}
        if OpCode.TRANSFER in ops:
            _sync_python_copies_after_native_transfers(executor, ctx, completed)
            ctx.note_activation_live(ctx.copies.activation_live_bytes())
        if OpCode.RELEASE in ops or OpCode.EVICT in ops:
            _drop_python_values_after_native_lifetime(executor, ctx, completed)
        _merge_native_streaming_io_intervals(executor)
    finally:
        if spill_dir is not None:
            shutil.rmtree(spill_dir, ignore_errors=True)

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
        stats["non_compute_python_io"] = False
        counters = dict(native.debug_counters())
        stats["compute_callbacks"] = int(counters.get("compute_callbacks", 0) or 0)
        stats["non_compute_python_callbacks"] = int(counters.get("non_compute_python_callbacks", 0) or 0)
        stats["gil_acquisitions"] = int(counters.get("gil_acquisitions", 0) or 0)
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


def _merge_native_streaming_io_intervals(executor: Any) -> None:
    store = getattr(executor, "parameter_store", None)
    native = getattr(store, "_native_store", None)
    if store is None or native is None or not hasattr(native, "io_intervals"):
        return
    origin = float(getattr(store, "_native_io_origin", 0.0) or 0.0)
    if origin <= 0.0:
        return
    from streamcompiler.runtime.tensor_store import IoInterval

    intervals = list(getattr(store, "_io_intervals", []))
    for start, end, nbytes in native.io_intervals():
        intervals.append(
            IoInterval(
                name="native_prefetch",
                start_s=origin + float(start),
                end_s=origin + float(end),
                nbytes=int(nbytes),
            )
        )
    store._io_intervals = intervals


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
            if not ctx.native_residency.session.has(tensor_id, dst):
                continue
            if not ctx.copies.has(tensor_id, dst):
                try:
                    value = ctx.native_residency.require_value(
                        tensor_id, src if ctx.native_residency.session.has(tensor_id, src) else dst
                    )
                except (RuntimePlanError, KeyError, ValueError):
                    try:
                        value = ctx.native_residency.require_value(tensor_id, dst)
                    except (RuntimePlanError, KeyError, ValueError):
                        continue
                if _tier_is_device_name(dst) and not isinstance(value, VirtualDeviceTensor):
                    value = wrap_virtual(value, dst)
                elif not _tier_is_device_name(dst) and isinstance(value, VirtualDeviceTensor):
                    value = value.to_host()
                src_res = src if ctx.copies.has(tensor_id, src) else None
                ownership = "transfer"
                if src_res is not None:
                    src_copy = ctx.copies.try_get(tensor_id, src_res)
                    if src_copy is not None:
                        ownership = str(src_copy.ownership or "transfer")
                ctx.copies.replicate(
                    tensor_id,
                    dst,
                    value,
                    ownership=ownership,
                    source_resource=src_res,
                )
                with ctx.native_residency._lock:
                    handle = ctx.native_residency.require_handle(tensor_id, dst)
                    ctx.native_residency._index[(str(tensor_id), str(dst))] = handle
            else:
                # Dest already mirrored mid-run (e.g. Compute wrap). Promote ownership
                # so activation peak counts distinct host+device physical copies.
                src_res = src if ctx.copies.has(tensor_id, src) else None
                if src_res is not None:
                    src_copy = ctx.copies.try_get(tensor_id, src_res)
                    dst_copy = ctx.copies.try_get(tensor_id, dst)
                    if (
                        src_copy is not None
                        and dst_copy is not None
                        and src_copy.ownership == "activation"
                        and dst_copy.ownership != "activation"
                    ):
                        dst_copy.ownership = "activation"
            resources = ctx.copies.resources_for(tensor_id, valid_only=True)
            if len(resources) > 1:
                ctx.telemetry.multi_copy_peaks.append(
                    {
                        "tensor_id": tensor_id,
                        "resources": list(resources),
                        "at": "transfer_complete",
                    }
                )
            # Distinct host+device activation copies count toward peak (sim parity).
            live = ctx.copies.activation_live_bytes()
            ctx.note_activation_live(live)
