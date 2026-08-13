"""Native schedule execution entry points."""

from __future__ import annotations

import contextlib
import shutil
import time
from pathlib import Path
from typing import Any, cast

import torch

from tensortorrent.closed import CopyOwnership
from tensortorrent.errors import ExecutionCancelled, RuntimePlanError
from tensortorrent.ir.graph import OpCode
from tensortorrent.native import require_native
from tensortorrent.runtime.execution_context import ExecutionContext
from tensortorrent.runtime.native_bridge.residency import (
    _alias_host_compute_resources,
    _configure_virtual_backends,
    _move_tensor_to_resource,
    _register_persistent_residency,
    _schedule_needs_parameter_load,
    _schedule_needs_spill_callbacks,
    _tensor_already_on_resource,
    _torch_device_for_resource,
)
from tensortorrent.runtime.native_bridge.spill import (
    _merge_native_streaming_io_intervals,
    _setup_native_spill,
)
from tensortorrent.runtime.schedule.types import MemoryTier
from tensortorrent.runtime.schedule_report import (
    InstructionEvent,
    ScheduleReport,
    max_concurrency_from_intervals,
)


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


def run_schedule_native(
    executor: Any,
    flat_inputs: list[Any],
    *,
    cancel_token: Any | None = None,
    enable_grad: bool = False,
) -> tuple[list[Any], ScheduleReport]:
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

    ctx = ExecutionContext(
        host_resource=executor._default_host_resource(),
        enable_grad=bool(enable_grad),
        device_streams=None if enable_grad else getattr(executor, "_device_streams", None),
    )
    report = ScheduleReport(wall_time_s=0.0)
    host = ctx.host_resource
    if len(flat_inputs) != len(executor.program.user_inputs):
        raise RuntimePlanError(f"Expected {len(executor.program.user_inputs)} inputs, got {len(flat_inputs)}")

    executor.parameter_store.begin_execution()
    wall0 = time.perf_counter()
    completed: set[str] = set()
    events_by_name: dict[str, InstructionEvent] = {}
    pending_exc: list[BaseException] = []
    # Do not capture ``_native_artifact`` here: residency seeding may OOM and
    # reinstall a transfer/evict artifact. The body re-reads after registration.
    # Per-forward cancel token — concurrent forwards must not share one flag.
    run_cancel = cancel_token if cancel_token is not None else native.NativeCancelToken()
    if not hasattr(run_cancel, "cancel"):
        raise TypeError("cancel_token must be a NativeCancelToken-compatible object")
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
    run_cancel: Any,
    native: Any,
) -> tuple[list[Any], ScheduleReport]:
    native_data_plane = False
    native_artifact_reused = False
    native_artifact_id: int | None = None
    native_report: dict[str, Any] = {}
    shared_execution_id: int | None = None
    from tensortorrent.runtime.handles import NativeResidencyBridge

    # Fresh context per forward so cancel_token, events, and residency lifetime
    # match this run (never reuse a sticky cancel flag across forwards).
    needs_spill = _schedule_needs_spill_callbacks(executor)
    native_ctx = native.NativeExecutionContext(cancel_token=run_cancel)
    shared_execution_id = int(native_ctx.execution_id)
    ctx.native_execution_context = native_ctx
    residency = NativeResidencyBridge.create_from_context(native_ctx)
    ctx.attach_native_residency(residency)
    _configure_virtual_backends(native_ctx, executor)

    input_destinations = executor._input_transfer_destinations
    for name, value in zip(executor.program.user_inputs, flat_inputs, strict=True):
        dest = str(input_destinations.get(name) or "")
        if dest and _tensor_already_on_resource(value, dest):
            ctx.publish_tensor(name, dest, value, tier=MemoryTier.DEVICE, ownership=CopyOwnership.INPUT)
            continue
        ctx.publish_tensor(name, host, value, tier=MemoryTier.SYSTEM_RAM, ownership=CopyOwnership.INPUT)
        if host != "cpu":
            ctx.alias_copy(name, host, "cpu")
        if host != "host":
            ctx.alias_copy(name, host, "host")
        for res in executor._alias_target_resources:
            if res in {host, "cpu", "host"}:
                continue
            residency.mirror_alias(name, host, res)

    _register_persistent_residency(executor, ctx)
    # Residency may have fallen back to transfer/evict and rebuilt the artifact.
    artifact = getattr(executor, "_native_artifact", None)

    compute_by_region = executor._compute_by_region

    def region_handler(batch: list[tuple[str, list[str], list[str]]]) -> None:
        if run_cancel.is_cancelled():
            raise ExecutionCancelled("Schedule execution cancelled")

        def _run_one(region_id: str, _inputs: list[str], _outputs: list[str]) -> InstructionEvent:
            if run_cancel.is_cancelled():
                raise ExecutionCancelled("Schedule execution cancelled")
            inst = compute_by_region.get(region_id)
            if inst is None:
                raise RuntimePlanError(f"no Compute instruction for region {region_id!r}")
            submitted = time.perf_counter()
            ctx.state_for(inst.name).submitted_s = submitted
            try:
                return cast(InstructionEvent, executor._exec_compute(inst, ctx, submitted))
            except BaseException as exc:
                pending_exc.append(exc)
                raise

        # Autograd graphs are not safe across concurrent region threads.
        workers = 1 if ctx.enable_grad else max(1, int(getattr(executor, "max_workers", 1) or 1))
        if len(batch) <= 1 or workers <= 1:
            events = [_run_one(*item) for item in batch]
        else:
            pool = executor._ensure_region_pool(min(len(batch), workers))
            futs = [pool.submit(_run_one, *item) for item in batch]
            events = [fut.result() for fut in futs]

        for event in events:
            events_by_name[event.name] = event
            report.events.append(event)
            st = ctx.state_for(event.name)
            st.start_s = event.start_s
            st.completion_s = event.end_s
            st.result = event
            completed.add(event.name)
            executor._assert_activation_budget(ctx, completed)

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
    spill_dir: Path | None = None
    if needs_spill:
        from tensortorrent.runtime.activation_spill import (
            spill_bytes_to_tensor,
            tensor_to_spill_bytes,
        )

        spill_dir = _setup_native_spill(native, native_ctx, executor)

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
            # Rust spill path already released residency; drop Python handle only.
            ctx.drop_copy(tensor_id, resource, rust_already_released=True)
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
            if ctx.copies.has(tensor_id, dest):
                ctx.republish_value(tensor_id, dest, tensor, tier=MemoryTier.SYSTEM_RAM, nbytes=int(tensor.nbytes))
            else:
                ctx.publish_replica(
                    tensor_id,
                    dest,
                    tensor,
                    tier=MemoryTier.SYSTEM_RAM,
                    ownership=CopyOwnership.ACTIVATION,
                    nbytes=int(tensor.nbytes),
                    source_resource="disk",
                )
            ctx.telemetry.record_reload(
                name=tensor_id,
                nbytes=int(tensor.nbytes),
                latency_s=time.perf_counter() - t0,
                instruction="native_activation_reload",
            )

    if needs_param_load:

        def parameter_load_cb(pairs: list[tuple[str, str]]) -> list[int]:
            """Materialize onto each Load's destination resource (not a guessed host)."""
            sizes: list[int] = []
            for tensor_id, dest in pairs:
                dest = str(dest or ctx.host_resource)
                tensor = executor.parameter_store.acquire(tensor_id)
                nbytes = int(getattr(tensor, "nbytes", 0) or 0)
                if ctx.copies.has(tensor_id, dest):
                    ctx.republish_value(tensor_id, dest, tensor, tier=MemoryTier.SYSTEM_RAM, nbytes=nbytes)
                else:
                    ctx.publish_tensor(
                        tensor_id,
                        dest,
                        tensor,
                        tier=MemoryTier.SYSTEM_RAM,
                        ownership=CopyOwnership.PARAMETER,
                        nbytes=nbytes,
                    )
                # Also publish under the runtime host label when Load targeted pinned RAM.
                if dest != ctx.host_resource and not ctx.copies.has(tensor_id, ctx.host_resource):
                    ctx.alias_copy(tensor_id, dest, ctx.host_resource)
                _alias_host_compute_resources(executor, ctx, tensor_id, dest)
                sizes.append(nbytes)
            return sizes

    def handle_release_cb(pairs: list[tuple[str, str]]) -> None:
        """Rust final-released these copies — drop Python handles in one GIL cross."""
        # Training: keep Python tensor identities for autograd saved-for-backward;
        # the per-run ExecutionContext still drops everything when the forward ends.
        if ctx.enable_grad:
            return
        release_ids: list[str] = []
        for tensor_id, resource_id in pairs:
            rid = resource_id
            if not ctx.copies.has(tensor_id, rid) and ctx.copies.has(tensor_id, "disk"):
                rid = "disk"
            for alias in ctx.copies.resources_for(tensor_id):
                copy = ctx.copies.try_get(tensor_id, alias)
                if copy is not None:
                    copy.wait_ready()
            ctx.drop_copy(tensor_id, rid, rust_already_released=True)
            release_ids.append(tensor_id)
        # Unpin streaming decoded tensors so the RAM budget admits the next Load.
        if release_ids and hasattr(executor.parameter_store, "release"):
            executor.parameter_store.release(tuple(release_ids))
            native.record_parameter_release()

    def copy_sync_cb(pairs: list[tuple[str, str, str, int]]) -> None:
        """Keep Python handle table coherent with Rust Transfer (no post-run repair).

        Must not authoritative-``put`` the destination — that invalidates live source
        copies other ready Computes still need under concurrent schedules.
        """
        for tensor_id, src, dst, nbytes in pairs:
            if src == dst:
                continue
            if ctx.copies.has(tensor_id, dst):
                continue
            src_copy = ctx.copies.try_get(tensor_id, src)
            if src_copy is None:
                if ctx.native_residency is not None and ctx.native_residency.session.has(tensor_id, src):
                    value = ctx.native_residency.require_value(tensor_id, src)
                else:
                    continue
            else:
                value = src_copy.value
            from tensortorrent.runtime.virtual_tensor import VirtualDeviceTensor, wrap_virtual_native

            # Schedule training keeps live tensors: virtual byte wrap detaches grads.
            # Inference must still ``.to`` real CUDA/ROCm resources — residency labels
            # alone do not move storage.
            ready_event = None
            if not ctx.enable_grad:
                if "mock" in dst.lower() and not isinstance(value, VirtualDeviceTensor):
                    value = wrap_virtual_native(value, dst, native_ctx)
                elif "mock" not in dst.lower() and isinstance(value, VirtualDeviceTensor):
                    value = value.to_host()
                elif isinstance(value, torch.Tensor):
                    from tensortorrent.runtime.device_streams import runtime_for_context

                    streams = runtime_for_context(ctx)
                    if streams is not None:
                        target = _torch_device_for_resource(dst)
                        if target is not None:
                            value, ready_event = streams.transfer(value, target)
                        else:
                            value = _move_tensor_to_resource(value, dst, enable_grad=False)
                    else:
                        value = _move_tensor_to_resource(value, dst, enable_grad=False)
            elif isinstance(value, VirtualDeviceTensor):
                value = value.payload
            elif isinstance(value, torch.Tensor):
                value = _move_tensor_to_resource(value, dst, enable_grad=True)
            ownership: CopyOwnership | str = CopyOwnership.TRANSFER
            if src_copy is not None:
                ownership = getattr(src_copy, "ownership", None) or CopyOwnership.TRANSFER
            else:
                for rid in ctx.copies.resources_for(tensor_id):
                    c = ctx.copies.try_get(tensor_id, rid)
                    if c is not None and c.ownership == CopyOwnership.ACTIVATION:
                        ownership = CopyOwnership.ACTIVATION
                        break
            ctx.publish_replica(
                tensor_id,
                dst,
                value,
                ownership=ownership,
                nbytes=int(nbytes or getattr(value, "nbytes", 0) or 0),
                source_resource=src,
                ready_event=ready_event,
            )
            resources = ctx.copies.resources_for(tensor_id)
            if len(resources) > 1:
                ctx.telemetry.multi_copy_peaks.append(
                    {
                        "tensor_id": tensor_id,
                        "resources": list(resources),
                        "at": "transfer_complete",
                    }
                )
            if isinstance(value, VirtualDeviceTensor) and value.native_buffer_id is not None:
                native_ctx.bind_virtual_buffer(tensor_id, dst, int(value.native_buffer_id))

    try:
        if artifact is None:
            raise RuntimePlanError(
                "NativeCompiledArtifact required (install failed — refuse execute_schedule fallback)"
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
        exec_kwargs["handle_release_callback"] = handle_release_cb
        exec_kwargs["copy_sync_callback"] = copy_sync_cb
        # Align Rust Instant-relative telemetries with Python perf_counter Compute times.
        rust_time_origin = time.perf_counter()
        native_report = artifact.execute(**exec_kwargs)
        native_artifact_reused = True
        native_artifact_id = int(artifact.artifact_id)
        native_data_plane = True
        # Retain for leak diagnostics / tests (dropped on next forward or close).
        executor._last_native_ctx = native_ctx
        for ev in native_report.get("events") or []:
            name = str(ev.get("name") or "")
            opcode_raw = str(ev.get("opcode") or "")
            try:
                opcode = OpCode(opcode_raw)
            except ValueError:
                # Unknown native opcode tag — skip event rather than poison report.
                continue
            if opcode == OpCode.COMPUTE and name in events_by_name:
                # Region handler owns timings; Rust owns simulated label (VirtualBackend).
                if bool(ev.get("simulated")):
                    events_by_name[name].simulated = True
                completed.add(name)
                continue
            if name in events_by_name:
                continue
            if opcode == OpCode.COMPUTE:
                completed.add(name)
                continue
            event = InstructionEvent(
                name=name,
                opcode=opcode,
                resource=str(ev.get("resource") or ""),
                submitted_s=rust_time_origin + float(ev.get("submitted_s") or 0.0),
                start_s=rust_time_origin + float(ev.get("start_s") or 0.0),
                end_s=rust_time_origin + float(ev.get("end_s") or 0.0),
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
        # Handle drop + Transfer sync happen mid-schedule via Rust callbacks.
        ctx.note_activation_live(ctx.copies.activation_live_bytes())
        _merge_native_streaming_io_intervals(executor)
    finally:
        if spill_dir is not None:
            shutil.rmtree(spill_dir, ignore_errors=True)

    if pending_exc:
        _reraise_pending(executor, pending_exc)

    missing = [name for name in executor._native_instruction_names if name not in completed]
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
    # Rust AllocationTable is authoritative on the native path.
    report.allocation_peak_bytes = int(native_report.get("allocation_peak_bytes") or 0)
    if report.allocation_peak_bytes == 0 and report.peak_activation_bytes > 0:
        report.allocation_peak_bytes = report.peak_activation_bytes
    report.spill_events = list(ctx.telemetry.spill_events)
    compute_intervals = [(e.start_s, e.end_s) for e in report.events if e.opcode == OpCode.COMPUTE]
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
            stats["handle_live"] = int(native_residency_stats.get("handle_live", 0) or 0)
            stats["handle_live_bytes"] = int(native_residency_stats.get("handle_live_bytes", 0) or 0)
        else:
            stats["native_residency"] = False
            stats["handle_live"] = 0
            stats["handle_live_bytes"] = 0
        stats["non_compute_python_io"] = False
        counters = dict(native.debug_counters())
        stats["compute_callbacks"] = int(counters.get("compute_callbacks", 0) or 0)
        stats["non_compute_python_callbacks"] = int(counters.get("non_compute_python_callbacks", 0) or 0)
        stats["gil_acquisitions"] = int(counters.get("gil_acquisitions", 0) or 0)
        stats["parameter_load_callbacks"] = int(counters.get("parameter_load_callbacks", 0) or 0)
        stats["handle_release_callbacks"] = int(counters.get("handle_release_callbacks", 0) or 0)
        stats["spill_dematerialize_callbacks"] = int(counters.get("spill_dematerialize_callbacks", 0) or 0)
        stats["spill_materialize_callbacks"] = int(counters.get("spill_materialize_callbacks", 0) or 0)
        stats["copy_sync_callbacks"] = int(counters.get("copy_sync_callbacks", 0) or 0)
        stats["peak_activation_bytes"] = report.peak_activation_bytes
        stats["activation_bytes_written"] = report.activation_bytes_written
        stats["activation_bytes_read"] = report.activation_bytes_read
        if hasattr(native_ctx, "virtual_peak_bytes"):
            stats["virtual_peak_bytes"] = int(native_ctx.virtual_peak_bytes())
            # Sum live virtual bytes across mock resources seen this forward.
            live_vb = 0
            for res in executor._mock_resources:
                live_vb = max(live_vb, int(native_ctx.virtual_backend_used_bytes(res)))
            stats["virtual_live_bytes"] = live_vb
    report.parameter_store = stats if isinstance(stats, dict) else {}
    report.max_concurrent = max(
        1,
        max_concurrency_from_intervals([(e.start_s, e.end_s) for e in report.events]),
    )
    if artifact is not None and not artifact.is_unmutated():
        raise RuntimePlanError("native compiled artifact mutated during execution")
    return executor._collect_outputs(ctx), report
