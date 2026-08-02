"""Bridge: Rust schedules instructions; Python executes tensor-bearing ops."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import torch

from streamcompiler.errors import ExecutionCancelled, RuntimePlanError
from streamcompiler.ir.graph import OpCode
from streamcompiler.native import require_native
from streamcompiler.runtime.execution_context import ExecutionContext
from streamcompiler.runtime.resource_names import is_host_resource
from streamcompiler.runtime.schedule_executor import (
    InstructionEvent,
    ScheduleReport,
    max_concurrency_from_intervals,
)
from streamcompiler.runtime.schedule_executor import (
    _tier_is_device as _tier_is_device_name,
)


def _move_tensor_to_resource(value: torch.Tensor, resource: str, *, enable_grad: bool = False) -> torch.Tensor:
    """Place a torch tensor on the device implied by a schedule resource id.

    Inference Transfers historically re-labeled host tensors as ``cuda_gpu_*``
    without calling ``.to``, so Compute ran on CPU and outputs looked host-side
    despite a GPU plan. Training already moved via :func:`move_for_training`;
    inference uses the same residency rule with a plain ``.to``.
    """
    name = resource.lower()
    if "mock" in name:
        return value
    if is_host_resource(name):
        if value.device.type == "cpu":
            return value
        if enable_grad:
            from streamcompiler.runtime.grad_transfer import move_for_training

            return move_for_training(value, torch.device("cpu"))
        return value.to("cpu")

    from streamcompiler.backends import backend_by_id, backend_id_for_resource

    backend_id = backend_id_for_resource(resource)
    if backend_id == "cpu":
        raise RuntimePlanError(f"Transfer targets unknown non-host resource {resource!r}")
    backend = backend_by_id(backend_id)
    if backend is None:
        raise RuntimePlanError(f"Transfer targets unavailable backend {backend_id!r} for resource {resource!r}")
    try:
        torch_device = backend.resource_to_torch_device(resource)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimePlanError(f"Backend {backend_id!r} cannot map transfer resource {resource!r}: {exc}") from exc
    target = torch.device(torch_device)
    if value.device == target:
        return value
    if enable_grad:
        from streamcompiler.runtime.grad_transfer import move_for_training

        return move_for_training(value, target)
    return value.to(target)


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
    Resident packs have no parameter_materialize Load ops — weights are seeded here
    as artifact initial residency before the schedule runs.

    Tensor objects are cached on the executor so repeated forwards skip pack
    ``acquire``; each forward still puts into a fresh native residency session
    bound to that forward's cancel token.
    """
    if getattr(executor.parameter_store, "needs_prefetch", False):
        return

    dest = ctx.host_resource
    tier = "system_ram"
    cache = getattr(executor, "_persistent_param_cache", None)
    if cache is None:
        seen: set[str] = set()
        entries: list[tuple[str, str, Any, int]] = []
        env_names = list(getattr(executor.program, "state_bindings", {}) or {})
        if not env_names:
            for binding in (getattr(executor, "bindings", {}) or {}).values():
                for name in getattr(binding.region, "state_inputs", ()) or ():
                    env_names.append(str(name))
        for env_name in env_names:
            if env_name in seen:
                continue
            seen.add(env_name)
            tensor = executor.parameter_store.acquire(env_name)
            nbytes = int(getattr(tensor, "nbytes", 0) or 0)
            entries.append((env_name, env_name, tensor, nbytes))
            target = executor.program.state_bindings.get(env_name, env_name)
            if target != env_name:
                entries.append((target, env_name, tensor, nbytes))
        executor._persistent_param_cache = entries
        cache = entries

    for name, _src, tensor, nbytes in cache:
        if not ctx.copies.has(name, dest, valid_only=True):
            ctx.copies.put(name, dest, tensor, tier=tier, ownership="parameter")
        else:
            ctx.copies.replace_handle(name, dest, tensor, tier=tier)
        ctx.mirror_native_put(name, dest, tensor, nbytes=nbytes)
        _alias_host_compute_resources(executor, ctx, name, dest)


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


def _configure_virtual_backends(native_ctx: Any, executor: Any) -> None:
    """Seed VirtualBackend capacity/timing from ResourceGraph + host priors."""
    mock_resources = sorted(
        {str(inst.resource) for inst in executor.schedule.instructions if "mock" in str(inst.resource).lower()}
    )
    if not mock_resources:
        return
    machine = getattr(executor, "machine", None)
    priors: dict[str, Any] | None = None
    for resource in mock_resources:
        memory_bytes: int | None = None
        bw: float | None = None
        lat: float | None = None
        delay: float | None = None
        if machine is not None:
            comp = machine.compute.get(resource)
            if comp is not None:
                delay = float(comp.attributes.get("mock_delay_s") or 0.05)
                for mem_name in comp.memory_affinity:
                    mem = machine.memory.get(mem_name)
                    if mem is not None:
                        memory_bytes = int(mem.allocatable_bytes or mem.capacity_bytes)
                        mem_names = {mem_name, resource}
                        for link in machine.links.values():
                            ends = {link.source, link.destination}
                            if ends & mem_names:
                                if link.bytes_per_s:
                                    bw = float(link.bytes_per_s)
                                if link.latency_s is not None:
                                    lat = float(link.latency_s)
                                break
                        break
        # Hot path: only reuse an already-filled cache — never measure here.
        if bw is None or lat is None:
            if priors is None:
                from streamcompiler.planner.cost.calibration import cached_host_priors

                priors = cached_host_priors()
            if bw is None and priors.get("beta_bytes_per_s") is not None:
                bw = float(priors["beta_bytes_per_s"])
            if lat is None and priors.get("alpha_s") is not None:
                lat = float(priors["alpha_s"])
        kwargs: dict[str, Any] = {}
        if memory_bytes is not None:
            kwargs["memory_bytes"] = int(memory_bytes)
        if bw is not None:
            kwargs["transfer_bandwidth_bytes_per_s"] = float(bw)
        if lat is not None:
            kwargs["transfer_latency_s"] = float(lat)
        if delay is not None:
            kwargs["compute_delay_s"] = float(delay)
        if kwargs and hasattr(native_ctx, "set_virtual_backend_config"):
            native_ctx.set_virtual_backend_config(resource, **kwargs)


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
    )
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

    # Fresh context per forward so cancel_token, events, and residency lifetime
    # match this run (never reuse a sticky cancel flag across forwards).
    needs_spill = _schedule_needs_spill_callbacks(executor)
    native_ctx = native.NativeExecutionContext(cancel_token=run_cancel)
    shared_execution_id = int(native_ctx.execution_id)
    ctx.native_execution_context = native_ctx
    ctx.native_residency = NativeResidencyBridge.create_from_context(native_ctx)
    _configure_virtual_backends(native_ctx, executor)

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

        def _run_one(region_id: str, _inputs: list[str], _outputs: list[str]) -> InstructionEvent:
            if ctx.cancellation.cancelled or run_cancel.is_cancelled():
                run_cancel.cancel()
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

        def parameter_load_cb(pairs: list[tuple[str, str]]) -> list[int]:
            """Materialize onto each Load's destination resource (not a guessed host)."""
            sizes: list[int] = []
            for tensor_id, dest in pairs:
                dest = str(dest or ctx.host_resource)
                tensor = executor.parameter_store.acquire(tensor_id)
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
                # Also publish under the runtime host label when Load targeted pinned RAM.
                if dest != ctx.host_resource and not ctx.copies.has(tensor_id, ctx.host_resource):
                    ctx.copies.alias(tensor_id, dest, ctx.host_resource)
                    if ctx.native_residency is not None:
                        ctx.native_residency.mirror_alias(tensor_id, dest, ctx.host_resource)
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
            if ctx.copies.has(tensor_id, rid):
                ctx.copies.drop(tensor_id, rid)
            if ctx.native_residency is not None:
                ctx.native_residency.drop_python_only(tensor_id, rid)
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
            if ctx.copies.has(tensor_id, dst, valid_only=True):
                continue
            src_copy = ctx.copies.try_get(tensor_id, src)
            if src_copy is None:
                if ctx.native_residency is not None and ctx.native_residency.session.has(tensor_id, src):
                    value = ctx.native_residency.require_value(tensor_id, src)
                else:
                    continue
            else:
                value = src_copy.value
            from streamcompiler.runtime.virtual_tensor import VirtualDeviceTensor, wrap_virtual_native

            # Schedule training keeps live tensors: virtual byte wrap detaches grads.
            # Inference must still ``.to`` real CUDA/ROCm resources — residency labels
            # alone do not move storage.
            if not ctx.enable_grad:
                if "mock" in dst.lower() and not isinstance(value, VirtualDeviceTensor):
                    value = wrap_virtual_native(value, dst, native_ctx)
                elif "mock" not in dst.lower() and isinstance(value, VirtualDeviceTensor):
                    value = value.to_host()
                elif isinstance(value, torch.Tensor):
                    value = _move_tensor_to_resource(value, dst, enable_grad=False)
            elif isinstance(value, VirtualDeviceTensor):
                value = value.payload
            elif isinstance(value, torch.Tensor):
                value = _move_tensor_to_resource(value, dst, enable_grad=True)
            ownership = "transfer"
            if src_copy is not None:
                ownership = str(getattr(src_copy, "ownership", None) or "transfer")
            else:
                for rid in ctx.copies.resources_for(tensor_id):
                    c = ctx.copies.try_get(tensor_id, rid)
                    if c is not None and c.ownership == "activation":
                        ownership = "activation"
                        break
            ctx.copies.replicate(
                tensor_id,
                dst,
                value,
                ownership=ownership,
                source_resource=src,
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
            # Replicate into Python handle table + Rust external handle without sibling invalidation.
            if ctx.native_residency is not None:
                ctx.native_residency.mirror_put(
                    tensor_id,
                    dst,
                    value,
                    nbytes=int(nbytes or getattr(value, "nbytes", 0) or 0),
                    authoritative=False,
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
    # Rust AllocationTable is authoritative; Python table is unused on native path.
    report.allocation_peak_bytes = int(native_report.get("allocation_peak_bytes") or 0)
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
            for inst in executor.schedule.instructions:
                res = str(inst.resource)
                if "mock" in res.lower():
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
