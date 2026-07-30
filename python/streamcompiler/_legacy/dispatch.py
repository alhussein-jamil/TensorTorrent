"""Legacy Python DAG instruction bodies (oracle / benches only).

Production forwards use :mod:`streamcompiler.runtime.native_bridge`.
These helpers exist so :mod:`streamcompiler._legacy.runtime` and unit
tests can exercise the pre-native schedule path without bloating
:class:`ScheduleExecutor`.
"""

# Oracle path — keep typing light; production is native_bridge + Rust.
# mypy: ignore-errors

from __future__ import annotations

import contextlib
import time
from concurrent.futures import Future
from typing import Any

import torch

from streamcompiler.errors import MemoryCapacityError, RuntimePlanError, StorageError
from streamcompiler.ir.graph import OpCode
from streamcompiler.runtime.execution_context import ExecutionContext
from streamcompiler.runtime.schedule import PlanInstruction
from streamcompiler.runtime.streams import StreamEvent
from streamcompiler.runtime.transfers import select_transfer_backend

# Lazy imports from schedule_executor avoid circular import at module load.


def _se():
    from streamcompiler.runtime import schedule_executor as se

    return se


def _copy_tier(memory_tier):
    return _se()._copy_tier(memory_tier)


def _ensure_pinned(value):
    return _se()._ensure_pinned(value)


def _tier_is_device(resource):
    return _se()._tier_is_device(resource)


def dispatch(
    executor,
    inst: PlanInstruction,
    ctx: ExecutionContext,
    submitted: float,
) -> Future[Any]:
    opcode = inst.opcode
    if opcode == OpCode.PREFETCH:
        return executor._submit_sync(lambda: executor._exec_prefetch(inst, ctx, submitted))
    if opcode == OpCode.LOAD:
        return executor._submit_sync(lambda: executor._exec_load(inst, ctx, submitted))
    if opcode == OpCode.TRANSFER:
        return executor._submit_transfer(inst, ctx, submitted)
    if opcode == OpCode.RECORD_EVENT:
        return executor._submit_sync(lambda: executor._exec_record(inst, ctx, submitted))
    if opcode == OpCode.WAIT_EVENT:
        return executor._submit_sync(lambda: executor._exec_wait(inst, ctx, submitted))
    if opcode == OpCode.COMPUTE:
        return executor._submit_compute(inst, ctx, submitted)
    if opcode == OpCode.RELEASE:
        return executor._submit_sync(lambda: executor._exec_release(inst, ctx, submitted))
    if opcode == OpCode.EVICT:
        return executor._submit_sync(lambda: executor._exec_evict(inst, ctx, submitted))
    raise RuntimePlanError(f"Unsupported schedule opcode {opcode}")


def submit_sync(executor, fn: Any) -> Future[Any]:
    if executor._closed:
        fut: Future[Any] = Future()
        fut.set_exception(RuntimePlanError("ScheduleExecutor is closed"))
        return fut
    return executor._ensure_sync_pool().submit(fn)


def exec_prefetch(executor, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Any:
    start = time.perf_counter()
    tensor_id = inst.inputs[0] if inst.inputs else ""
    hit = False
    nbytes = inst.nbytes
    if tensor_id and getattr(executor.parameter_store, "needs_prefetch", False):
        names = executor._state_env_names(inst)
        try:
            executor.parameter_store.prefetch(tuple(names))
        except (StorageError, MemoryCapacityError, RuntimePlanError) as exc:
            end = time.perf_counter()
            return _se().InstructionEvent(
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
    return _se().InstructionEvent(
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


def exec_load(executor, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Any:
    start = time.perf_counter()
    kind = str(inst.attributes.get("kind") or "")
    if kind == "activation_reload":
        return executor._exec_activation_reload(inst, ctx, submitted)

    stall0 = time.perf_counter()
    names = executor._state_env_names(inst)
    nbytes = 0
    # Load always materializes into host-accessible RAM — never device VRAM.
    dest = str(inst.destination or inst.resource)
    if _tier_is_device(dest):
        dest = ctx.host_resource
    before: dict[str, Any] = {}
    store_stats = getattr(executor.parameter_store, "stats", None)
    if callable(store_stats):
        before = dict(store_stats())
    tier = _copy_tier(inst.memory_tier)
    for env_name in names:
        tensor = executor.parameter_store.acquire(env_name)
        if tier == "pinned_ram":
            tensor = _ensure_pinned(tensor)
        target = executor.program.state_bindings.get(env_name, env_name)
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
    return _se().InstructionEvent(
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


def exec_activation_reload(executor, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Any:
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
    return _se().InstructionEvent(
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


def submit_transfer(executor, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Future[Any]:
    tensor_id = inst.inputs[0] if inst.inputs else ""
    dest = str(inst.destination or inst.resource)
    src = str(inst.source or ctx.host_resource)
    key = (tensor_id, dest)
    st = ctx.state_for(inst.name)

    def _body() -> Any:
        enqueue_start = time.perf_counter()
        existing_dest = ctx.copies.try_get(tensor_id, dest)
        if existing_dest is not None:
            ready = existing_dest.ready_event
            incomplete = False
            if ready is not None:
                if hasattr(ready, "is_complete"):
                    incomplete = not bool(ready.is_complete())
                elif hasattr(ready, "done"):
                    incomplete = not bool(ready.done())
            if not incomplete:
                end = time.perf_counter()
                return _se().InstructionEvent(
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
        with executor._transfer_lock:
            existing = ctx.pending_transfers.get(key)
        if existing is not None and not existing.done():
            st.async_future = existing
            st.enqueue_start_s = enqueue_start
            st.enqueue_end_s = time.perf_counter()
            end = time.perf_counter()
            return _se().InstructionEvent(
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
                f"Transfer {inst.name}: required source copy missing tensor={tensor_id!r} source={src_resource!r}"
            ) from exc

        backend = select_transfer_backend(inst.transfer_backend, destination=dest)
        delay = float(inst.attributes.get("mock_transfer_delay_s", 0.0))
        is_mock = "mock" in dest.lower() or delay > 0 or backend.backend_id == "simulated_device"
        stream = executor.streams.copy_stream(dest, delay_s=delay if is_mock else 0.0)
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
            with executor._transfer_lock:
                ctx.pending_transfers.pop(key, None)
            return result

        fut = stream.submit(_xfer, delay_s=delay if is_mock else 0.0)
        enqueue_end = time.perf_counter()
        pending_event.bind_future(
            fut,
            enqueue_start_s=enqueue_start,
            enqueue_end_s=enqueue_end,
        )
        with executor._transfer_lock:
            ctx.pending_transfers[key] = fut
        st.async_future = fut
        st.enqueue_start_s = enqueue_start
        st.enqueue_end_s = enqueue_end
        st.completion_event = pending_event
        return _se().InstructionEvent(
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

    return executor._submit_sync(_body)


def exec_record(executor, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Any:
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
    return _se().InstructionEvent(
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


def exec_wait(executor, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Any:
    start = time.perf_counter()
    waits_for = str(inst.attributes.get("waits_for") or (inst.depends_on[0] if inst.depends_on else ""))
    event = ctx.events.get(waits_for)
    wait0 = time.perf_counter()
    event.wait()
    wait_s = time.perf_counter() - wait0
    ctx.state_for(inst.name).wait_duration_s = wait_s
    end = time.perf_counter()
    return _se().InstructionEvent(
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


def submit_compute(executor, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Future[Any]:
    delay = float(inst.attributes.get("mock_compute_delay_s", 0.0))
    binding = executor.bindings[str(inst.executable_ref or "")]
    resource = binding.device
    if delay <= 0 and "mock" in resource:
        delay = (
            float(binding.compiled.attributes.get("mock_delay_s", 0.05))
            if hasattr(binding.compiled, "attributes")
            else 0.05
        )
    # Async only when mock delay or process workers need a real stream/pool.
    if delay <= 0 and executor.process_pool is None:
        out: Future[Any] = Future()
        try:
            out.set_result(executor._exec_compute(inst, ctx, submitted))
        except BaseException as exc:
            out.set_exception(exc)
        return out

    stream = executor.streams.compute_stream(
        resource,
        delay_s=delay if delay > 0 else 0.0,
        workers=max(1, executor.max_inflight),
    )
    fut = stream.submit(lambda: executor._exec_compute(inst, ctx, submitted), delay_s=delay if delay > 0 else 0.0)
    out2: Future[Any] = Future()

    def _done(f: Future[Any]) -> None:
        try:
            out2.set_result(f.result())
        except Exception as exc:
            out2.set_exception(exc)

    fut.add_done_callback(_done)
    return out2


def exec_release(executor, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Any:
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
        if ctx.native_residency is not None and ctx.native_residency.session.has(tensor_id, resource):
            ctx.native_residency.release(tensor_id, resource)
        freed += ctx.copies.drop(tensor_id, resource)
        if tensor_id in executor.program.state_bindings or tensor_id in executor.program.state_bindings.values():
            executor.parameter_store.release((tensor_id,))
    end = time.perf_counter()
    return _se().InstructionEvent(
        name=inst.name,
        opcode=inst.opcode.value,
        resource=str(inst.resource),
        submitted_s=submitted,
        start_s=start,
        end_s=end,
        nbytes=freed,
        notes="schedule Release",
    )


def exec_evict(executor, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Any:
    start = time.perf_counter()
    kind = str(inst.attributes.get("kind") or "")
    if kind == "activation_spill":
        return executor._exec_activation_spill(inst, ctx, submitted)
    freed = 0
    for tensor_id in inst.inputs:
        resource = str(inst.destination or inst.resource)
        if not ctx.copies.has(tensor_id, resource):
            continue
        freed += ctx.copies.drop(tensor_id, resource)
        if ctx.native_residency is not None and ctx.native_residency.session.has(tensor_id, resource):
            ctx.native_residency.release(tensor_id, resource)
        if tensor_id in executor.program.state_bindings:
            executor.parameter_store.release((tensor_id,))
    end = time.perf_counter()
    return _se().InstructionEvent(
        name=inst.name,
        opcode=inst.opcode.value,
        resource=str(inst.resource),
        submitted_s=submitted,
        start_s=start,
        end_s=end,
        nbytes=freed,
        notes="schedule Evict",
    )


def exec_activation_spill(executor, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> Any:
    from streamcompiler.runtime.activation_spill import is_spilled, spill_tensor
    from streamcompiler.runtime.virtual_tensor import VirtualDeviceTensor

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
            *ctx.copies.resources_for(tensor_id),
        ):
            if alt == "disk":
                continue
            cand = ctx.copies.try_get(tensor_id, alt)
            if cand is None:
                continue
            if isinstance(cand.value, (torch.Tensor, VirtualDeviceTensor)):
                copy = cand
                resource = alt
                break
        if copy is None:
            disk = ctx.copies.try_get(tensor_id, "disk")
            if disk is not None and is_spilled(disk.value):
                # Fully spilled: no host/device tensor residency remains.
                freed += int(disk.nbytes)
                continue
            raise RuntimePlanError(
                f"activation_spill {inst.name}: required copy of {tensor_id!r} missing on {resource!r}"
            )

        spill_src = copy.value.to_host() if isinstance(copy.value, VirtualDeviceTensor) else copy.value
        spilled = spill_tensor(spill_src)
        ownership = copy.ownership if copy.ownership == "activation" else "activation"
        # Drop every non-disk label, then install the disk spill handle.
        for rid in list(ctx.copies.resources_for(tensor_id)):
            if rid == "disk":
                continue
            ctx.copies.drop(tensor_id, rid)
        old_disk = ctx.copies.try_get(tensor_id, "disk")
        if old_disk is not None:
            ctx.copies.drop(tensor_id, "disk")
            if is_spilled(old_disk.value):
                with contextlib.suppress(OSError):
                    old_disk.value.path.unlink(missing_ok=True)
        ctx.copies.put(
            tensor_id,
            "disk",
            spilled,
            tier="disk",
            ownership=ownership,
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
    return _se().InstructionEvent(
        name=inst.name,
        opcode=inst.opcode.value,
        resource=str(inst.resource),
        submitted_s=submitted,
        start_s=start,
        end_s=end,
        nbytes=freed,
        notes="schedule Evict activation RAM→disk",
    )
