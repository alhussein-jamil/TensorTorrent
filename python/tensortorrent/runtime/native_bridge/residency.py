"""Tensor residency helpers for native schedule execution."""

from __future__ import annotations

from typing import Any

import torch

from tensortorrent.errors import RuntimePlanError
from tensortorrent.runtime.execution_context import ExecutionContext
from tensortorrent.runtime.resource_names import is_host_resource


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
            from tensortorrent.runtime.grad_transfer import move_for_training

            return move_for_training(value, torch.device("cpu"))
        return value.to("cpu")

    from tensortorrent.backends import backend_by_id, backend_id_for_resource

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
        from tensortorrent.runtime.grad_transfer import move_for_training

        return move_for_training(value, target)
    return value.to(target)


def _schedule_needs_spill_callbacks(executor: Any) -> bool:
    return bool(executor._needs_spill_callbacks)


def _schedule_needs_parameter_load(executor: Any) -> bool:
    return bool(executor._needs_parameter_load)


def _register_persistent_residency(executor: Any, ctx: ExecutionContext) -> None:
    """Register already-resident parameters into value bag + native residency.

    Does **not** run schedule Load and does **not** call ``_exec_load``.
    Resident packs have no parameter_materialize Load ops — weights are seeded here
    as artifact initial residency before the schedule runs.

    Tensor objects *and* their static view/copy metadata are cached on the
    executor so repeated forwards skip both pack ``acquire`` and re-deriving
    shape/stride/storage-id — the tensor identity never changes for a resident
    parameter. In inference, scheduled device copies are also hoisted into an
    executor-owned cache. Each forward only registers those stable tensors into
    its fresh residency session; the native Transfer instruction then sees the
    destination's current logical version and becomes a no-op.

    Training never consumes device-cache entries because optimizers mutate the
    host parameters. Inference cache entries carry the source tensor identity and
    PyTorch version counter, so ``load_state_dict`` or other in-place updates
    invalidate and refresh them before the next forward.
    """
    if getattr(executor.parameter_store, "needs_prefetch", False):
        return

    dest = ctx.host_resource
    tier = "system_ram"
    from tensortorrent.runtime.copies import describe_tensor
    from tensortorrent.runtime.handles import _tensor_view_meta

    with executor._persistent_param_lock:
        cache = executor._persistent_param_cache
        if cache is None:
            seen: set[str] = set()
            entries: list[tuple[str, str, Any, int, Any, dict[str, Any]]] = []
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
                copy_meta = describe_tensor(tensor, env_name, dest)
                view_meta = _tensor_view_meta(tensor)
                entries.append((env_name, env_name, tensor, copy_meta.nbytes, copy_meta, view_meta))
                target = executor.program.state_bindings.get(env_name, env_name)
                if target != env_name:
                    entries.append((target, env_name, tensor, copy_meta.nbytes, copy_meta, view_meta))
            executor._persistent_param_cache = entries
            cache = entries

        host_entries = list(cache)
        device_entries: list[tuple[str, str, Any, int, Any, dict[str, Any]]] = []
        if not ctx.enable_grad:
            by_name = {name: entry for entry in host_entries for name in (entry[0],)}
            for name, destinations in executor._resident_parameter_targets.items():
                source_entry = by_name.get(name)
                if source_entry is None:
                    continue
                source = source_entry[2]
                signature = (id(source), int(getattr(source, "_version", 0)))
                for resource in destinations:
                    key = (name, resource)
                    cached = executor._persistent_device_param_cache.get(key)
                    if cached is None or cached[0] != signature:
                        value = _move_tensor_to_resource(source, resource, enable_grad=False)
                        copy_meta = describe_tensor(value, name, resource)
                        view_meta = _tensor_view_meta(value)
                        cached = (signature, value, copy_meta.nbytes, copy_meta, view_meta)
                        executor._persistent_device_param_cache[key] = cached
                    device_entries.append((name, resource, cached[1], cached[2], cached[3], cached[4]))

    for name, _src, tensor, nbytes, copy_meta, view_meta in host_entries:
        ctx.publish_tensor(
            name,
            dest,
            tensor,
            tier=tier,
            ownership="parameter",
            nbytes=nbytes,
            view_meta=view_meta,
            precomputed=copy_meta,
        )
        _alias_host_compute_resources(executor, ctx, name, dest)
    for name, resource, tensor, nbytes, copy_meta, view_meta in device_entries:
        # Replica registration must not invalidate the host/authoritative copy.
        ctx.publish_tensor(
            name,
            resource,
            tensor,
            tier="device",
            ownership="parameter",
            nbytes=nbytes,
            view_meta=view_meta,
            precomputed=copy_meta,
            authoritative=False,
        )


def _alias_host_compute_resources(executor: Any, ctx: ExecutionContext, tensor_id: str, dest: str) -> None:
    if ctx.native_residency is None:
        return
    for res in executor._alias_target_resources:
        if res == dest:
            continue
        ctx.native_residency.mirror_alias(tensor_id, dest, res)


def _configure_virtual_backends(native_ctx: Any, executor: Any) -> None:
    """Seed VirtualBackend capacity/timing from ResourceGraph + host priors."""
    mock_resources = executor._mock_resources
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
                from tensortorrent.planner.cost.calibration import cached_host_priors

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
