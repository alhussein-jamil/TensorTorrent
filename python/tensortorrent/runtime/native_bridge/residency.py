"""Tensor residency helpers for native schedule execution."""

from __future__ import annotations

from typing import Any

import torch

from tensortorrent.errors import RuntimePlanError
from tensortorrent.runtime.execution_context import ExecutionContext
from tensortorrent.runtime.resource_names import is_host_resource

_resource_torch_device_cache: dict[str, Any] = {}


def _param_cache_signature(tensor: Any, *, version: int | None = None) -> tuple[int, int]:
    """Stable (id, version) key for persistent device-parameter cache entries."""
    ver = int(version) if version is not None else int(getattr(tensor, "_version", 0))
    return (id(tensor), ver)


def _torch_device_for_resource(resource: str) -> Any | None:
    """Map a schedule resource id to ``torch.device``, with a process-local cache.

    Returns ``None`` for mock resources and for non-host ids that do not resolve
    to a real accelerator backend (``backend_id_for_resource`` defaults to
    ``\"cpu\"`` for unknowns — that must not become a silent CPU placement).
    """
    key = str(resource or "")
    if key in _resource_torch_device_cache:
        return _resource_torch_device_cache[key]
    name = key.lower()
    device: Any | None
    if not name or "mock" in name:
        device = None
    elif is_host_resource(name):
        device = torch.device("cpu")
    else:
        from tensortorrent.backends import backend_by_id, backend_id_for_resource

        backend_id = backend_id_for_resource(key)
        # cpu fallback means "unrecognized resource", not host placement.
        if backend_id == "cpu":
            device = None
        else:
            backend = backend_by_id(backend_id)
            if backend is None:
                device = None
            else:
                try:
                    mapped = backend.resource_to_torch_device(key)
                    device = mapped if isinstance(mapped, torch.device) else torch.device(mapped)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    device = None
    _resource_torch_device_cache[key] = device
    return device


def _tensor_already_on_resource(value: Any, resource: str) -> bool:
    """True when ``value`` is a torch tensor already placed on ``resource``."""
    if not isinstance(value, torch.Tensor):
        return False
    target = _torch_device_for_resource(resource)
    if target is None:
        return False
    return bool(value.device == target)


def _move_tensor_to_resource(
    value: torch.Tensor,
    resource: str,
    *,
    enable_grad: bool = False,
) -> torch.Tensor:
    """Place a torch tensor on the device implied by a schedule resource id.

    Inference uses pinned-aware ``.to`` (non-blocking when the host buffer is
    page-locked). Training goes through :func:`move_for_training`.
    Unknown non-host resources fail closed (never silently relabel as CPU).
    """
    name = resource.lower()
    if "mock" in name:
        return value
    if is_host_resource(name):
        target = torch.device("cpu")
        if value.device == target:
            return value
        if enable_grad:
            from tensortorrent.runtime.grad_transfer import move_for_training

            return move_for_training(value, target)
        return value.to("cpu")

    from tensortorrent.backends import backend_by_id, backend_id_for_resource

    backend_id = backend_id_for_resource(resource)
    if backend_id == "cpu":
        raise RuntimePlanError(f"Transfer targets unknown non-host resource {resource!r}")
    backend = backend_by_id(backend_id)
    if backend is None:
        raise RuntimePlanError(f"Transfer targets unavailable backend {backend_id!r} for resource {resource!r}")
    try:
        mapped = backend.resource_to_torch_device(resource)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimePlanError(f"Backend {backend_id!r} cannot map transfer resource {resource!r}: {exc}") from exc
    target = mapped if isinstance(mapped, torch.device) else torch.device(mapped)
    _resource_torch_device_cache[str(resource)] = target
    if value.device == target:
        return value
    if enable_grad:
        from tensortorrent.runtime.grad_transfer import move_for_training

        return move_for_training(value, target)
    return value.to(target, non_blocking=bool(value.is_pinned()))


def _schedule_needs_spill_callbacks(executor: Any) -> bool:
    return bool(executor._needs_spill_callbacks)


def _schedule_needs_parameter_load(executor: Any) -> bool:
    return bool(executor._needs_parameter_load)


def _is_memory_exhaustion(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cannot allocate memory" in message


def _register_persistent_residency(executor: Any, ctx: ExecutionContext) -> None:
    """Register already-resident parameters into value bag + native residency.

    Does **not** run schedule Load. Resident packs seed weights here before the
    schedule runs. Inference may hoist device copies into an executor cache and
    skip host republish when a device copy is present. Training never consumes
    the device cache (optimizers mutate host params); cache entries track source
    identity + version so ``load_state_dict`` invalidates them.
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
        hoist_device = bool(getattr(executor, "_hoist_resident_parameters", True)) and not bool(
            getattr(executor, "_partial_hoist_oom", False)
        )
        if not ctx.enable_grad and hoist_device and executor._resident_parameter_targets:
            by_name = {name: entry for entry in host_entries for name in (entry[0],)}
            try:
                for name, destinations in list(executor._resident_parameter_targets.items()):
                    source_entry = by_name.get(name)
                    if source_entry is None:
                        # Transfer was dropped for this id — missing host source is fatal.
                        raise RuntimePlanError(f"persistent parameter {name!r} has no host copy to hoist")
                    source = source_entry[2]
                    signature = _param_cache_signature(source)
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
            except Exception as exc:
                if not _is_memory_exhaustion(exc):
                    raise
                # OOM this generation — stream via transfer/evict; keep hoist config.
                executor.release_device_residency(demote_hoist=False)
                device_entries = []

    device_names = {name for name, _resource, _tensor, _nbytes, _copy_meta, _view_meta in device_entries}
    for name, _src, tensor, nbytes, copy_meta, view_meta in host_entries:
        if name in device_names:
            continue
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
        ctx.publish_tensor(
            name,
            resource,
            tensor,
            tier="device",
            ownership="parameter",
            nbytes=nbytes,
            view_meta=view_meta,
            precomputed=copy_meta,
            authoritative=True,
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
