"""Shared capacity leases for concurrent forwards on one CompiledModule.

Parameter stores are shared across in-flight requests. Each forward therefore
leases only its *incremental* working set (activations + streaming window),
while a one-time base reservation covers resident parameter bytes when the host
budget is an explicit absolute ceiling. Live-available host budgets (OS/cgroup
remaining) already exclude resident model RAM, so they do not deduct state
again. Admit fails closed when the next lease would exceed the resolved host /
device / disk budget.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from tensortorrent.errors import MemoryCapacityError, RuntimePlanError, TensorTorrentError


@dataclass(frozen=True, slots=True)
class CapacityLease:
    """Bytes charged to one in-flight forward (incremental, not shared params)."""

    host_bytes: int = 0
    device_bytes: int = 0
    disk_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_bytes", max(0, int(self.host_bytes)))
        object.__setattr__(self, "device_bytes", max(0, int(self.device_bytes)))
        object.__setattr__(self, "disk_bytes", max(0, int(self.disk_bytes)))


@dataclass(frozen=True, slots=True)
class CapacityBudgets:
    host_bytes: int
    device_bytes: int
    disk_bytes: int
    # Provenance from resolve_host_memory_budget (explicit / os_available / …).
    host_source_kind: str = "explicit"
    # True when allowed host bytes already exclude currently resident model RAM
    # (live remaining: os_available / cgroup). Explicit absolute budgets are False.
    host_reflects_live_remaining: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_bytes", max(0, int(self.host_bytes)))
        object.__setattr__(self, "device_bytes", max(0, int(self.device_bytes)))
        object.__setattr__(self, "disk_bytes", max(0, int(self.disk_bytes)))
        object.__setattr__(self, "host_source_kind", str(self.host_source_kind or "explicit"))
        object.__setattr__(self, "host_reflects_live_remaining", bool(self.host_reflects_live_remaining))


def host_budget_reflects_live_remaining(source_kind: str) -> bool:
    """Live-available host budgets already net out resident process memory."""
    return str(source_kind) in {"os_available", "cgroup_v2", "cgroup_v1"}


def _is_device_peak_key(key: str) -> bool:
    k = key.lower()
    return any(
        token in k
        for token in (
            "vram",
            "cuda",
            "rocm",
            "hip",
            "xpu",
            "gpu",
            "mock_accel",
            "device_vram",
        )
    )


def _plan_is_cpu_only(plan: Any | None) -> bool:
    """True when the selected plan only uses host CPU compute devices."""
    if plan is None:
        return False
    devices = [str(d) for d in (getattr(plan, "devices_used", None) or ())]
    if not devices:
        return False
    return all(d.startswith("cpu") or d.startswith("cpu_numa") for d in devices)


def _program_state_bytes(program: Any | None) -> int:
    """Return resident state bytes, including export-free fused CPU roots.

    Export-free DirectPlan programs intentionally have no state bindings because
    they execute the caller's original ``nn.Module`` directly. Their generic
    ``total_state_bytes()`` is therefore zero even though the weights remain
    resident in host RAM. Count those tensors from the root so explicit RAM
    ceilings remain truthful.
    """
    if program is None:
        return 0
    state_bytes = int(program.total_state_bytes())
    if state_bytes > 0:
        return state_bytes
    metadata = getattr(program, "metadata", None) or {}
    if not isinstance(metadata, dict) or not metadata.get("eager_fused_export_free"):
        return 0
    root = getattr(program, "root", None)
    import torch

    if not isinstance(root, torch.nn.Module):
        return 0
    return sum(int(t.numel()) * int(t.element_size()) for t in (*root.parameters(), *root.buffers()))


def _resident_device_parameter_bytes(
    *,
    state_bytes: int,
    streaming: bool,
    cpu_only: bool,
    config: Any,
    machine: Any | None,
) -> int:
    """Shared base / subtractable device-resident weight bytes for capacity math."""
    if streaming or cpu_only or state_bytes <= 0:
        return 0
    from tensortorrent.compile.fit import accelerator_hoist_budget_bytes

    # Same outcome as should_hoist_resident_parameters + partial min, one budget fetch.
    budget = accelerator_hoist_budget_bytes(config, machine)
    if budget is None:
        return 0 if bool(getattr(config, "allow_training", False)) else int(state_bytes)
    return max(0, min(int(state_bytes), int(budget)))


def _resolve_capacity_footprint(
    *,
    program: Any,
    plan: Any | None,
    config: Any,
    parameter_store: Any | None,
    machine: Any | None,
) -> tuple[int, bool, bool, int]:
    """Once: state_bytes, streaming, cpu_only, resident_device_parameter_bytes."""
    state_bytes = _program_state_bytes(program)
    streaming = bool(getattr(parameter_store, "needs_prefetch", False))
    if parameter_store is None and not streaming and hasattr(config, "ram_budget_bytes"):
        from tensortorrent.compile.fit import needs_parameter_streaming

        streaming = needs_parameter_streaming(config, state_bytes=state_bytes)
    cpu_only = _plan_is_cpu_only(plan)
    resident_device = _resident_device_parameter_bytes(
        state_bytes=state_bytes,
        streaming=streaming,
        cpu_only=cpu_only,
        config=config,
        machine=machine,
    )
    return state_bytes, streaming, cpu_only, resident_device


def estimate_request_capacity(
    *,
    program: Any,
    plan: Any | None,
    config: Any,
    parameter_store: Any | None = None,
    machine: Any | None = None,
    _footprint: tuple[int, bool, bool, int] | None = None,
) -> CapacityLease:
    """Incremental per-forward lease (shared parameter bytes excluded)."""
    peaks = dict(getattr(plan, "predicted_peak_bytes", None) or {})
    host_peak = 0
    device_peak = 0
    activation_peak = int(peaks.get("activations", 0) or 0)
    for key, value in peaks.items():
        amount = int(value or 0)
        name = str(key)
        if name == "activations":
            continue
        if _is_device_peak_key(name):
            device_peak = max(device_peak, amount)
        else:
            host_peak = max(host_peak, amount)

    if _footprint is None:
        state_bytes, streaming, cpu_only, resident_device = _resolve_capacity_footprint(
            program=program,
            plan=plan,
            config=config,
            parameter_store=parameter_store,
            machine=machine,
        )
    else:
        state_bytes, streaming, cpu_only, resident_device = _footprint

    if streaming:
        region_cap = None
        if hasattr(config, "ram_budget_bytes") and config.ram_budget_bytes is not None:
            from tensortorrent.compile.fit import streaming_region_budget

            region_cap = streaming_region_budget(config, parameter_bytes=state_bytes)
        if region_cap is None and program is not None and hasattr(program, "max_region_state_bytes"):
            region_cap = int(program.max_region_state_bytes())
        prefetch = int(getattr(config, "prefetch_distance", 1) or 0)
        window = int(region_cap or 0) * (1 + max(0, prefetch))
        host_need = max(activation_peak, max(0, host_peak - state_bytes), window)
    else:
        # Resident params are shared across in-flight requests.
        host_need = max(activation_peak, max(0, host_peak - state_bytes))

    # Device peaks from the simulator include hoisted weights when they fit.
    # Charge only the non-shared remainder so concurrency is not collapsed to 1.
    # CPU-only / fused-CPU DirectPlan selections must not reserve fake VRAM.
    if cpu_only:
        device_need = 0
    elif resident_device > 0:
        device_need = max(activation_peak, max(0, device_peak - resident_device))
    else:
        device_need = max(device_peak, activation_peak if device_peak == 0 else 0)

    disk_need = 0
    if getattr(config, "activation_budget_bytes", None) is not None:
        # Spill may write roughly one activation peak to disk for this forward.
        disk_need = max(activation_peak, int(config.activation_budget_bytes or 0) // 8)

    # Empty working-set estimates still need a non-zero lease so the byte
    # ledger can serialize concurrent admits. One byte is enough — do not
    # invent a fake MiB footprint that under-admits tiny models.
    if host_need == 0 and device_need == 0 and disk_need == 0:
        host_need = 1

    return CapacityLease(host_bytes=host_need, device_bytes=device_need, disk_bytes=disk_need)


def resolve_capacity_budgets(config: Any, *, machine: Any | None = None) -> CapacityBudgets:
    """Resolve shared host/device/disk ceilings for concurrent leases."""
    from tensortorrent.hardware.budget import (
        default_vram_headroom_bytes,
        resolve_device_memory_budget,
        resolve_host_memory_budget,
    )

    host = resolve_host_memory_budget(
        getattr(config, "ram_budget_bytes", None),
        reserve_bytes=getattr(config, "host_memory_reserve_bytes", None),
    )
    device_allowed = 0
    explicit_vram = getattr(config, "vram_budget_bytes", None)
    configured_headroom = getattr(config, "vram_headroom_bytes", None)
    if explicit_vram is not None:
        device_allowed = int(explicit_vram)
    elif machine is not None:
        from tensortorrent.ir.resource_graph import MemoryClass

        for mem in machine.memory_by_class(MemoryClass.DEVICE_VRAM):
            attrs = getattr(mem, "attributes", {}) or {}
            display = bool(attrs.get("display_attached"))
            attached = tuple(getattr(mem, "attached_compute", ()) or ())
            virtual = any(
                str(getattr(machine.compute.get(name), "backend_id", "")) in {"mock_accel", "virtual"}
                for name in attached
            )
            headroom = (
                0
                if virtual
                else int(configured_headroom)
                if configured_headroom is not None
                else default_vram_headroom_bytes(display)
            )
            total = int(getattr(mem, "capacity_bytes", 0) or 0)
            free = getattr(mem, "allocatable_bytes", None)
            free_i = int(free) if free is not None else None
            resolved = resolve_device_memory_budget(total, free_i, None, headroom)
            device_allowed += int(resolved.allowed_bytes)
    else:
        try:
            import torch

            if torch.cuda.is_available():
                headroom = (
                    int(configured_headroom) if configured_headroom is not None else default_vram_headroom_bytes(False)
                )
                for idx in range(int(torch.cuda.device_count())):
                    free, total = torch.cuda.mem_get_info(idx)
                    resolved = resolve_device_memory_budget(int(total), int(free), None, headroom)
                    device_allowed += int(resolved.allowed_bytes)
        except Exception:  # noqa: BLE001 - CPU-only hosts
            device_allowed = 0

    disk = getattr(config, "max_total_spill_bytes", None)
    if disk is None:
        disk = 0
        if getattr(config, "activation_budget_bytes", None) is not None:
            disk = 1 << 30  # 1 GiB default spill headroom when spill is enabled

    return CapacityBudgets(
        host_bytes=int(host.allowed_bytes),
        device_bytes=max(0, device_allowed),
        disk_bytes=max(0, int(disk)),
        host_source_kind=str(getattr(host.source, "kind", "explicit") or "explicit"),
        host_reflects_live_remaining=host_budget_reflects_live_remaining(str(getattr(host.source, "kind", "") or "")),
    )


class CapacityLedger:
    """Thread-safe byte ledger for concurrent request admits."""

    def __init__(
        self,
        budgets: CapacityBudgets,
        *,
        per_request: CapacityLease,
        base: CapacityLease | None = None,
    ) -> None:
        base = base or CapacityLease()
        for resource, reserved, available in (
            ("host", base.host_bytes, budgets.host_bytes),
            ("device", base.device_bytes, budgets.device_bytes),
            ("disk", base.disk_bytes, budgets.disk_bytes),
        ):
            if reserved > available:
                raise MemoryCapacityError(
                    f"shared {resource} reservation exceeds capacity: reserved={reserved} available={available}"
                )
        # Shared parameter footprint reserved once; remaining is for in-flight leases.
        self._budgets = CapacityBudgets(
            host_bytes=budgets.host_bytes - base.host_bytes,
            device_bytes=budgets.device_bytes - base.device_bytes,
            disk_bytes=budgets.disk_bytes - base.disk_bytes,
            host_source_kind=budgets.host_source_kind,
            host_reflects_live_remaining=budgets.host_reflects_live_remaining,
        )
        self._per_request = per_request
        self._lock = threading.Lock()
        self._used_host = 0
        self._used_device = 0
        self._used_disk = 0
        self._inflight = 0

    @property
    def per_request(self) -> CapacityLease:
        return self._per_request

    @property
    def budgets(self) -> CapacityBudgets:
        return self._budgets

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def max_concurrent(self) -> int:
        """Upper bound on simultaneous leases from remaining budgets."""
        limits: list[int] = []
        need = self._per_request
        if need.host_bytes > 0:
            limits.append(self._budgets.host_bytes // need.host_bytes)
        if need.device_bytes > 0:
            limits.append(self._budgets.device_bytes // need.device_bytes)
        if need.disk_bytes > 0:
            limits.append(self._budgets.disk_bytes // need.disk_bytes)
        if not limits:
            return 1 << 30
        return max(0, min(limits))

    def try_acquire(self, need: CapacityLease | None = None) -> bool:
        lease = need if need is not None else self._per_request
        with self._lock:
            if self._used_host + lease.host_bytes > self._budgets.host_bytes:
                return False
            if lease.device_bytes > 0 and self._used_device + lease.device_bytes > self._budgets.device_bytes:
                return False
            if lease.disk_bytes > 0 and self._used_disk + lease.disk_bytes > self._budgets.disk_bytes:
                return False
            self._used_host += lease.host_bytes
            self._used_device += lease.device_bytes
            self._used_disk += lease.disk_bytes
            self._inflight += 1
            return True

    def release(self, need: CapacityLease | None = None) -> None:
        lease = need if need is not None else self._per_request
        with self._lock:
            if (
                self._inflight <= 0
                or self._used_host < lease.host_bytes
                or self._used_device < lease.device_bytes
                or self._used_disk < lease.disk_bytes
            ):
                raise RuntimePlanError("CapacityLedger release without matching acquisition")
            self._used_host -= lease.host_bytes
            self._used_device -= lease.device_bytes
            self._used_disk -= lease.disk_bytes
            self._inflight -= 1

    def acquire_or_raise(self, need: CapacityLease | None = None, *, model_id: str | None = None) -> None:
        if self.try_acquire(need):
            return
        lease = need if need is not None else self._per_request
        where = f" model {model_id}" if model_id else ""
        with self._lock:
            inflight = self._inflight
            host_budget = self._budgets.host_bytes
            host_need = lease.host_bytes
            max_n = self._budgets.host_bytes // lease.host_bytes if lease.host_bytes > 0 else (1 << 30)
        raise TensorTorrentError(
            f"backpressure:{where} capacity exhausted "
            f"(need host={lease.host_bytes} device={lease.device_bytes} disk={lease.disk_bytes}; "
            f"inflight={inflight} max≈{max_n} host_budget={host_budget} host_need={host_need})"
        )


def build_module_capacity_ledger(
    *,
    program: Any,
    plan: Any | None,
    config: Any,
    parameter_store: Any | None,
    machine: Any | None = None,
) -> CapacityLedger:
    """Build the shared ledger for one compiled module generation."""
    footprint = _resolve_capacity_footprint(
        program=program,
        plan=plan,
        config=config,
        parameter_store=parameter_store,
        machine=machine,
    )
    state_bytes, streaming, _cpu_only, device_base = footprint
    budgets = resolve_capacity_budgets(config, machine=machine)
    # Live-available host budgets (psutil/cgroup remaining) already exclude the
    # resident model. Deducting state_bytes again would double-count. Explicit
    # absolute RAM budgets still need a base reservation.
    host_base = 0 if streaming or budgets.host_reflects_live_remaining else state_bytes
    base = CapacityLease(
        host_bytes=host_base,
        device_bytes=device_base,
        disk_bytes=0,
    )
    per_request = estimate_request_capacity(
        program=program,
        plan=plan,
        config=config,
        parameter_store=parameter_store,
        machine=machine,
        _footprint=footprint,
    )
    return CapacityLedger(budgets, per_request=per_request, base=base)
