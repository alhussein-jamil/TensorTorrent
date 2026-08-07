"""Shared capacity leases for concurrent forwards on one CompiledModule.

Parameter stores are shared across in-flight requests. Each forward therefore
leases only its *incremental* working set (activations + streaming window),
while a one-time base reservation covers resident parameter bytes. Admit fails
closed when the next lease would exceed the resolved host / device / disk budget.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from tensortorrent.errors import TensorTorrentError

# Serve path acquires the lease in ModelManager; forward skips a second lease.
_CAPACITY_PREHELD: ContextVar[bool] = ContextVar("tt_capacity_preheld", default=False)


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_bytes", max(0, int(self.host_bytes)))
        object.__setattr__(self, "device_bytes", max(0, int(self.device_bytes)))
        object.__setattr__(self, "disk_bytes", max(0, int(self.disk_bytes)))


def capacity_preheld() -> bool:
    return bool(_CAPACITY_PREHELD.get())


@contextlib.contextmanager
def capacity_preheld_scope() -> Iterator[None]:
    token = _CAPACITY_PREHELD.set(True)
    try:
        yield
    finally:
        _CAPACITY_PREHELD.reset(token)


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


def estimate_request_capacity(
    *,
    program: Any,
    plan: Any | None,
    config: Any,
    parameter_store: Any | None = None,
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

    state_bytes = int(program.total_state_bytes()) if program is not None else 0
    streaming = bool(getattr(parameter_store, "needs_prefetch", False))
    if not streaming and hasattr(config, "ram_budget_bytes"):
        from tensortorrent.compile.fit import needs_parameter_streaming

        streaming = needs_parameter_streaming(config, state_bytes=state_bytes)

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
    from tensortorrent.compile.fit import should_hoist_resident_parameters

    hoist = (not streaming) and should_hoist_resident_parameters(config, state_bytes=state_bytes)
    if hoist and state_bytes > 0:
        device_need = max(activation_peak, max(0, device_peak - state_bytes))
    else:
        device_need = max(device_peak, activation_peak if device_peak == 0 else 0)

    disk_need = 0
    if getattr(config, "activation_budget_bytes", None) is not None:
        # Spill may write roughly one activation peak to disk for this forward.
        disk_need = max(activation_peak, int(config.activation_budget_bytes or 0) // 8)

    # Floor so empty estimates still serialize concurrent admits under a byte ledger.
    if host_need == 0 and device_need == 0 and disk_need == 0:
        host_need = max(1 << 20, activation_peak)  # 1 MiB minimum host lease

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
            headroom = (
                int(configured_headroom) if configured_headroom is not None else default_vram_headroom_bytes(display)
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
        # Shared parameter footprint reserved once; remaining is for in-flight leases.
        self._budgets = CapacityBudgets(
            host_bytes=max(0, budgets.host_bytes - base.host_bytes),
            device_bytes=max(0, budgets.device_bytes - base.device_bytes),
            disk_bytes=max(0, budgets.disk_bytes - base.disk_bytes),
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
        if need.device_bytes > 0 and self._budgets.device_bytes > 0:
            limits.append(self._budgets.device_bytes // need.device_bytes)
        if need.disk_bytes > 0 and self._budgets.disk_bytes > 0:
            limits.append(self._budgets.disk_bytes // need.disk_bytes)
        if not limits:
            return 1 << 30
        return max(0, min(limits))

    def try_acquire(self, need: CapacityLease | None = None) -> bool:
        lease = need if need is not None else self._per_request
        with self._lock:
            if self._used_host + lease.host_bytes > self._budgets.host_bytes:
                return False
            if (
                lease.device_bytes > 0
                and self._budgets.device_bytes > 0
                and self._used_device + lease.device_bytes > self._budgets.device_bytes
            ):
                return False
            if (
                lease.disk_bytes > 0
                and self._budgets.disk_bytes > 0
                and self._used_disk + lease.disk_bytes > self._budgets.disk_bytes
            ):
                return False
            self._used_host += lease.host_bytes
            self._used_device += lease.device_bytes
            self._used_disk += lease.disk_bytes
            self._inflight += 1
            return True

    def release(self, need: CapacityLease | None = None) -> None:
        lease = need if need is not None else self._per_request
        with self._lock:
            self._used_host = max(0, self._used_host - lease.host_bytes)
            self._used_device = max(0, self._used_device - lease.device_bytes)
            self._used_disk = max(0, self._used_disk - lease.disk_bytes)
            self._inflight = max(0, self._inflight - 1)

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
    state_bytes = int(program.total_state_bytes()) if program is not None else 0
    streaming = bool(getattr(parameter_store, "needs_prefetch", False))
    from tensortorrent.compile.fit import should_hoist_resident_parameters

    hoist = (not streaming) and should_hoist_resident_parameters(config, state_bytes=state_bytes)
    base = CapacityLease(
        host_bytes=0 if streaming else state_bytes,
        device_bytes=state_bytes if hoist else 0,
        disk_bytes=0,
    )
    per_request = estimate_request_capacity(
        program=program,
        plan=plan,
        config=config,
        parameter_store=parameter_store,
    )
    budgets = resolve_capacity_budgets(config, machine=machine)
    return CapacityLedger(budgets, per_request=per_request, base=base)
