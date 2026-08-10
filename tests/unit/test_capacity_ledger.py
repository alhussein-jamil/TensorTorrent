"""Shared capacity ledger for concurrent inference admits."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import pytest

from tensortorrent.errors import RuntimePlanError, TensorTorrentError
from tensortorrent.runtime.capacity import (
    CapacityBudgets,
    CapacityLease,
    CapacityLedger,
    estimate_request_capacity,
)


def test_ledger_admits_until_budget_exhausted() -> None:
    ledger = CapacityLedger(
        CapacityBudgets(host_bytes=1000, device_bytes=0, disk_bytes=0),
        per_request=CapacityLease(host_bytes=400),
    )
    assert ledger.max_concurrent() == 2
    assert ledger.try_acquire()
    assert ledger.try_acquire()
    assert not ledger.try_acquire()
    ledger.release()
    assert ledger.try_acquire()


def test_cpu_only_plan_does_not_reserve_device_base() -> None:
    """Fused-CPU / CPU DirectPlan must not charge fake VRAM for resident state."""
    from types import SimpleNamespace

    from tensortorrent.runtime.capacity import (
        build_module_capacity_ledger,
        resolve_capacity_budgets,
    )

    state = 50_000
    program = SimpleNamespace(total_state_bytes=lambda: state)
    plan = SimpleNamespace(
        devices_used=["cpu_numa_0"],
        predicted_peak_bytes={"host_ram": state + 1000, "activations": 500},
    )
    config = SimpleNamespace(
        ram_budget_bytes=None,
        vram_budget_bytes=8 << 30,
        vram_headroom_bytes=0,
        host_memory_reserve_bytes=None,
        max_total_spill_bytes=None,
        activation_budget_bytes=None,
        prefetch_distance=1,
        allow_training=False,
        allow_gpu=True,
    )
    store = SimpleNamespace(needs_prefetch=False)
    ledger = build_module_capacity_ledger(
        program=program,
        plan=plan,
        config=config,
        parameter_store=store,
        machine=None,
    )
    raw = resolve_capacity_budgets(config, machine=None)
    assert ledger.per_request.device_bytes == 0
    assert ledger.budgets.device_bytes == raw.device_bytes


def test_ledger_device_budget_fail_closed() -> None:
    ledger = CapacityLedger(
        CapacityBudgets(host_bytes=1 << 30, device_bytes=800, disk_bytes=0),
        per_request=CapacityLease(host_bytes=1, device_bytes=500),
    )
    assert ledger.try_acquire()
    assert not ledger.try_acquire()
    ledger.release()
    assert ledger.try_acquire()


def test_zero_device_budget_rejects_device_lease() -> None:
    ledger = CapacityLedger(
        CapacityBudgets(host_bytes=1 << 20, device_bytes=0, disk_bytes=0),
        per_request=CapacityLease(host_bytes=1, device_bytes=1),
    )
    assert ledger.max_concurrent() == 0
    assert not ledger.try_acquire()


def test_zero_disk_budget_rejects_disk_lease() -> None:
    ledger = CapacityLedger(
        CapacityBudgets(host_bytes=1 << 20, device_bytes=0, disk_bytes=0),
        per_request=CapacityLease(host_bytes=1, disk_bytes=1),
    )
    assert ledger.max_concurrent() == 0
    assert not ledger.try_acquire()


def test_ledger_release_requires_matching_acquire() -> None:
    ledger = CapacityLedger(
        CapacityBudgets(host_bytes=100, device_bytes=0, disk_bytes=0),
        per_request=CapacityLease(host_bytes=10),
    )
    with pytest.raises(RuntimePlanError, match="without matching acquisition"):
        ledger.release()
    assert ledger.try_acquire()
    ledger.release()
    with pytest.raises(RuntimePlanError, match="without matching acquisition"):
        ledger.release()


def test_ledger_acquire_or_raise_message() -> None:
    ledger = CapacityLedger(
        CapacityBudgets(host_bytes=100, device_bytes=0, disk_bytes=0),
        per_request=CapacityLease(host_bytes=80),
    )
    ledger.acquire_or_raise(model_id="m0")
    with pytest.raises(TensorTorrentError, match="backpressure:.*capacity exhausted"):
        ledger.acquire_or_raise(model_id="m0")


def test_estimate_request_excludes_shared_resident_params() -> None:
    program = SimpleNamespace(
        total_state_bytes=lambda: 10_000,
        max_region_state_bytes=lambda: 4_000,
    )
    plan = SimpleNamespace(predicted_peak_bytes={"host_ram": 12_000, "activations": 2_000, "vram_0": 11_000})
    config = SimpleNamespace(
        ram_budget_bytes=None,
        vram_budget_bytes=20_000,
        allow_nvme_streaming=True,
        allow_training=False,
        prefetch_distance=1,
        activation_budget_bytes=None,
    )
    lease = estimate_request_capacity(program=program, plan=plan, config=config, parameter_store=None)
    assert lease.host_bytes == 2_000
    assert lease.device_bytes == 2_000


def test_concurrent_ledger_never_overcommits() -> None:
    ledger = CapacityLedger(
        CapacityBudgets(host_bytes=8_000, device_bytes=0, disk_bytes=0),
        per_request=CapacityLease(host_bytes=1_000),
    )
    errors: list[str] = []
    acquired = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal acquired
        if ledger.try_acquire():
            with lock:
                acquired += 1
            # Hold briefly so peers contend.
            threading.Event().wait(0.01)
            ledger.release()

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(worker) for _ in range(64)]
        for fut in as_completed(futures):
            fut.result()
    assert acquired >= 8
    assert ledger.inflight == 0
    assert errors == []


def test_explicit_ram_budget_reserves_resident_host_base() -> None:
    """Absolute ram_budget_bytes still deducts resident state once as base."""
    from tensortorrent.runtime.capacity import build_module_capacity_ledger, resolve_capacity_budgets

    state = 50_000
    program = SimpleNamespace(total_state_bytes=lambda: state)
    plan = SimpleNamespace(
        devices_used=["cpu_numa_0"],
        predicted_peak_bytes={"host_ram": state + 1000, "activations": 500},
    )
    config = SimpleNamespace(
        ram_budget_bytes=8 << 30,
        vram_budget_bytes=0,
        vram_headroom_bytes=0,
        host_memory_reserve_bytes=0,
        max_total_spill_bytes=None,
        activation_budget_bytes=None,
        prefetch_distance=1,
        allow_training=False,
        allow_gpu=False,
    )
    store = SimpleNamespace(needs_prefetch=False)
    raw = resolve_capacity_budgets(config, machine=None)
    assert raw.host_source_kind == "explicit"
    assert raw.host_reflects_live_remaining is False
    ledger = build_module_capacity_ledger(
        program=program,
        plan=plan,
        config=config,
        parameter_store=store,
        machine=None,
    )
    assert ledger.budgets.host_bytes == raw.host_bytes - state


def test_live_available_host_budget_skips_resident_base() -> None:
    """os_available/cgroup ceilings already exclude resident model RAM."""
    from tensortorrent.runtime.capacity import build_module_capacity_ledger, resolve_capacity_budgets

    state = 50_000
    program = SimpleNamespace(total_state_bytes=lambda: state)
    plan = SimpleNamespace(
        devices_used=["cpu_numa_0"],
        predicted_peak_bytes={"host_ram": state + 1000, "activations": 500},
    )
    config = SimpleNamespace(
        ram_budget_bytes=None,
        vram_budget_bytes=0,
        vram_headroom_bytes=0,
        host_memory_reserve_bytes=None,
        max_total_spill_bytes=None,
        activation_budget_bytes=None,
        prefetch_distance=1,
        allow_training=False,
        allow_gpu=False,
    )
    store = SimpleNamespace(needs_prefetch=False)
    raw = resolve_capacity_budgets(config, machine=None)
    assert raw.host_reflects_live_remaining is True
    assert raw.host_source_kind in {"os_available", "cgroup_v2", "cgroup_v1"}
    ledger = build_module_capacity_ledger(
        program=program,
        plan=plan,
        config=config,
        parameter_store=store,
        machine=None,
    )
    assert ledger.budgets.host_bytes == raw.host_bytes
