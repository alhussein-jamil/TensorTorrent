"""Shared capacity ledger for concurrent inference admits."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import pytest

from tensortorrent.errors import TensorTorrentError
from tensortorrent.runtime.capacity import (
    CapacityBudgets,
    CapacityLease,
    CapacityLedger,
    capacity_preheld,
    capacity_preheld_scope,
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


def test_ledger_device_budget_fail_closed() -> None:
    ledger = CapacityLedger(
        CapacityBudgets(host_bytes=1 << 30, device_bytes=800, disk_bytes=0),
        per_request=CapacityLease(host_bytes=1, device_bytes=500),
    )
    assert ledger.try_acquire()
    assert not ledger.try_acquire()
    ledger.release()
    assert ledger.try_acquire()


def test_ledger_acquire_or_raise_message() -> None:
    ledger = CapacityLedger(
        CapacityBudgets(host_bytes=100, device_bytes=0, disk_bytes=0),
        per_request=CapacityLease(host_bytes=80),
    )
    ledger.acquire_or_raise(model_id="m0")
    with pytest.raises(TensorTorrentError, match="backpressure:.*capacity exhausted"):
        ledger.acquire_or_raise(model_id="m0")


def test_capacity_preheld_scope() -> None:
    assert capacity_preheld() is False
    with capacity_preheld_scope():
        assert capacity_preheld() is True
    assert capacity_preheld() is False


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
