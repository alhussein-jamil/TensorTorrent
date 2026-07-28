"""Contention model tests."""

from __future__ import annotations

from streamcompiler.cost_model.contention import adjust_latency, concurrent_slowdown


def test_contention_increases_with_parallel_transfers() -> None:
    idle = concurrent_slowdown(active_compute=1, active_transfers=1, active_storage=0)
    busy = concurrent_slowdown(active_compute=2, active_transfers=3, active_storage=2)
    assert busy.transfer > idle.transfer
    assert busy.storage > idle.storage
    assert adjust_latency(1.0, busy.transfer) > 1.0
