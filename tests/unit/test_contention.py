"""Contention model tests."""

from __future__ import annotations

from tensortorrent.planner.cost.transfer import concurrent_slowdown


def test_concurrent_slowdown_grows_with_pressure() -> None:
    idle = concurrent_slowdown(active_compute=1, active_transfers=0, active_storage=0)
    busy = concurrent_slowdown(active_compute=4, active_transfers=2, active_storage=1)
    assert busy.compute > idle.compute
    assert busy.transfer > idle.transfer
    assert busy.storage > idle.storage
