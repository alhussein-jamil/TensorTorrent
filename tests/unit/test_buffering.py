"""Buffering and exposed-transfer helpers."""

from __future__ import annotations

from streamcompiler.planner.buffering import (
    choose_buffering,
    exposed_transfer_latency,
)


def test_triple_buffering_when_all_tiers_present() -> None:
    plan = choose_buffering(has_copy_engine=True, has_cpu_prepare=True, has_nvme=True)
    assert plan.depth == 3
    assert "NVMe" in plan.describe()


def test_exposed_transfer_latency_formula() -> None:
    assert exposed_transfer_latency(10.0, 4.0) == 6.0
    assert exposed_transfer_latency(2.0, 5.0) == 0.0
