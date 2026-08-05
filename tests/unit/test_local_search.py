"""Local search refinement tests."""

from __future__ import annotations

from tensortorrent.planner.local_search import refine_prefetch_distance
from tensortorrent.planner.maximal import ExecutionPlan, Placement


def test_refine_prefetch_annotates_config_distance_without_faking_latency() -> None:
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="f",
        objective="latency",
        placements=[Placement("a", "gpu0", "cuda", "float16", "k", 1.0)],
        decisions=[],
        devices_used=("gpu0", "gpu1"),
        communication_backend="host_staged",
        predicted_latency_s=1.0,
    )
    refined = refine_prefetch_distance(plan, distance=3)
    assert "prefetch_distance=3" in refined.notes
    assert refined.predicted_latency_s == 1.0
    assert plan.notes == []
    assert refined.placements[0] is not plan.placements[0]


def test_adaptive_prefetch_uses_compute_and_storage_ratio() -> None:
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="f",
        objective="latency",
        placements=[
            Placement("a", "gpu0", "cuda", "float16", "k", 0.020, state_bytes=64 << 20),
            Placement("b", "gpu0", "cuda", "float16", "k", 0.020, state_bytes=64 << 20),
        ],
        decisions=[],
        devices_used=("gpu0",),
        communication_backend="host_staged",
        predicted_latency_s=0.040,
    )
    refined = refine_prefetch_distance(
        plan,
        distance=1,
        adaptive=True,
        storage_bytes_per_s=1 << 30,
        ram_budget_bytes=512 << 20,
    )
    assert refined.prefetch_distance >= 3
    assert any("prefetch_rationale=adaptive" in note for note in refined.notes)
