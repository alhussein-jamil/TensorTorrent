"""Local search refinement tests."""

from __future__ import annotations

from streamcompiler.planner.local_search import rebalance_partitions, refine_prefetch_distance
from streamcompiler.planner.maximal import ExecutionPlan, Placement


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


def test_rebalance_moves_from_overloaded_device() -> None:
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="f",
        objective="latency",
        placements=[
            Placement("a", "gpu0", "cuda", "float16", "k", 1.0, output_bytes=100, state_bytes=50, measured=True),
            Placement("b", "gpu0", "cuda", "float16", "k", 1.0, output_bytes=80, state_bytes=40, measured=True),
            Placement("c", "gpu1", "cuda", "float16", "k", 0.1, output_bytes=10, state_bytes=5, measured=True),
        ],
        decisions=[],
        devices_used=("gpu0", "gpu1"),
        communication_backend="host_staged",
        predicted_latency_s=2.0,
    )
    out = rebalance_partitions(plan)
    counts = {}
    for p in out.placements:
        counts[p.device] = counts.get(p.device, 0) + 1
    assert counts.get("gpu0", 0) < 3
    # Byte metadata and measured flags must survive a rebalance.
    by_id = {p.region_id: p for p in out.placements}
    assert by_id["a"].output_bytes == 100
    assert by_id["a"].measured is True
