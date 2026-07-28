"""Local search refinement tests."""

from __future__ import annotations

from streamcompiler.planner.local_search import rebalance_partitions, refine_prefetch_distance
from streamcompiler.planner.maximal import ExecutionPlan, Placement


def test_refine_prefetch_annotates_distance() -> None:
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
    refined = refine_prefetch_distance(plan)
    assert any(n.startswith("prefetch_distance=") for n in refined.notes)


def test_rebalance_moves_from_overloaded_device() -> None:
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="f",
        objective="latency",
        placements=[
            Placement("a", "gpu0", "cuda", "float16", "k", 1.0),
            Placement("b", "gpu0", "cuda", "float16", "k", 1.0),
            Placement("c", "gpu1", "cuda", "float16", "k", 0.1),
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
