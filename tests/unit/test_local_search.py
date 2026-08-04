"""Local search refinement tests."""

from __future__ import annotations

from tensortorrent.planner.local_search import rebalance_partitions, refine_prefetch_distance
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


def test_rebalance_never_mutates_backend_device_compatibility() -> None:
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
    # Device/backend/kernel changes require a fresh candidate evaluation. The
    # joint planner owns that operation; this compatibility shim only clones.
    assert [p.device for p in out.placements] == [p.device for p in plan.placements]
    assert [p.backend_id for p in out.placements] == [p.backend_id for p in plan.placements]
    assert [p.kernel_id for p in out.placements] == [p.kernel_id for p in plan.placements]
    # Byte metadata and measured flags must survive the compatibility pass.
    by_id = {p.region_id: p for p in out.placements}
    assert by_id["a"].output_bytes == 100
    assert by_id["a"].measured is True
    assert all(output is not original for output, original in zip(out.placements, plan.placements, strict=True))


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
