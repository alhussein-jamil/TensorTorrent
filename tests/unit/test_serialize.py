"""Plan serialization tests."""

from __future__ import annotations

from pathlib import Path

from streamcompiler.planner.maximal import ExecutionPlan, Placement, ResourceDecision
from streamcompiler.planner.serialize import load_plan, write_plan


def test_plan_roundtrip(tmp_path: Path) -> None:
    plan = ExecutionPlan(
        graph_name="g",
        fingerprint="fp",
        objective="latency",
        placements=[Placement("r0", "cpu_numa_0", "cpu", "float32", "k", 0.1)],
        decisions=[ResourceDecision("cpu_numa_0", True, "selected")],
        devices_used=("cpu_numa_0",),
        communication_backend="gloo",
        predicted_latency_s=0.1,
        strategy="cpu_only",
        notes=["n"],
    )
    path = write_plan(plan, tmp_path / "plan.json")
    loaded = load_plan(path)
    assert loaded.graph_name == "g"
    assert loaded.placements[0].device == "cpu_numa_0"
    assert loaded.decisions[0].selected
