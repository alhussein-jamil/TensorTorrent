"""Static memory planning tests."""

from __future__ import annotations

from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.graph import HeterogeneousGraph, TensorMeta
from streamcompiler.planner.memory import plan_memory


def test_alias_groups_share_storage() -> None:
    machine = discover_resource_graph()
    ir = HeterogeneousGraph(name="mem")
    ir.add_tensor(
        TensorMeta(tensor_id="w", shape=(1024,), dtype="float32", size_bytes=4096, alias_group="g0", produced_at=0, last_use_at=5)
    )
    ir.add_tensor(
        TensorMeta(tensor_id="w_view", shape=(1024,), dtype="float32", size_bytes=4096, alias_group="g0", produced_at=0, last_use_at=5)
    )
    from streamcompiler.analysis.alias import run_alias_analysis
    from streamcompiler.analysis.liveness import run_liveness_analysis

    plan = plan_memory(ir, machine, alias=run_alias_analysis(ir), liveness=run_liveness_analysis(ir))
    assert len(plan.allocations) == 1


def test_nonoverlapping_lifetimes_can_reuse() -> None:
    machine = discover_resource_graph()
    ir = HeterogeneousGraph(name="mem2")
    ir.add_tensor(TensorMeta(tensor_id="a", shape=(256,), dtype="float32", size_bytes=1024, produced_at=0, last_use_at=1))
    ir.add_tensor(TensorMeta(tensor_id="b", shape=(256,), dtype="float32", size_bytes=1024, produced_at=2, last_use_at=3))
    from streamcompiler.analysis.alias import run_alias_analysis
    from streamcompiler.analysis.liveness import run_liveness_analysis

    plan = plan_memory(ir, machine, alias=run_alias_analysis(ir), liveness=run_liveness_analysis(ir))
    assert plan.reused_pairs or plan.peak_bytes
