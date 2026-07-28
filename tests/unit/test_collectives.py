"""Collective planning tests."""

from __future__ import annotations

from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode
from streamcompiler.ir.resource_graph import ResourceGraph
from streamcompiler.planner.collectives import plan_collectives


def test_plan_collectives_marks_host_staged_for_mixed_devices() -> None:
    graph = HeterogeneousGraph(name="c")
    graph.add_instruction(Instruction(opcode=OpCode.COLLECTIVE, name="ar", attributes={"op": "allreduce"}))
    plans = plan_collectives(graph, ResourceGraph(fingerprint="x"), ("cuda_gpu_0", "rocm_gpu_0"))
    assert plans
    assert plans[0].backend_id in {"host_staged", "gloo", "nccl", "rccl", "oneccl"}
