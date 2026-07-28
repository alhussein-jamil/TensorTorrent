"""Weight streaming schedule tests."""

from __future__ import annotations

from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode
from streamcompiler.planner.streaming import synthesize_weight_stream


def test_synthesize_weight_stream_emits_prefetch_evict() -> None:
    machine = discover_resource_graph()
    ir = HeterogeneousGraph(name="stream")
    for i in range(6):
        ir.add_instruction(Instruction(opcode=OpCode.COMPUTE, name=f"l{i}"))
    ir.repeated_blocks = tuple((f"l{i}",) for i in range(6))
    device = next(n for n in machine.compute if n.startswith("cpu_numa_"))
    stages = synthesize_weight_stream(ir, machine, compute_device=device)
    assert stages
    assert any(s.action == "compute" for s in stages)
    assert any(i.opcode == OpCode.PREFETCH for i in ir.instructions)
    assert any(i.opcode == OpCode.EVICT for i in ir.instructions)
