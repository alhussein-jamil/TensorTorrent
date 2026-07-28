"""Runtime executor tests."""

from __future__ import annotations

from streamcompiler.compile.pipeline import SpecializedArtifact
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode
from streamcompiler.planner import plan_execution
from streamcompiler.runtime import Coordinator, TieredAllocator


def test_tiered_allocator_respects_capacity() -> None:
    machine = discover_resource_graph()
    alloc = TieredAllocator(machine)
    mem_name = next(iter(machine.memory))
    alloc.allocate(mem_name, 1024)
    assert alloc.used()[mem_name] == 1024
    alloc.release(mem_name, 1024)
    assert alloc.used()[mem_name] == 0


def test_coordinator_runs_cpu_plan() -> None:
    machine = discover_resource_graph()
    ir = HeterogeneousGraph(name="rt")
    for i in range(3):
        ir.add_instruction(Instruction(opcode=OpCode.COMPUTE, name=f"r{i}"))
    ir.repeated_blocks = tuple((f"r{i}",) for i in range(3))
    plan = plan_execution(ir, machine)
    artifact = SpecializedArtifact(fingerprint=machine.fingerprint, plan=plan)
    result = Coordinator(artifact, machine).execute()
    assert len(result["results"]) == 3
    assert result["telemetry"]
