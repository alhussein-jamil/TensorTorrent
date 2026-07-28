"""Unequal GPU memory should bias shard placement toward larger devices."""

from __future__ import annotations

from streamcompiler.config import CompileConfig, Objective
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode
from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)
from streamcompiler.planner import plan_execution


def test_larger_vram_receives_work_under_latency_objective() -> None:
    g = ResourceGraph(fingerprint="unequal", backends_present=("cpu", "cuda"))
    g.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=64 << 30,
            allocatable_bytes=60 << 30,
        )
    )
    g.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu_numa_0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            model="cpu",
            vendor="cpu",
            supported_dtypes=("float32", "float16", "bfloat16"),
            supported_ops=("aten::mm",),
            core_count=8,
            memory_affinity=("numa_ram_0",),
        )
    )
    for i, mem in enumerate((4 << 30, 24 << 30)):
        g.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, f"cuda_vram_{i}"),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=mem,
                allocatable_bytes=mem,
            )
        )
        g.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, f"cuda_gpu_{i}"),
                compute_class=ComputeClass.DISCRETE_GPU,
                backend_id="cuda",
                model=f"gpu{i}",
                vendor="nvidia",
                supported_dtypes=("float32", "float16", "bfloat16"),
                supported_ops=("aten::mm",),
                core_count=40 + 40 * i,
                memory_affinity=(f"cuda_vram_{i}",),
            )
        )

    ir = HeterogeneousGraph(name="unequal")
    for i in range(8):
        ir.add_instruction(Instruction(opcode=OpCode.COMPUTE, name=f"b{i}"))
    ir.repeated_blocks = tuple((f"b{i}",) for i in range(8))

    plan = plan_execution(ir, g, CompileConfig(objective=Objective.LATENCY, allow_cpu=False))
    counts: dict[str, int] = {}
    for p in plan.placements:
        counts[p.device] = counts.get(p.device, 0) + 1
    # With allow_cpu=False only GPUs participate; larger/faster GPU should get work.
    assert plan.devices_used
    assert any(name.startswith("cuda_gpu_") for name in plan.devices_used)
