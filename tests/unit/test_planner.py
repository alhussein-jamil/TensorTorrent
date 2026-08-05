"""Planner must search subsets and explain inclusions/exclusions."""

from __future__ import annotations

from tensortorrent.config import CompileConfig, Objective
from tensortorrent.ir.graph import HeterogeneousGraph, Instruction, OpCode
from tensortorrent.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)
from tensortorrent.planner.maximal import enumerate_plan_strategies, plan_execution


def _machine_with_unequal_gpus() -> ResourceGraph:
    g = ResourceGraph(fingerprint="fake-prod", backends_present=("cpu", "cuda", "rocm"))
    g.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=64 << 30,
            allocatable_bytes=60 << 30,
            numa_node=0,
        )
    )
    g.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu_numa_0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            model="cpu",
            vendor="cpu",
            supported_dtypes=("float32", "bfloat16", "float16"),
            supported_ops=("aten::mm",),
            core_count=16,
            concurrency_limit=16,
            numa_node=0,
            memory_affinity=("numa_ram_0",),
        )
    )
    for i, (backend, vendor, mem, cores) in enumerate(
        (
            ("cuda", "nvidia", 8 << 30, 40),
            ("cuda", "nvidia", 24 << 30, 80),
            ("rocm", "amd", 16 << 30, 60),
        )
    ):
        # Use cuda backend id only for devices the CpuBackend-compatible planner can query;
        # for this unit test we register compute metadata and rely on backend_by_id for cpu/cuda.
        # ROCm device uses rocm backend id; planner skips kernels if backend missing ops — still present.
        cname = f"{backend}_gpu_{i}"
        mname = f"{backend}_vram_{i}"
        g.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, mname),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=mem,
                allocatable_bytes=mem,
                attached_compute=(cname,),
            )
        )
        g.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, cname),
                compute_class=ComputeClass.DISCRETE_GPU,
                backend_id="cuda" if backend == "cuda" else "rocm",
                model=f"{vendor}-{i}",
                vendor=vendor,
                supported_dtypes=("float32", "float16", "bfloat16"),
                supported_ops=("aten::mm",),
                core_count=cores,
                memory_affinity=(mname,),
            )
        )
    return g


def test_planner_reports_resource_decisions() -> None:
    ir = HeterogeneousGraph(name="toy")
    for i in range(4):
        ir.add_instruction(Instruction(opcode=OpCode.COMPUTE, name=f"block_{i}"))
    ir.repeated_blocks = tuple((f"block_{i}",) for i in range(4))

    plan = plan_execution(ir, _machine_with_unequal_gpus(), CompileConfig(objective=Objective.LATENCY))
    assert plan.decisions
    assert any(d.selected for d in plan.decisions)
    text = plan.explain()
    assert "SELECTED" in text or "EXCLUDED" in text
    assert plan.devices_used


def test_strategy_catalog_covers_required_modes() -> None:
    strategies = enumerate_plan_strategies()
    assert "cpu_only" in strategies
    assert "multi_gpu" in strategies
    assert "pipeline_gpu_cpu" in strategies
    assert "single_gpu" in strategies
    assert set(strategies) == {
        "cpu_only",
        "single_gpu",
        "multi_gpu",
        "multi_numa_cpu",
        "pipeline_gpu_cpu",
    }


def test_region_byte_counts_come_from_tensor_metadata() -> None:
    from tensortorrent.ir.graph import TensorMeta
    from tensortorrent.planner.maximal import region_byte_counts

    ir = HeterogeneousGraph(name="bytes")
    ir.add_tensor(TensorMeta(tensor_id="w", shape=(10, 10), dtype="float32", size_bytes=400, kind="parameter"))
    ir.add_tensor(TensorMeta(tensor_id="x", shape=(2, 10), dtype="float32", size_bytes=80, kind="input"))
    ir.add_tensor(TensorMeta(tensor_id="y", shape=(2, 10), dtype="float32", size_bytes=80, kind="activation"))
    ir.add_instruction(
        Instruction(
            opcode=OpCode.COMPUTE,
            name="region_0",
            inputs=("x", "w"),
            outputs=("y",),
        )
    )
    assert region_byte_counts(ir) == {"region_0": (80, 400)}


def test_planned_placements_carry_real_byte_counts() -> None:
    import torch
    import torch.nn as nn

    import tensortorrent as tt

    model = nn.Linear(16, 8).eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(model, (x,))
    placements = compiled.specialized.plan.placements
    assert placements
    assert all(p.working_set_bytes > 0 for p in placements)
    # Linear(16, 8) weights are 16*8*4 + 8*4 = 544 bytes of state.
    assert sum(p.state_bytes for p in placements) >= 544
    text = compiled.explain()
    assert "(measured)" in text or "(prior)" in text
    assert compiled.specialized.profile["simulator"]["simulated"] is True
    compiled.close()
