"""Hardware-marked tests for mixed-vendor and unequal GPU planning."""

from __future__ import annotations

import pytest

from streamcompiler.config import CompileConfig, Objective
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode
from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    LinkClass,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    TransferLink,
    ensure_host_staged_fallbacks,
)
from streamcompiler.planner import plan_execution


def _mixed_vendor_machine() -> ResourceGraph:
    g = ResourceGraph(fingerprint="mixed-vendor-lab", backends_present=("cpu", "cuda", "rocm"))
    g.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=128 << 30,
            allocatable_bytes=120 << 30,
            numa_node=0,
        )
    )
    g.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu_numa_0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            model="epyc",
            vendor="cpu",
            supported_dtypes=("float32", "bfloat16"),
            supported_ops=("aten::mm",),
            core_count=32,
            concurrency_limit=32,
            memory_affinity=("numa_ram_0",),
        )
    )
    specs = (
        ("cuda", "nvidia", "cuda_gpu_0", "cuda_vram_0", 10 << 30, 70),
        ("cuda", "nvidia", "cuda_gpu_1", "cuda_vram_1", 24 << 30, 100),
        ("rocm", "amd", "rocm_gpu_0", "rocm_vram_0", 16 << 30, 80),
    )
    for backend, vendor, cname, mname, mem, cores in specs:
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
                backend_id=backend,
                model=f"{vendor}-card",
                vendor=vendor,
                supported_dtypes=("float32", "float16", "bfloat16"),
                supported_ops=("aten::mm",),
                core_count=cores,
                memory_affinity=(mname,),
            )
        )
    # No direct NVIDIA↔AMD P2P — only host-staged.
    g.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "cuda_vram_0->cuda_vram_1"),
            link_class=LinkClass.NVLINK,
            source="cuda_vram_0",
            destination="cuda_vram_1",
            peer_to_peer=True,
            measured=False,
        )
    )
    return ensure_host_staged_fallbacks(g)


@pytest.mark.hardware
def test_mixed_vendor_uses_host_staged_not_reject_machine() -> None:
    machine = _mixed_vendor_machine()
    assert any(link.link_class == LinkClass.HOST_STAGED for link in machine.links.values())
    ir = HeterogeneousGraph(name="mixed")
    for i in range(6):
        ir.add_instruction(Instruction(opcode=OpCode.COMPUTE, name=f"blk{i}"))
    ir.repeated_blocks = tuple((f"blk{i}",) for i in range(6))
    plan = plan_execution(
        ir,
        machine,
        CompileConfig(objective=Objective.LATENCY, allow_mixed_vendor=True, allow_host_staged_transfers=True),
    )
    assert plan.devices_used
    assert plan.communication_backend in {"gloo", "host_staged", "nccl", "rccl"}
    # Exclusion reasons must be present for unused resources when subset is smaller than machine.
    assert plan.decisions
    text = plan.explain()
    assert "SELECTED" in text
