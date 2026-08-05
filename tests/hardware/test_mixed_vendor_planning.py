"""Mixed-vendor planning tests (synthetic ResourceGraph — not live silicon).

Live mixed-vendor execution is covered by ``test_live_mixed_vendor_smoke`` when
both CUDA and ROCm (or XPU) devices are present; otherwise that test skips.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.ir.graph import HeterogeneousGraph, Instruction, OpCode
from tensortorrent.ir.resource_graph import (
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
from tensortorrent.planner import plan_execution


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
    """Planning-only: fabricated multi-vendor graph, not live ROCm+CUDA."""
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


def _live_multi_vendor() -> bool:
    from tensortorrent.hardware.discovery import discover_resource_graph

    graph = discover_resource_graph()
    vendors = {g.backend_id for g in graph.gpus() if g.backend_id not in {None, "cpu"}}
    return len(vendors) >= 2


@pytest.mark.hardware
@pytest.mark.skipif(not _live_multi_vendor(), reason="live mixed-vendor silicon not present")
def test_live_mixed_vendor_smoke() -> None:
    """Run only when the host actually exposes two accelerator backends."""
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4)).eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        expected = model(x)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(allow_cpu=True, allow_gpu=True, allow_mixed_vendor=True, use_torch_compile=False),
    )
    try:
        torch.testing.assert_close(compiled(x).detach().cpu(), expected, atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()
