"""Demonstrate planning on a synthetic mixed-vendor production topology.

This does not require real GPUs on the development host. It shows how the
planner treats unequal devices and host-staged links.
"""

from __future__ import annotations

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


def build_synthetic_machine() -> ResourceGraph:
    g = ResourceGraph(fingerprint="synthetic-mixed", backends_present=("cpu", "cuda", "rocm"))
    g.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=256 << 30,
            allocatable_bytes=240 << 30,
            numa_node=0,
        )
    )
    g.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_1"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=256 << 30,
            allocatable_bytes=240 << 30,
            numa_node=1,
        )
    )
    for node in (0, 1):
        g.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, f"cpu_numa_{node}"),
                compute_class=ComputeClass.CPU_NUMA_POOL,
                backend_id="cpu",
                model="epyc",
                vendor="cpu",
                supported_dtypes=("float32", "bfloat16", "float16"),
                supported_ops=("aten::mm", "aten::embedding"),
                core_count=32,
                concurrency_limit=32,
                numa_node=node,
                memory_affinity=(f"numa_ram_{node}",),
            )
        )
    specs = (
        ("cuda", "nvidia", 0, 10 << 30, 70),
        ("cuda", "nvidia", 1, 24 << 30, 108),
        ("rocm", "amd", 0, 16 << 30, 80),
    )
    for backend, vendor, idx, mem, cores in specs:
        cname = f"{backend}_gpu_{idx}"
        mname = f"{backend}_vram_{idx}"
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
                model=f"{vendor}-{idx}",
                vendor=vendor,
                supported_dtypes=("float32", "float16", "bfloat16"),
                supported_ops=("aten::mm",),
                core_count=cores,
                memory_affinity=(mname,),
            )
        )
    g.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "cuda_vram_0->cuda_vram_1"),
            link_class=LinkClass.NVLINK,
            source="cuda_vram_0",
            destination="cuda_vram_1",
            peer_to_peer=True,
        )
    )
    return ensure_host_staged_fallbacks(g)


def main() -> None:
    machine = build_synthetic_machine()
    ir = HeterogeneousGraph(name="synthetic_transformer")
    for i in range(12):
        ir.add_instruction(Instruction(opcode=OpCode.COMPUTE, name=f"layer_{i}"))
    ir.repeated_blocks = tuple((f"layer_{i}",) for i in range(12))
    plan = plan_execution(
        ir,
        machine,
        CompileConfig(objective=Objective.LATENCY, allow_mixed_vendor=True),
    )
    print(plan.explain())
    staged = [n for n, link in machine.links.items() if link.link_class == LinkClass.HOST_STAGED]
    print(f"\nhost_staged_links={len(staged)}")
    print("sample:", staged[:3])


if __name__ == "__main__":
    main()
