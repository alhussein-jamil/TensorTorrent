"""Unit tests for heterogeneous resource graph independence."""

from __future__ import annotations

from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    LinkClass,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    ensure_host_staged_fallbacks,
)


def test_unequal_gpus_are_independent_resources() -> None:
    g = ResourceGraph(fingerprint="test")
    for i, (vendor, mem) in enumerate((("nvidia", 8 << 30), ("amd", 16 << 30))):
        cname = f"{vendor}_gpu_{i}"
        mname = f"{vendor}_vram_{i}"
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
                backend_id=vendor,
                model=f"{vendor}-{i}",
                vendor=vendor,
                supported_dtypes=("float16", "float32"),
                memory_affinity=(mname,),
            )
        )
    warnings = g.validate_independence()
    assert any("Unequal GPU memory" in w for w in warnings)
    assert len(g.gpus()) == 2
    assert g.gpus()[0].vendor != g.gpus()[1].vendor


def test_host_staged_fallback_when_no_p2p() -> None:
    g = ResourceGraph(fingerprint="test")
    for i in range(2):
        g.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, f"vram_{i}"),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=4 << 30,
                allocatable_bytes=4 << 30,
            )
        )
    g.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=16 << 30,
            allocatable_bytes=14 << 30,
        )
    )
    g = ensure_host_staged_fallbacks(g)
    link = g.links.get("vram_0=host=>vram_1")
    assert link is not None
    assert link.link_class == LinkClass.HOST_STAGED
    assert link.peer_to_peer is False
