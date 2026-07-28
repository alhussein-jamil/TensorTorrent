"""Property-based checks for resource graph invariants."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
    ensure_host_staged_fallbacks,
)


@given(st.integers(min_value=1, max_value=4), st.integers(min_value=1, max_value=4))
@settings(max_examples=20)
def test_host_staged_created_for_vram_pairs(n_gpu: int, _seed: int) -> None:
    g = ResourceGraph(fingerprint="hyp")
    g.add_memory(
        MemoryResource(
            id=ResourceId(ResourceKind.MEMORY, "numa_ram_0"),
            memory_class=MemoryClass.NUMA_RAM,
            capacity_bytes=1 << 30,
            allocatable_bytes=1 << 30,
        )
    )
    for i in range(n_gpu):
        g.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, f"vram_{i}"),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=(i + 1) << 30,
                allocatable_bytes=(i + 1) << 30,
            )
        )
        g.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, f"gpu_{i}"),
                compute_class=ComputeClass.DISCRETE_GPU,
                backend_id="cuda",
                model=f"g{i}",
                vendor="nvidia" if i % 2 == 0 else "amd",
                memory_affinity=(f"vram_{i}",),
            )
        )
    g = ensure_host_staged_fallbacks(g)
    if n_gpu >= 2:
        assert any("host" in name for name in g.links)
