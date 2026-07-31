"""Chrome trace export must survive lifetime/transfer timeline events."""

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
    TransferLink,
)
from streamcompiler.observability.trace import plan_to_chrome_trace
from streamcompiler.planner.maximal import ExecutionPlan, Placement
from streamcompiler.runtime.simulator import simulate_plan


def test_chrome_trace_includes_compute_transfer_and_release() -> None:
    machine = ResourceGraph(fingerprint="trace")
    for i in range(2):
        machine.add_memory(
            MemoryResource(
                id=ResourceId(ResourceKind.MEMORY, f"vram_{i}"),
                memory_class=MemoryClass.DEVICE_VRAM,
                capacity_bytes=8 << 30,
                allocatable_bytes=8 << 30,
            )
        )
        machine.add_compute(
            ComputeResource(
                id=ResourceId(ResourceKind.COMPUTE, f"gpu_{i}"),
                compute_class=ComputeClass.DISCRETE_GPU,
                backend_id="cuda",
                model=f"g{i}",
                vendor="nvidia",
                memory_affinity=(f"vram_{i}",),
            )
        )
    machine.add_link(
        TransferLink(
            id=ResourceId(ResourceKind.LINK, "vram_0->vram_1"),
            link_class=LinkClass.NVLINK,
            source="vram_0",
            destination="vram_1",
            peer_to_peer=True,
            measured=True,
            bytes_per_s=1e9,
            latency_s=1e-6,
        )
    )
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="trace",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 0.01, output_bytes=2_000_000),
            Placement("b", "gpu_1", "cuda", "float16", "k", 0.01, depends_on=("a",), output_bytes=0),
        ],
        decisions=[],
        devices_used=("gpu_0", "gpu_1"),
        communication_backend="nccl",
        predicted_latency_s=0.0,
        notes=["prefetch_distance=1"],
    )
    sim = simulate_plan(plan, machine)
    trace = plan_to_chrome_trace(plan, sim)
    assert trace["metadata"]["simulated"] is True
    cats = {e["cat"] for e in trace["traceEvents"]}
    assert "compute" in cats
    assert "transfer" in cats
    assert "memory" in cats
    names = {e["name"] for e in trace["traceEvents"]}
    assert any(n.startswith("landed:") for n in names)
    transfer_args = [e["args"] for e in trace["traceEvents"] if e.get("cat") == "transfer" and e.get("ph") == "X"]
    assert transfer_args and transfer_args[0].get("contention_factor", 0) >= 1.0
    release_args = [
        e["args"]
        for e in trace["traceEvents"]
        if e.get("cat") == "memory" and str(e.get("name", "")).startswith("release:")
    ]
    assert release_args
    assert any(
        a.get("kind") in {"transfer_copy", "Release", "activation", "Evict"} or a.get("nbytes") is not None
        for a in release_args
    )
