"""Simulator tests for concurrent multi-device schedules."""

from __future__ import annotations

import pytest

from streamcompiler.ir.resource_graph import (
    ComputeClass,
    ComputeResource,
    MemoryClass,
    MemoryResource,
    ResourceGraph,
    ResourceId,
    ResourceKind,
)
from streamcompiler.planner.maximal import ExecutionPlan, Placement
from streamcompiler.simulator import simulate_plan


def test_multi_device_overlap_beats_serial() -> None:
    machine = ResourceGraph(fingerprint="sim")
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
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="sim",
        objective="latency",
        placements=[
            Placement("a", "gpu_0", "cuda", "float16", "k", 1.0, depends_on=()),
            Placement("b", "gpu_1", "cuda", "float16", "k", 1.0, depends_on=()),
        ],
        decisions=[],
        devices_used=("gpu_0", "gpu_1"),
        communication_backend="host_staged",
        predicted_latency_s=2.0,
        strategy="tensor_or_pipeline_multi_gpu",
    )
    result = simulate_plan(plan, machine)
    # Independent regions on two devices overlap → ~1s, not 2s.
    assert result.makespan_s == pytest.approx(1.0)
    assert result.makespan_s < 2.0
