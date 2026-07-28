"""Residency schedule prep for future multi-device execution."""

from __future__ import annotations

from streamcompiler.planner.maximal import ExecutionPlan, Placement
from streamcompiler.runtime.residency import build_residency_schedule


def test_same_device_plan_needs_no_transfers() -> None:
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="x",
        objective="latency",
        placements=[
            Placement("a", "cpu_numa_0", "cpu", "float32", "k", 0.1, output_bytes=100),
            Placement("b", "cpu_numa_0", "cpu", "float32", "k", 0.1, depends_on=("a",), state_bytes=50),
        ],
        decisions=[],
        devices_used=("cpu_numa_0",),
        communication_backend="none",
        predicted_latency_s=0.2,
    )
    schedule = build_residency_schedule(plan)
    assert schedule.transfers == ()
    assert "single_device_plan" in schedule.notes[0]


def test_cross_device_dependency_schedules_transfer() -> None:
    plan = ExecutionPlan(
        graph_name="t",
        fingerprint="x",
        objective="latency",
        placements=[
            Placement("a", "cpu_numa_0", "cpu", "float32", "k", 0.1, output_bytes=1024),
            Placement("b", "cuda_gpu_0", "cuda", "float32", "k", 0.1, depends_on=("a",)),
        ],
        decisions=[],
        devices_used=("cpu_numa_0", "cuda_gpu_0"),
        communication_backend="host_staged",
        predicted_latency_s=0.2,
    )
    schedule = build_residency_schedule(plan)
    assert len(schedule.transfers) == 1
    transfer = schedule.transfers[0]
    assert transfer.source_device == "cpu_numa_0"
    assert transfer.destination_device == "cuda_gpu_0"
    assert transfer.nbytes == 1024
    assert "unvalidated" in schedule.notes[0]
