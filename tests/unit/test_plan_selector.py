"""PlanSelector tests."""

from __future__ import annotations

import pytest

from streamcompiler.planner.maximal import ExecutionPlan
from streamcompiler.planner.plan_family import PlanFamily
from streamcompiler.runtime.plan_selector import PlanSelector, RuntimeContext


def test_selector_falls_back_when_vram_insufficient() -> None:
    tight = ExecutionPlan(
        graph_name="t",
        fingerprint="f",
        objective="latency",
        placements=[],
        decisions=[],
        devices_used=("cuda_gpu_0",),
        communication_backend="nccl",
        predicted_latency_s=1.0,
        predicted_peak_bytes={"cuda_vram_0": 8 << 30},
    )
    loose = ExecutionPlan(
        graph_name="t",
        fingerprint="f",
        objective="latency",
        placements=[],
        decisions=[],
        devices_used=("cpu_numa_0",),
        communication_backend="gloo",
        predicted_latency_s=2.0,
        predicted_peak_bytes={"numa_ram_0": 1 << 20},
    )
    family = PlanFamily(
        fingerprint="f",
        plans={"decode_b1_s512": tight, "fallback": loose},
        fallback="fallback",
    )
    selector = PlanSelector(family)
    chosen = selector.select(RuntimeContext(batch=1, seq=128, free_vram_bytes={"cuda_vram_0": 1 << 20}))
    assert chosen is loose


def test_selector_errors_without_viable_fallback() -> None:
    tight = ExecutionPlan(
        graph_name="t",
        fingerprint="f",
        objective="latency",
        placements=[],
        decisions=[],
        devices_used=("cuda_gpu_0",),
        communication_backend="nccl",
        predicted_latency_s=1.0,
        predicted_peak_bytes={"cuda_vram_0": 8 << 30},
    )
    family = PlanFamily(fingerprint="f", plans={"decode_b1_s512": tight})
    with pytest.raises(MemoryError):
        PlanSelector(family).select(RuntimeContext(batch=1, seq=128, free_vram_bytes={"cuda_vram_0": 1 << 20}))
