"""Regression tests for the final CPU-side architecture foundations."""

from __future__ import annotations

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.analysis.liveness import run_schedule_liveness
from streamcompiler.backends.mock_accel import make_mock_accel_graph
from streamcompiler.config import CompileConfig
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.graph import OpCode
from streamcompiler.ir.resource_graph import merge_graphs
from streamcompiler.runtime.copies import CopyStore
from streamcompiler.runtime.schedule import (
    ExecutableSchedule,
    PlanInstruction,
    validate_schedule_tensor_sizes,
)
from streamcompiler.runtime.virtual_tensor import VirtualDeviceTensor


def test_views_share_backing_allocation_even_with_different_offsets() -> None:
    """Passive CopyStore counts shared storage once via allocation identity."""
    store = CopyStore()
    base = torch.arange(64, dtype=torch.float32)
    left = base[:16]
    right = base[16:32]
    assert left.storage_offset() != right.storage_offset()
    store.put("left", "cpu", left, ownership="activation")
    store.put("right", "host", right, ownership="activation")
    backing_bytes = int(base.untyped_storage().nbytes())
    assert store.activation_live_bytes() == backing_bytes
    assert store.live_bytes() == backing_bytes
    assert store.drop("left", "cpu") == int(left.numel() * left.element_size())
    assert store.live_bytes() == backing_bytes  # shared storage still referenced
    assert store.drop("right", "host") == int(right.numel() * right.element_size())
    assert store.live_bytes() == 0


def test_distinct_resource_allocations_count_separately() -> None:
    store = CopyStore()
    host = torch.ones(32)
    store.put("x", "cpu", host, ownership="activation")
    virtual = VirtualDeviceTensor(
        payload=host.clone().detach(),
        device_id="mock_accel_0",
        nbytes=int(host.clone().numel() * host.element_size()),
        allocation_key="",
        simulated=True,
    )
    store.replicate(
        "x",
        "mock_accel_0",
        virtual,
        source_resource="cpu",
        ownership="activation",
    )
    expected = int(host.untyped_storage().nbytes()) + int(virtual.nbytes)
    assert store.activation_live_bytes() == expected
    # Distinct resources / allocation keys remain independent in the value bag.
    assert store.has("x", "cpu", valid_only=True)
    assert store.has("x", "mock_accel_0", valid_only=True)


def test_specialization_profiles_explicit_virtual_resources() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(2, 8)
    machine = merge_graphs(discover_resource_graph(), make_mock_accel_graph())
    compiled = sc.compile(
        model,
        (x,),
        machine=machine,
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=True,
            region_measure_iters=1,
        ),
    )
    try:
        profile = compiled.specialized.profile["region_measurements"]
        assert profile
        assert all("mock_accel_0" in by_device for by_device in profile.values())
        assert all(by_device["mock_accel_0"]["simulated"] for by_device in profile.values())
        torch.testing.assert_close(compiled(x), model(x))
    finally:
        compiled.close()


def test_schedule_size_validation_rejects_ambiguous_multi_tensor_bytes() -> None:
    schedule = ExecutableSchedule(
        graph_name="bad_sizes",
        fingerprint="test",
        instructions=(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="compute",
                resource="cpu",
                inputs=("a", "b"),
                outputs=("c", "d"),
                nbytes=128,
            ),
        ),
    )
    errors = validate_schedule_tensor_sizes(schedule)
    assert errors
    assert any("tensor_nbytes" in error for error in errors)


def test_schedule_liveness_uses_final_async_consumer_frontier() -> None:
    schedule = ExecutableSchedule(
        graph_name="async_liveness",
        fingerprint="test",
        instructions=(
            PlanInstruction(OpCode.COMPUTE, "produce", "cpu", outputs=("x",), nbytes=64),
            PlanInstruction(
                OpCode.TRANSFER,
                "copy",
                "copy_engine:cpu->mock",
                depends_on=("produce",),
                inputs=("x",),
                outputs=("x",),
                nbytes=64,
                source="cpu",
                destination="mock_accel_0",
            ),
            PlanInstruction(
                OpCode.RECORD_EVENT,
                "record",
                "cpu",
                depends_on=("copy",),
                inputs=("x",),
            ),
            PlanInstruction(
                OpCode.WAIT_EVENT,
                "wait",
                "mock_accel_0",
                depends_on=("record",),
                inputs=("x",),
                attributes={"waits_for": "record"},
            ),
            PlanInstruction(
                OpCode.COMPUTE,
                "consume",
                "mock_accel_0",
                depends_on=("wait",),
                inputs=("x",),
                outputs=("y",),
                nbytes=64,
            ),
            PlanInstruction(
                OpCode.RELEASE,
                "release",
                "cpu",
                depends_on=("consume",),
                inputs=("x",),
                nbytes=64,
            ),
        ),
    )
    analysis = run_schedule_liveness(schedule)
    assert analysis.final_consumers["x"] == ("consume",)
    assert analysis.release_dependencies["release"] == ("consume",)
