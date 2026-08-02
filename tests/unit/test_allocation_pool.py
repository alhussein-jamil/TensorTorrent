"""ActivationAllocator must back buffer-reuse slots with one real allocation."""

from __future__ import annotations

import pytest
import torch

from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
from streamcompiler.ir.liveness import run_liveness_analysis
from streamcompiler.runtime.allocation_pool import ActivationAllocator
from streamcompiler.runtime.buffer_reuse import plan_buffer_reuse


def _reuse_plan_for_chain() -> tuple[list[str], dict[str, int]]:
    graph = HeterogeneousGraph(name="reuse", outputs=("c",))
    graph.add_tensor(TensorMeta("a", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_tensor(TensorMeta("tmp", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_tensor(TensorMeta("b", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_tensor(TensorMeta("c", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r0", inputs=(), outputs=("a",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r1", inputs=("a",), outputs=("tmp",)))
    graph.add_instruction(Instruction(OpCode.RELEASE, "rel", inputs=("a",), outputs=()))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r2", inputs=("tmp",), outputs=("b",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r3", inputs=("b",), outputs=("c",)))
    live = run_liveness_analysis(graph)
    plan = plan_buffer_reuse(graph, live)
    return ["a", "b"], plan.assignment


def test_non_overlapping_tensors_share_one_physical_allocation() -> None:
    pair, assignment = _reuse_plan_for_chain()
    a_id, b_id = pair
    assert assignment[a_id] == assignment[b_id], "planner must have picked one shared slot for this test"
    slot = assignment[a_id]

    allocator = ActivationAllocator()
    a_value = torch.arange(8, dtype=torch.float32)
    a_placed = allocator.acquire(slot, a_id, a_value)
    torch.testing.assert_close(a_placed, a_value)
    allocator.release(slot)

    b_value = torch.arange(8, 16, dtype=torch.float32)
    b_placed = allocator.acquire(slot, b_id, b_value)
    torch.testing.assert_close(b_placed, b_value)

    assert a_placed.data_ptr() == b_placed.data_ptr() == allocator.storage_ptr(slot), (
        "reused slot must be backed by the same physical buffer"
    )

    record = allocator.snapshot()[slot]
    assert record.reuse_history == [a_id, b_id]
    assert [e["event"] for e in record.events] == ["allocate", "release", "reuse"]
    assert record.current_tensor_id == b_id


def test_overlapping_tensors_get_distinct_physical_allocations() -> None:
    graph = HeterogeneousGraph(name="overlap", outputs=("c",))
    graph.add_tensor(TensorMeta("a", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_tensor(TensorMeta("b", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_tensor(TensorMeta("c", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r0", inputs=(), outputs=("a",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r1", inputs=("a",), outputs=("b",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r2", inputs=("a", "b"), outputs=("c",)))
    live = run_liveness_analysis(graph)
    plan = plan_buffer_reuse(graph, live)
    assert plan.assignment["a"] != plan.assignment["b"]

    allocator = ActivationAllocator()
    a_placed = allocator.acquire(plan.assignment["a"], "a", torch.ones(8))
    b_placed = allocator.acquire(plan.assignment["b"], "b", torch.zeros(8))
    assert a_placed.data_ptr() != b_placed.data_ptr()
    # Both remain independently readable: overlap would have corrupted one.
    torch.testing.assert_close(a_placed, torch.ones(8))
    torch.testing.assert_close(b_placed, torch.zeros(8))


def test_allocation_grows_to_fit_larger_reused_tensor() -> None:
    allocator = ActivationAllocator()
    allocator.acquire(0, "small", torch.zeros(4))
    small_capacity = allocator.snapshot()[0].capacity_bytes
    allocator.release(0)
    big = torch.ones(64)
    placed = allocator.acquire(0, "big", big)
    torch.testing.assert_close(placed, big)
    assert allocator.snapshot()[0].capacity_bytes >= big.numel() * big.element_size()
    assert allocator.snapshot()[0].capacity_bytes > small_capacity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_acquire_preserves_cuda_device() -> None:
    allocator = ActivationAllocator()
    like = torch.arange(8, dtype=torch.float32, device="cuda")
    placed = allocator.acquire(0, "y", like)
    assert placed.device.type == "cuda"
    torch.testing.assert_close(placed, like)
    allocator.release(0)
    # Reusing the slot for a host tensor must reallocate on host.
    host = torch.ones(8)
    host_placed = allocator.acquire(0, "z", host)
    assert host_placed.device.type == "cpu"
    torch.testing.assert_close(host_placed, host)
