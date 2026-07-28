"""Buffer reuse from liveness must never overlap live activations."""

from __future__ import annotations

from streamcompiler.analysis.liveness import run_liveness_analysis
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta
from streamcompiler.runtime.buffer_reuse import assert_reuse_safe, plan_buffer_reuse


def test_non_overlapping_activations_reuse_one_slot() -> None:
    graph = HeterogeneousGraph(name="reuse", outputs=("c",))
    graph.add_tensor(TensorMeta("a", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_tensor(TensorMeta("b", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_tensor(TensorMeta("c", (8,), "float32", size_bytes=32, kind="activation"))
    # a live [0,1], released before b is produced at 2 → a and b may share.
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r0", inputs=(), outputs=("a",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r1", inputs=("a",), outputs=("tmp",)))
    graph.add_tensor(TensorMeta("tmp", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_instruction(Instruction(OpCode.RELEASE, "rel", inputs=("a",), outputs=()))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r2", inputs=("tmp",), outputs=("b",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r3", inputs=("b",), outputs=("c",)))
    live = run_liveness_analysis(graph)
    plan = plan_buffer_reuse(graph, live)
    assert_reuse_safe(plan, live)
    assert plan.assignment["a"] == plan.assignment["b"]
    assert plan.saved_bytes > 0


def test_overlapping_activations_get_distinct_slots() -> None:
    graph = HeterogeneousGraph(name="overlap", outputs=("c",))
    graph.add_tensor(TensorMeta("a", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_tensor(TensorMeta("b", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_tensor(TensorMeta("c", (8,), "float32", size_bytes=32, kind="activation"))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r0", inputs=(), outputs=("a",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r1", inputs=("a",), outputs=("b",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r2", inputs=("a", "b"), outputs=("c",)))
    live = run_liveness_analysis(graph)
    plan = plan_buffer_reuse(graph, live)
    assert_reuse_safe(plan, live)
    assert plan.assignment["a"] != plan.assignment["b"]
