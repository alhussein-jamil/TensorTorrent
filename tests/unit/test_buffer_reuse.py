"""Buffer reuse from liveness must never overlap live activations."""

from __future__ import annotations

from streamcompiler.analysis.alias import run_alias_analysis
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


def test_reuse_rejected_while_an_alias_remains_live() -> None:
    """A view sharing storage with an earlier-released tensor extends that
    tensor's true lifetime; an unrelated tensor must not be handed the same
    slot just because the *view's own* narrow interval starts later."""
    graph = HeterogeneousGraph(name="alias_reuse", outputs=("dummy3",))
    for name in ("a", "b", "v", "dummy1", "dummy2", "dummy3"):
        storage = "buf0" if name in ("a", "v") else None
        graph.add_tensor(TensorMeta(name, (4,), "float32", size_bytes=16, kind="activation", storage_id=storage))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r0", inputs=(), outputs=("a",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r1", inputs=("a",), outputs=("dummy1",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r2", inputs=(), outputs=("b",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r3", inputs=("b",), outputs=("dummy2",)))
    # "v" is a late view onto "a"'s storage: its own interval starts long after
    # "a"'s own last use, but the underlying buffer is still live until here.
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r4", inputs=(), outputs=("v",)))
    graph.add_instruction(Instruction(OpCode.COMPUTE, "r5", inputs=("v",), outputs=("dummy3",)))

    alias = run_alias_analysis(graph)
    assert alias.groups["a"] == alias.groups["v"]
    live = run_liveness_analysis(graph)

    # Naive per-tensor intervals would say "a" is dead after index 1, well
    # before "b" is even produced at index 2 -- a naive planner would reuse
    # a's slot for b, corrupting the storage "v" still needs at index 4-5.
    assert live.intervals["a"][1] < live.intervals["b"][0]

    plan = plan_buffer_reuse(graph, live, alias)
    assert_reuse_safe(plan, live, graph, alias)
    assert plan.assignment["a"] != plan.assignment["b"], "b must not reuse a's storage while alias v is still live"


def test_reuse_plan_never_overlaps_release_chains() -> None:
    """Straight-line activations with explicit release: reuse must stay conflict-free."""
    for n in (3, 5, 8):
        graph = HeterogeneousGraph(name=f"chain{n}", outputs=(f"t{n - 1}",))
        for i in range(n):
            graph.add_tensor(TensorMeta(f"t{i}", (4,), "float32", size_bytes=16, kind="activation"))
        graph.add_instruction(Instruction(OpCode.COMPUTE, "r0", inputs=(), outputs=("t0",)))
        for i in range(1, n):
            graph.add_instruction(Instruction(OpCode.COMPUTE, f"r{i}", inputs=(f"t{i - 1}",), outputs=(f"t{i}",)))
            graph.add_instruction(Instruction(OpCode.RELEASE, f"rel{i}", inputs=(f"t{i - 1}",), outputs=()))
        live = run_liveness_analysis(graph)
        plan = plan_buffer_reuse(graph, live)
        assert_reuse_safe(plan, live)
