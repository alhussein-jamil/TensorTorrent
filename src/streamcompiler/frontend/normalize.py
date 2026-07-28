"""Graph normalization helpers after export."""

from __future__ import annotations

from streamcompiler.ir.graph import HeterogeneousGraph, OpCode


def normalize_graph(graph: HeterogeneousGraph) -> HeterogeneousGraph:
    """Normalize naming and drop trivial no-op placeholders."""
    kept = []
    for inst in graph.instructions:
        if inst.opcode == OpCode.COMPUTE and inst.name.endswith("_noop"):
            continue
        kept.append(inst)
    graph.instructions = kept
    # Ensure every compute output has tensor metadata.
    for inst in graph.instructions:
        for out in inst.outputs:
            if out not in graph.tensors:
                from streamcompiler.ir.graph import TensorMeta

                graph.add_tensor(TensorMeta(tensor_id=out, shape=(), dtype="float32"))
    return graph
