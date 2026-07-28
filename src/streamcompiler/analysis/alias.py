"""Alias and storage analysis used by portable compilation."""

from __future__ import annotations

from dataclasses import dataclass, field

from streamcompiler.ir.graph import HeterogeneousGraph


@dataclass
class AliasAnalysis:
    groups: dict[str, str] = field(default_factory=dict)


def run_alias_analysis(graph: HeterogeneousGraph) -> AliasAnalysis:
    """Group tensors that share one underlying storage.

    Parameter/buffer tensors that load the same module attribute (shared weights,
    reused buffers) form one alias group keyed by that storage target. Distinct
    activations keep their own group. The analysis writes ``alias_group`` and
    ``storage_id`` back onto the IR so planners and stores can dedupe bytes.
    """
    bindings = graph.metadata.get("state_bindings") or {}
    if not isinstance(bindings, dict):
        bindings = {}

    groups: dict[str, str] = {}
    for tid, tensor in graph.tensors.items():
        storage = tensor.storage_id or bindings.get(tid)
        if storage:
            group = f"storage::{storage}"
            tensor.storage_id = str(storage)
        else:
            group = tensor.alias_group or tid
        tensor.alias_group = group
        groups[tid] = group
    return AliasAnalysis(groups=groups)
