"""Alias and storage analysis used by portable compilation."""

from __future__ import annotations

from dataclasses import dataclass, field

from tensortorrent.errors import UnsupportedFeatureError
from tensortorrent.ir.graph import HeterogeneousGraph


@dataclass
class AliasAnalysis:
    groups: dict[str, str] = field(default_factory=dict)
    view_of: dict[str, str] = field(default_factory=dict)
    """tensor_id -> storage root when the tensor is a view onto another."""
    mutable_groups: set[str] = field(default_factory=set)
    """Alias groups that contain at least one mutable tensor."""


def run_alias_analysis(graph: HeterogeneousGraph) -> AliasAnalysis:
    """Group tensors that share one underlying storage.

    Detects:
    - Shared parameters / tied weights (same ``state_bindings`` target)
    - Explicit ``storage_id`` / ``alias_group`` already on the IR
    - View relationships recorded in tensor attributes (``view_of``)
    - Mutations: mutable tensors poison their alias group

    Unsupported in-place mutation of a live aliased activation raises.
    """
    bindings = graph.metadata.get("state_bindings") or {}
    if not isinstance(bindings, dict):
        bindings = {}

    groups: dict[str, str] = {}
    view_of: dict[str, str] = {}
    mutable_groups: set[str] = set()

    for tid, tensor in graph.tensors.items():
        storage = tensor.storage_id or bindings.get(tid)
        view_src = tensor.attributes.get("view_of")
        if view_src:
            view_of[tid] = str(view_src)
            root = groups.get(str(view_src)) or str(view_src)
            storage = graph.tensors[str(view_src)].storage_id if str(view_src) in graph.tensors else storage
            group = f"storage::{storage}" if storage else f"view::{root}"
        elif storage:
            group = f"storage::{storage}"
            tensor.storage_id = str(storage)
        else:
            group = tensor.alias_group or tid
        tensor.alias_group = group
        if storage and not tensor.storage_id:
            tensor.storage_id = str(storage)
        groups[tid] = group
        if tensor.mutable:
            mutable_groups.add(group)

    # Reject mutations that would invalidate a still-live alias without an explicit plan.
    for tid, tensor in graph.tensors.items():
        if not tensor.mutable:
            continue
        group = groups[tid]
        siblings = [other for other, g in groups.items() if g == group and other != tid]
        if not siblings:
            continue
        # If any sibling is still live past this tensor's produced_at, mutation is unsafe
        # unless every sibling is also marked mutable and the same storage.
        for sibling in siblings:
            other = graph.tensors[sibling]
            if other.kind == "parameter":
                # Tied weights: mutation of one view of a parameter is unsupported.
                raise UnsupportedFeatureError(
                    f"Mutable tensor {tid} aliases parameter/storage group {group} "
                    f"(also used by {sibling}); in-place mutation of shared weights is unsupported"
                )

    return AliasAnalysis(groups=groups, view_of=view_of, mutable_groups=mutable_groups)


def storage_bytes_by_group(graph: HeterogeneousGraph, groups: dict[str, str]) -> dict[str, int]:
    """Deduplicate size_bytes per alias group (shared weights counted once)."""
    best: dict[str, int] = {}
    for tid, group in groups.items():
        tensor = graph.tensors.get(tid)
        if tensor is None:
            continue
        best[group] = max(best.get(group, 0), max(0, tensor.size_bytes))
    return best
