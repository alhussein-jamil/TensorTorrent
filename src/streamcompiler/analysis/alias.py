"""Alias and storage analysis placeholders used by portable compilation."""

from __future__ import annotations

from dataclasses import dataclass, field

from streamcompiler.ir.graph import HeterogeneousGraph


@dataclass
class AliasAnalysis:
    groups: dict[str, str] = field(default_factory=dict)


def run_alias_analysis(graph: HeterogeneousGraph) -> AliasAnalysis:
    groups = {tid: (t.alias_group or tid) for tid, t in graph.tensors.items()}
    return AliasAnalysis(groups=groups)
