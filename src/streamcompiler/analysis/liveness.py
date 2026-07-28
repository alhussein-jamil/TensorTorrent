"""Liveness analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

from streamcompiler.ir.graph import HeterogeneousGraph


@dataclass
class LivenessAnalysis:
    intervals: dict[str, tuple[int | None, int | None]] = field(default_factory=dict)


def run_liveness_analysis(graph: HeterogeneousGraph) -> LivenessAnalysis:
    intervals = {tid: (t.produced_at, t.last_use_at) for tid, t in graph.tensors.items()}
    return LivenessAnalysis(intervals=intervals)
