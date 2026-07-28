"""Liveness analysis from producer–consumer dependencies.

Does not trust IR timestamps alone: recomputes intervals from instruction
def–use edges and validates any stored produced_at/last_use_at fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from streamcompiler.ir.graph import HeterogeneousGraph, OpCode


@dataclass
class LivenessAnalysis:
    intervals: dict[str, tuple[int | None, int | None]] = field(default_factory=dict)
    """tensor_id -> (first_def_index, last_use_index)."""
    reuse_groups: list[tuple[str, str]] = field(default_factory=list)
    """Pairs of tensors whose live ranges do not overlap (safe buffer reuse)."""
    mismatches: list[str] = field(default_factory=list)
    """Stored IR timestamps that disagreed with derived intervals."""


def run_liveness_analysis(graph: HeterogeneousGraph) -> LivenessAnalysis:
    """Derive live intervals from instruction order and use edges."""
    produced: dict[str, int] = {}
    last_use: dict[str, int] = {}

    for index, inst in enumerate(graph.instructions):
        for tid in inst.outputs:
            produced.setdefault(tid, index)
            last_use[tid] = max(last_use.get(tid, index), index)
        for tid in inst.inputs:
            last_use[tid] = max(last_use.get(tid, -1), index)
            if tid not in produced and tid in graph.tensors:
                # Parameters / inputs may be live from the start.
                produced.setdefault(tid, 0)

    # Graph outputs stay live through the end of the program.
    end = max(0, len(graph.instructions) - 1)
    for tid in graph.outputs:
        last_use[tid] = max(last_use.get(tid, end), end)
        produced.setdefault(tid, 0)
    for tid in graph.parameters:
        produced.setdefault(tid, 0)
        last_use.setdefault(tid, end)

    intervals: dict[str, tuple[int | None, int | None]] = {}
    mismatches: list[str] = []
    for tid, tensor in graph.tensors.items():
        start = produced.get(tid)
        finish = last_use.get(tid)
        intervals[tid] = (start, finish)
        if tensor.produced_at is not None and start is not None and tensor.produced_at != start:
            mismatches.append(f"{tid}: produced_at {tensor.produced_at} != derived {start}")
        if tensor.last_use_at is not None and finish is not None and tensor.last_use_at != finish:
            mismatches.append(f"{tid}: last_use_at {tensor.last_use_at} != derived {finish}")
        # Write back validated intervals so downstream consumers share one source of truth.
        tensor.produced_at = start
        tensor.last_use_at = finish

    reuse_groups = _non_overlapping_pairs(intervals)
    return LivenessAnalysis(intervals=intervals, reuse_groups=reuse_groups, mismatches=mismatches)


def live_at(intervals: dict[str, tuple[int | None, int | None]], index: int) -> set[str]:
    """Tensors whose live range covers instruction index ``index``."""
    live: set[str] = set()
    for tid, (start, finish) in intervals.items():
        if start is None or finish is None:
            continue
        if start <= index <= finish:
            live.add(tid)
    return live


def ranges_overlap(a: tuple[int | None, int | None], b: tuple[int | None, int | None]) -> bool:
    if a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return True
    return not (a[1] < b[0] or b[1] < a[0])


def _non_overlapping_pairs(
    intervals: dict[str, tuple[int | None, int | None]],
) -> list[tuple[str, str]]:
    ids = sorted(intervals)
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if not ranges_overlap(intervals[left], intervals[right]):
                pairs.append((left, right))
    return pairs


def peak_live_bytes(graph: HeterogeneousGraph, intervals: dict[str, tuple[int | None, int | None]]) -> int:
    """Peak bytes of simultaneously live tensors (activations + params present)."""
    if not graph.instructions:
        return sum(t.size_bytes for t in graph.tensors.values())
    peak = 0
    for index in range(len(graph.instructions)):
        total = 0
        for tid in live_at(intervals, index):
            tensor = graph.tensors.get(tid)
            if tensor is not None:
                total += max(0, tensor.size_bytes)
        peak = max(peak, total)
    return peak


def compute_ops_only(graph: HeterogeneousGraph) -> list[int]:
    return [i for i, inst in enumerate(graph.instructions) if inst.opcode == OpCode.COMPUTE]
