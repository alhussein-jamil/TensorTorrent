"""Liveness analysis from producer–consumer dependencies.

Does not trust IR timestamps alone: recomputes intervals from instruction
def–use edges and validates any stored produced_at/last_use_at fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tensortorrent.ir.graph import HeterogeneousGraph

if TYPE_CHECKING:
    from tensortorrent.runtime.schedule import ExecutableSchedule


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


@dataclass
class ScheduleLivenessAnalysis:
    """Final asynchronous consumers and safe release dependencies per tensor."""

    producers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    consumers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    final_consumers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    release_dependencies: dict[str, tuple[str, ...]] = field(default_factory=dict)


def run_schedule_liveness(schedule: ExecutableSchedule) -> ScheduleLivenessAnalysis:
    """Derive liveness from the executable instruction DAG.

    A tensor stays live until every maximal consumer in the dependency partial
    order completes. This captures asynchronous transfers/events and overlapping
    compute streams rather than relying on portable graph indices.
    """
    from tensortorrent.ir.graph import OpCode
    from tensortorrent.runtime.schedule import ExecutableSchedule

    if not isinstance(schedule, ExecutableSchedule):
        raise TypeError(f"run_schedule_liveness expects ExecutableSchedule, got {type(schedule).__name__}")
    by_name = {inst.name: inst for inst in schedule.instructions}
    memo: dict[str, set[str]] = {}

    def ancestors(name: str) -> set[str]:
        cached = memo.get(name)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = list(by_name[name].depends_on)
        while stack:
            current = stack.pop()
            if current in seen or current not in by_name:
                continue
            seen.add(current)
            stack.extend(by_name[current].depends_on)
        memo[name] = seen
        return seen

    producers: dict[str, list[str]] = {}
    consumers: dict[str, list[str]] = {}
    for inst in schedule.instructions:
        if inst.opcode not in {OpCode.RECORD_EVENT, OpCode.WAIT_EVENT, OpCode.RELEASE}:
            for tensor in inst.outputs:
                producers.setdefault(tensor, []).append(inst.name)
        if inst.opcode != OpCode.RELEASE:
            for tensor in inst.inputs:
                consumers.setdefault(tensor, []).append(inst.name)

    final: dict[str, tuple[str, ...]] = {}
    for tensor, names in consumers.items():
        unique = tuple(dict.fromkeys(names))
        maximal = [name for name in unique if not any(name in ancestors(other) for other in unique if other != name)]
        final[tensor] = tuple(maximal)

    release_dependencies: dict[str, tuple[str, ...]] = {}
    for inst in schedule.instructions:
        if inst.opcode != OpCode.RELEASE:
            continue
        derived: list[str] = []
        for tensor in inst.inputs:
            derived.extend(final.get(tensor, ()))
        release_dependencies[inst.name] = tuple(dict.fromkeys(derived))

    return ScheduleLivenessAnalysis(
        producers={k: tuple(v) for k, v in producers.items()},
        consumers={k: tuple(v) for k, v in consumers.items()},
        final_consumers=final,
        release_dependencies=release_dependencies,
    )


def apply_schedule_liveness(schedule: ExecutableSchedule) -> ExecutableSchedule:
    """Return a schedule whose Release ops wait for final async consumers."""
    from dataclasses import replace

    from tensortorrent.ir.graph import OpCode
    from tensortorrent.runtime.schedule import ExecutableSchedule

    if not isinstance(schedule, ExecutableSchedule):
        raise TypeError(f"apply_schedule_liveness expects ExecutableSchedule, got {type(schedule).__name__}")
    analysis = run_schedule_liveness(schedule)
    updated = []
    for inst in schedule.instructions:
        if inst.opcode != OpCode.RELEASE:
            updated.append(inst)
            continue
        derived = analysis.release_dependencies.get(inst.name, ())
        # Preserve explicit transfer/event completion edges and add the derived
        # final-consumer frontier. Keeping redundant safety edges is intentional:
        # they make release correctness auditable and prevent backend-specific
        # asynchronous completion from being optimized away prematurely.
        deps = tuple(dict.fromkeys([*inst.depends_on, *derived]))
        attrs = {**dict(inst.attributes), "final_async_consumers": derived}
        updated.append(replace(inst, depends_on=deps, attributes=attrs))
    return replace(schedule, instructions=tuple(updated))
