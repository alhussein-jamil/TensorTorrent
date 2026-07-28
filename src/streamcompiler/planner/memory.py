"""Memory planning across tiers with alias/liveness constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

from streamcompiler.analysis.alias import AliasAnalysis
from streamcompiler.analysis.liveness import LivenessAnalysis
from streamcompiler.ir.graph import HeterogeneousGraph
from streamcompiler.ir.resource_graph import ResourceGraph


@dataclass
class Allocation:
    tensor_id: str
    memory: str
    offset: int
    size: int


@dataclass
class MemoryPlan:
    allocations: list[Allocation] = field(default_factory=list)
    peak_bytes: dict[str, int] = field(default_factory=dict)
    reused_pairs: list[tuple[str, str]] = field(default_factory=list)


def plan_memory(
    graph: HeterogeneousGraph,
    machine: ResourceGraph,
    *,
    alias: AliasAnalysis | None = None,
    liveness: LivenessAnalysis | None = None,
    preferred_memory: str | None = None,
) -> MemoryPlan:
    """Greedy first-fit decreasing with non-overlapping lifetime reuse."""
    alias = alias or AliasAnalysis({tid: tid for tid in graph.tensors})
    liveness = liveness or LivenessAnalysis({tid: (t.produced_at, t.last_use_at) for tid, t in graph.tensors.items()})
    mem_name = preferred_memory or next(iter(machine.memory), "numa_ram_0")
    plan = MemoryPlan()
    # Group by alias to avoid duplicating tied storage.
    representatives: dict[str, str] = {}
    for tid, group in alias.groups.items():
        representatives.setdefault(group, tid)

    free_intervals: list[tuple[int, int, int]] = []  # start, end, size of free holes (logical)
    offset = 0
    ordered = sorted(
        representatives.values(),
        key=lambda tid: -(graph.tensors[tid].size_bytes or 1),
    )
    live_allocs: list[tuple[str, int, int, int, int | None, int | None]] = []
    # tensor, offset, size, end_offset, produced, last_use

    for tid in ordered:
        meta = graph.tensors[tid]
        size = max(64, meta.size_bytes or 64)
        produced, last_use = liveness.intervals.get(tid, (None, None))
        reused = False
        for i, (other_tid, offset, other_size, _, _op, ol) in enumerate(list(live_allocs)):
            # Reuse if lifetimes do not overlap.
            if produced is not None and ol is not None and produced > ol and other_size >= size:
                plan.allocations.append(Allocation(tid, mem_name, offset, size))
                plan.reused_pairs.append((other_tid, tid))
                live_allocs[i] = (tid, offset, size, offset + size, produced, last_use)
                reused = True
                break
        if not reused:
            plan.allocations.append(Allocation(tid, mem_name, offset, size))
            live_allocs.append((tid, offset, size, offset + size, produced, last_use))
            offset += size
        plan.peak_bytes[mem_name] = max(plan.peak_bytes.get(mem_name, 0), offset)
        _ = free_intervals
    return plan
