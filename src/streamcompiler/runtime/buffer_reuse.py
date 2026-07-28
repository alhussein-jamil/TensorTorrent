"""Liveness-driven activation buffer reuse.

Non-overlapping live ranges may share one host buffer slot. Overlapping ranges
must not. This is CPU-side reuse prep for future VRAM allocators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from streamcompiler.analysis.liveness import LivenessAnalysis, ranges_overlap
from streamcompiler.errors import RuntimePlanError
from streamcompiler.ir.graph import HeterogeneousGraph


@dataclass
class BufferSlot:
    slot_id: int
    capacity_bytes: int
    assigned: list[str] = field(default_factory=list)


@dataclass
class BufferReusePlan:
    """Maps tensor ids to reusable slot ids."""

    assignment: dict[str, int] = field(default_factory=dict)
    slots: list[BufferSlot] = field(default_factory=list)
    saved_bytes: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "assignment": dict(self.assignment),
            "slot_count": len(self.slots),
            "saved_bytes": self.saved_bytes,
            "notes": list(self.notes),
        }


def plan_buffer_reuse(graph: HeterogeneousGraph, liveness: LivenessAnalysis) -> BufferReusePlan:
    """Assign activation tensors to the fewest slots that respect live ranges.

    Parameters and mutable tensors never share slots with unrelated values.
    """
    intervals = liveness.intervals
    candidates = [
        tid
        for tid, tensor in graph.tensors.items()
        if tensor.kind == "activation" and not tensor.mutable and tensor.size_bytes > 0
    ]
    candidates.sort(key=lambda tid: (-graph.tensors[tid].size_bytes, tid))

    slots: list[BufferSlot] = []
    assignment: dict[str, int] = {}
    naive = 0
    for tid in candidates:
        tensor = graph.tensors[tid]
        naive += tensor.size_bytes
        interval = intervals.get(tid, (None, None))
        placed = False
        for slot in slots:
            if tensor.size_bytes > slot.capacity_bytes:
                continue
            conflict = False
            for other in slot.assigned:
                if ranges_overlap(interval, intervals.get(other, (None, None))):
                    conflict = True
                    break
            if conflict:
                continue
            slot.assigned.append(tid)
            assignment[tid] = slot.slot_id
            placed = True
            break
        if not placed:
            slot = BufferSlot(slot_id=len(slots), capacity_bytes=tensor.size_bytes, assigned=[tid])
            slots.append(slot)
            assignment[tid] = slot.slot_id

    reused = sum(slot.capacity_bytes for slot in slots)
    saved = max(0, naive - reused)
    notes = [
        f"activation_tensors={len(candidates)}",
        f"slots={len(slots)}",
        f"naive_bytes={naive}",
        f"pooled_bytes={reused}",
        f"saved_bytes={saved}",
    ]
    return BufferReusePlan(assignment=assignment, slots=slots, saved_bytes=saved, notes=notes)


def assert_reuse_safe(plan: BufferReusePlan, liveness: LivenessAnalysis) -> None:
    """Raise if any slot holds overlapping live ranges (regression guard)."""
    by_slot: dict[int, list[str]] = {}
    for tid, slot_id in plan.assignment.items():
        by_slot.setdefault(slot_id, []).append(tid)
    for slot_id, tids in by_slot.items():
        for i, left in enumerate(tids):
            for right in tids[i + 1 :]:
                if ranges_overlap(liveness.intervals[left], liveness.intervals[right]):
                    raise RuntimePlanError(
                        f"Unsafe buffer reuse in slot {slot_id}: {left} overlaps {right}"
                    )
