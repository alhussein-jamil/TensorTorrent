"""Schedule execution report types (no dependency on ScheduleExecutor)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InstructionEvent:
    name: str
    opcode: str
    resource: str
    submitted_s: float
    start_s: float
    end_s: float
    nbytes: int = 0
    notes: str = ""
    prefetch_hit: bool | None = None
    exposed_stall_s: float = 0.0
    enqueue_start_s: float = 0.0
    enqueue_end_s: float = 0.0
    complete_s: float = 0.0
    consumer_wait_s: float = 0.0
    simulated: bool = False
    region_id: str | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def max_concurrency_from_intervals(intervals: list[tuple[float, float]]) -> int:
    """Peak concurrency via sweep over half-open ``[start, end)`` intervals."""
    points: list[tuple[float, int]] = []
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        points.append((start, 1))
        points.append((end, -1))
    # Ends (-1) before starts (+1) at the same timestamp.
    points.sort(key=lambda p: (p[0], p[1]))
    cur = peak = 0
    for _, delta in points:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


@dataclass
class ScheduleReport:
    wall_time_s: float
    events: list[InstructionEvent] = field(default_factory=list)
    parallel_overlaps: int = 0
    max_concurrent: int = 1
    copy_snapshot: dict[str, Any] = field(default_factory=dict)
    parameter_store: dict[str, Any] = field(default_factory=dict)
    multi_copy_peaks: list[dict[str, Any]] = field(default_factory=list)
    peak_activation_bytes: int = 0
    activation_bytes_written: int = 0
    activation_bytes_read: int = 0
    spill_latency_s: float = 0.0
    reload_latency_s: float = 0.0
    allocation_peak_bytes: int = 0
    spill_events: list[dict[str, Any]] = field(default_factory=list)

    def overlapping_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        ordered = sorted(self.events, key=lambda e: e.start_s)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                if second.start_s >= first.end_s:
                    break
                if (
                    first.opcode == "Compute"
                    and second.opcode == "Compute"
                    or {first.opcode, second.opcode} & {"Transfer", "Compute", "Prefetch", "Load"}
                ):
                    pairs.append((first.name, second.name))
        return pairs

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_time_s": self.wall_time_s,
            "instruction_count": len(self.events),
            "parallel_overlaps": self.parallel_overlaps,
            "max_concurrent": self.max_concurrent,
            "copy_snapshot": self.copy_snapshot,
            "multi_copy_peaks": self.multi_copy_peaks,
            "peak_activation_bytes": self.peak_activation_bytes,
            "activation_bytes_written": self.activation_bytes_written,
            "activation_bytes_read": self.activation_bytes_read,
            "spill_latency_s": self.spill_latency_s,
            "reload_latency_s": self.reload_latency_s,
            "allocation_peak_bytes": self.allocation_peak_bytes,
            "parameter_store": self.parameter_store,
            "instructions": [
                {
                    "name": e.name,
                    "opcode": e.opcode,
                    "resource": e.resource,
                    "duration_s": e.duration_s,
                    "nbytes": e.nbytes,
                    "prefetch_hit": e.prefetch_hit,
                    "exposed_stall_s": e.exposed_stall_s,
                    "notes": e.notes,
                }
                for e in self.events
            ],
        }
