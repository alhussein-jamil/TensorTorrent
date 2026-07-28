"""Buffering and overlap helpers for streaming schedules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BufferingPlan:
    depth: int  # 2 = double, 3 = triple
    compute_slot: int
    transfer_slot: int
    prepare_slot: int | None
    storage_slot: int | None

    def describe(self) -> str:
        lines = [
            f"buffering_depth={self.depth}",
            f"GPU computes block i in slot {self.compute_slot}",
            f"copy engine receives weights for block i+1 in slot {self.transfer_slot}",
        ]
        if self.prepare_slot is not None:
            lines.append(
                f"CPU prepares/transforms weights for block i+2 in slot {self.prepare_slot}"
            )
        if self.storage_slot is not None:
            lines.append(f"NVMe reads weights for block i+3 in slot {self.storage_slot}")
        lines.append("previous buffers are released or evicted when lifetimes end")
        return "\n".join(lines)


def choose_buffering(
    *,
    has_copy_engine: bool,
    has_cpu_prepare: bool,
    has_nvme: bool,
) -> BufferingPlan:
    if has_nvme and has_cpu_prepare and has_copy_engine:
        return BufferingPlan(3, 0, 1, 2, 0)
    if has_copy_engine and has_cpu_prepare:
        return BufferingPlan(3, 0, 1, 2, None)
    if has_copy_engine:
        return BufferingPlan(2, 0, 1, None, None)
    return BufferingPlan(1, 0, 0, None, None)


def exposed_transfer_latency(transfer_latency_s: float, useful_independent_work_s: float) -> float:
    """exposed_transfer_latency = max(0, transfer_latency - useful_independent_work)."""
    return max(0.0, transfer_latency_s - useful_independent_work_s)
