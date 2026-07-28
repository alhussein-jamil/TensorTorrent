"""Explicit tensor residency and transfer scheduling (CPU–GPU prep).

No GPU backend stubs live here. These types describe where a value must live and
which transfers a future multi-device executor must issue before a region runs.
Today the CPU executor keeps activations in host RAM; this module only records
the schedule so mixed-device plans can be validated later with real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from streamcompiler.planner.maximal import ExecutionPlan


@dataclass(frozen=True)
class ResidencyRequirement:
    """Where one named value must be resident for a region to start."""

    value_name: str
    device: str
    nbytes: int
    kind: str  # parameter | activation | input


@dataclass(frozen=True)
class ScheduledTransfer:
    """A required movement of bytes between devices before a consumer runs."""

    value_name: str
    source_device: str
    destination_device: str
    nbytes: int
    after_region: str
    before_region: str


@dataclass
class ResidencySchedule:
    """Per-region residency and the transfers implied by cross-device edges."""

    by_region: dict[str, tuple[ResidencyRequirement, ...]] = field(default_factory=dict)
    transfers: tuple[ScheduledTransfer, ...] = ()
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "by_region": {
                rid: [
                    {
                        "value_name": r.value_name,
                        "device": r.device,
                        "nbytes": r.nbytes,
                        "kind": r.kind,
                    }
                    for r in reqs
                ]
                for rid, reqs in self.by_region.items()
            },
            "transfers": [
                {
                    "value_name": t.value_name,
                    "source_device": t.source_device,
                    "destination_device": t.destination_device,
                    "nbytes": t.nbytes,
                    "after_region": t.after_region,
                    "before_region": t.before_region,
                }
                for t in self.transfers
            ],
            "notes": list(self.notes),
        }


def build_residency_schedule(plan: ExecutionPlan) -> ResidencySchedule:
    """Derive residency requirements and cross-device transfers from a plan.

    Activations are named by producing region id until IR value names are wired
    through placements. Transfers are only emitted when a dependency crosses
    devices; same-device edges need no copy.
    """
    by_id = {p.region_id: p for p in plan.placements}
    by_region: dict[str, list[ResidencyRequirement]] = {}
    transfers: list[ScheduledTransfer] = []
    multi_device = len(plan.devices_used) > 1

    for placement in plan.placements:
        reqs: list[ResidencyRequirement] = []
        if placement.state_bytes > 0:
            reqs.append(
                ResidencyRequirement(
                    value_name=f"state::{placement.region_id}",
                    device=placement.device,
                    nbytes=placement.state_bytes,
                    kind="parameter",
                )
            )
        for dep in placement.depends_on:
            producer = by_id.get(dep)
            if producer is None:
                continue
            nbytes = max(0, producer.output_bytes)
            reqs.append(
                ResidencyRequirement(
                    value_name=f"activation::{dep}",
                    device=placement.device,
                    nbytes=nbytes,
                    kind="activation",
                )
            )
            if producer.device != placement.device and nbytes > 0:
                transfers.append(
                    ScheduledTransfer(
                        value_name=f"activation::{dep}",
                        source_device=producer.device,
                        destination_device=placement.device,
                        nbytes=nbytes,
                        after_region=dep,
                        before_region=placement.region_id,
                    )
                )
        by_region[placement.region_id] = reqs

    notes: list[str] = []
    if multi_device:
        notes.append(
            "multi_device_plan: residency schedule prepared; simultaneous CPU–GPU "
            "execution remains unvalidated until run on real accelerators"
        )
    else:
        notes.append("single_device_plan: no cross-device transfers required")

    return ResidencySchedule(
        by_region={k: tuple(v) for k, v in by_region.items()},
        transfers=tuple(transfers),
        notes=tuple(notes),
    )


def attach_residency_to_plan(plan: ExecutionPlan) -> ResidencySchedule:
    """Build the schedule and record a short note on the plan."""
    schedule = build_residency_schedule(plan)
    for note in schedule.notes:
        if note not in plan.notes:
            plan.notes.append(note)
    if schedule.transfers:
        plan.notes.append(f"scheduled_transfers={len(schedule.transfers)}")
    return schedule
