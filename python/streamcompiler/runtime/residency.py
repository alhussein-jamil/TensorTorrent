"""Explicit tensor residency and transfer scheduling.

Derives cross-device transfers from real region input/output tensor ids so the
executable schedule never invents synthetic ``activation::region`` names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from streamcompiler.compile.regions import RegionProgram
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


def build_residency_schedule(
    plan: ExecutionPlan,
    program: RegionProgram | None = None,
) -> ResidencySchedule:
    """Derive residency requirements and cross-device transfers from a plan.

    When ``program`` is provided, transfers use exact region input tensor ids
    (supporting multi-output fan-out). Without a program, uses one synthetic
    activation name per producer (hand-built plans / tests).
    """
    by_id = {p.region_id: p for p in plan.placements}
    output_producer: dict[str, str] = {}
    region_outputs: dict[str, tuple[str, ...]] = {}
    if program is not None:
        for region in program.regions:
            region_outputs[region.region_id] = region.outputs
            for name in region.outputs:
                output_producer[name] = region.region_id

    by_region: dict[str, list[ResidencyRequirement]] = {}
    transfers: list[ScheduledTransfer] = []
    multi_device = len(plan.devices_used) > 1
    # Deduplicate identical transfers (shared consumers).
    seen_xfer: set[tuple[str, str, str]] = set()

    for placement in plan.placements:
        reqs: list[ResidencyRequirement] = []
        if placement.state_bytes > 0:
            state_names: tuple[str, ...] = ()
            if program is not None:
                state_names = program.region_by_id(placement.region_id).state_inputs
            if not state_names:
                state_names = (f"state::{placement.region_id}",)
            for sname in state_names:
                nbytes = 0
                if program is not None:
                    spec = getattr(program, "values", {}).get(sname)
                    nbytes = int(getattr(spec, "nbytes", 0) or 0) if spec is not None else 0
                if nbytes <= 0:
                    nbytes = (
                        max(1, int(placement.state_bytes or 1))
                        if len(state_names) == 1
                        else max(1, int(placement.state_bytes or 1) // max(1, len(state_names)))
                    )
                reqs.append(
                    ResidencyRequirement(
                        value_name=sname,
                        device=placement.device,
                        nbytes=nbytes,
                        kind="parameter",
                    )
                )

        consumer_inputs: tuple[str, ...] = ()
        if program is not None:
            consumer_inputs = program.region_by_id(placement.region_id).inputs

        if consumer_inputs:
            for input_name in consumer_inputs:
                if input_name in program.state_bindings:  # type: ignore[union-attr]
                    continue
                producer_id = output_producer.get(input_name)
                if producer_id is None:
                    # User input — seeded on host; copy to consumer when placement is not host.
                    reqs.append(
                        ResidencyRequirement(
                            value_name=input_name,
                            device=placement.device,
                            nbytes=0,
                            kind="input",
                        )
                    )
                    hostish = any(tok in placement.device.lower() for tok in ("cpu", "numa", "host"))
                    if not hostish:
                        # Prefer a real host compute resource from the plan when present.
                        host_src = "cpu"
                        for p in plan.placements:
                            if any(tok in p.device.lower() for tok in ("cpu", "numa", "host")):
                                host_src = p.device
                                break
                        key = (input_name, host_src, placement.device)
                        if key not in seen_xfer:
                            seen_xfer.add(key)
                            transfers.append(
                                ScheduledTransfer(
                                    value_name=input_name,
                                    source_device=host_src,
                                    destination_device=placement.device,
                                    nbytes=1,
                                    after_region="",
                                    before_region=placement.region_id,
                                )
                            )
                    continue
                producer = by_id.get(producer_id)
                if producer is None:
                    continue
                nbytes = 0
                if program is not None:
                    spec = getattr(program, "values", {}).get(input_name)
                    nbytes = int(getattr(spec, "nbytes", 0) or 0) if spec is not None else 0
                if nbytes <= 0:
                    outs = region_outputs.get(producer_id, (input_name,))
                    nbytes = max(0, int(producer.output_bytes or 0)) if len(outs) == 1 else 0
                reqs.append(
                    ResidencyRequirement(
                        value_name=input_name,
                        device=placement.device,
                        nbytes=nbytes,
                        kind="activation",
                    )
                )
                if producer.device != placement.device and nbytes >= 0:
                    key = (input_name, producer.device, placement.device)
                    if key not in seen_xfer:
                        seen_xfer.add(key)
                        transfers.append(
                            ScheduledTransfer(
                                value_name=input_name,
                                source_device=producer.device,
                                destination_device=placement.device,
                                nbytes=max(1, nbytes),
                                after_region=producer_id,
                                before_region=placement.region_id,
                            )
                        )
        else:
            # Planner path when RegionProgram is unavailable (tests / hand-built plans).
            for dep in placement.depends_on:
                producer = by_id.get(dep)
                if producer is None:
                    continue
                nbytes = max(0, producer.output_bytes)
                value_name = f"activation::{dep}"
                reqs.append(
                    ResidencyRequirement(
                        value_name=value_name,
                        device=placement.device,
                        nbytes=nbytes,
                        kind="activation",
                    )
                )
                if producer.device != placement.device and nbytes > 0:
                    transfers.append(
                        ScheduledTransfer(
                            value_name=value_name,
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
        notes.append("multi_device_plan: residency tracks tensor ids and cross-device copies")
    else:
        notes.append("single_device_plan: no cross-device transfers required")

    return ResidencySchedule(
        by_region={k: tuple(v) for k, v in by_region.items()},
        transfers=tuple(transfers),
        notes=tuple(notes),
    )


def attach_residency_to_plan(
    plan: ExecutionPlan,
    program: RegionProgram | None = None,
) -> ResidencySchedule:
    """Build the schedule and record a short note on the plan."""
    schedule = build_residency_schedule(plan, program)
    for note in schedule.notes:
        if note not in plan.notes:
            plan.notes.append(note)
    if schedule.transfers:
        plan.notes.append(f"scheduled_transfers={len(schedule.transfers)}")
    return schedule
