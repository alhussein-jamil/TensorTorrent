"""Validate executable schedules.

Structural + residency rules live in Rust ``tt_ir::validate``. Python fills
explicit stream/engine ids then delegates so compile-time and execute-time
checks cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tensortorrent.config import Objective
from tensortorrent.ir.graph import OpCode
from tensortorrent.native import require_native
from tensortorrent.planner.maximal import ExecutionPlan, Placement
from tensortorrent.runtime.schedule.streams import ensure_explicit_streams
from tensortorrent.runtime.schedule.types import ExecutableSchedule, ScheduleValidationError


def validate_schedule(schedule: ExecutableSchedule) -> list[str]:
    """Check structural invariants a runtime/simulator must be able to rely on.

    Returns a list of human-readable violations; empty means the schedule is
    safe to simulate or execute. Never silently drops or reorders instructions
    to "fix" a bad schedule -- planner bugs must surface, not be papered over.

    Source of truth: Rust ``tt_ir::validate_schedule`` via the native extension.
    """
    schedule = ensure_explicit_streams(schedule)
    return list(require_native().validate_schedule(schedule))


def validate_schedule_tensor_sizes(schedule: ExecutableSchedule) -> list[str]:
    """Require exact per-tensor sizes on a specialized executable schedule.

    Zero-byte scalar/control values are allowed only when the instruction itself
    reports zero bytes. Tensor-bearing movement and compute operations must not
    rely on aggregate equal-split guesses.
    """
    errors: list[str] = []
    for inst in schedule.instructions:
        tensors = tuple(dict.fromkeys((*inst.inputs, *inst.outputs)))
        if not tensors:
            continue
        if inst.opcode not in {
            OpCode.PREFETCH,
            OpCode.LOAD,
            OpCode.TRANSFER,
            OpCode.COMPUTE,
            OpCode.EVICT,
            OpCode.RELEASE,
        }:
            continue
        raw = inst.attributes.get("tensor_nbytes")
        sizes = dict(raw) if isinstance(raw, Mapping) else {}
        for tensor in tensors:
            if tensor in sizes:
                try:
                    size = int(sizes[tensor])
                except (TypeError, ValueError):
                    errors.append(f"instruction {inst.name!r} has invalid byte size for {tensor!r}")
                    continue
                if size < 0:
                    errors.append(f"instruction {inst.name!r} has negative byte size for {tensor!r}")
                continue
            if len(tensors) == 1 and int(inst.nbytes or 0) >= 0:
                continue
            errors.append(f"instruction {inst.name!r} lacks exact tensor_nbytes for {tensor!r}")
    return errors


def validate_schedule_resources(schedule: ExecutableSchedule, machine: Any) -> list[str]:
    """Check that every Compute instruction names a real compute resource.

    Transfer/Prefetch ``resource`` labels are synthetic engine identifiers
    (e.g. ``copy_engine:src->dst``), not resource-graph ids, so only Compute
    placements -- which must run on an actual discovered device -- are
    checked here. ``machine`` is a :class:`~tensortorrent.ir.resource_graph.ResourceGraph`.
    """
    errors: list[str] = []
    known = set(machine.compute)
    for inst in schedule.compute_ops():
        if inst.resource not in known:
            errors.append(f"compute {inst.name!r} references unknown compute resource {inst.resource!r}")
    return errors


def assert_schedule_valid(schedule: ExecutableSchedule) -> None:
    errors = validate_schedule(schedule)
    if errors:
        raise ScheduleValidationError(f"ExecutableSchedule {schedule.graph_name!r} failed validation: {errors}")


def _transfer_resource(source: str, destination: str) -> str:
    return f"copy_engine:{source}->{destination}"


def schedule_matches_plan(schedule: ExecutableSchedule, plan: ExecutionPlan) -> list[str]:
    """Return mismatch reasons; empty means schedule covers every placement."""
    compute_refs = {i.executable_ref for i in schedule.compute_ops()}
    errors: list[str] = []
    for placement in plan.placements:
        if placement.region_id not in compute_refs:
            errors.append(f"missing compute for {placement.region_id}")
    for inst in schedule.compute_ops():
        if inst.executable_ref not in {p.region_id for p in plan.placements}:
            errors.append(f"orphan compute {inst.name}")
    return errors


def schedule_from_bindings(
    program: Any,
    bindings: dict[str, Any],
    *,
    streaming: bool = False,
    prefetch_distance: int = 0,
    fingerprint: str = "",
) -> ExecutableSchedule:
    """Build an ExecutableSchedule from region bindings when no plan schedule exists.

    Used so GraphExecutor never falls back to a region topo walker: every run
    goes through the instruction DAG.
    """
    from tensortorrent.runtime.residency import attach_residency_to_plan

    output_producer: dict[str, str] = {}
    for region in program.regions:
        for name in region.outputs:
            output_producer[name] = region.region_id

    placements: list[Placement] = []
    for region in program.regions:
        binding = bindings[region.region_id]
        deps: list[str] = []
        for inp in region.inputs:
            if inp in program.state_bindings:
                continue
            producer = output_producer.get(inp)
            if producer is not None and producer not in deps:
                deps.append(producer)
        state_bytes = 0
        state_map = program.state_tensors() if hasattr(program, "state_tensors") else {}
        for name in region.state_inputs:
            tensor = state_map.get(name) if isinstance(state_map, dict) else None
            if tensor is not None and hasattr(tensor, "numel"):
                state_bytes += int(tensor.numel() * tensor.element_size())
            else:
                # Unknown size: still force a Load so Compute never materializes weights.
                state_bytes = max(state_bytes, 1)
        if region.state_inputs and state_bytes <= 0:
            state_bytes = 1
        placements.append(
            Placement(
                region_id=region.region_id,
                device=binding.device,
                backend_id=binding.backend_id,
                dtype="float32",
                kernel_id="eager",
                estimated_latency_s=0.0,
                depends_on=tuple(deps),
                measured=False,
                output_bytes=0,
                state_bytes=state_bytes,
            )
        )
    # Placement list order must be producer-before-consumer so schedule emission
    # can resolve last_compute when wiring Transfer deps (region list may be shuffled).
    by_id = {p.region_id: p for p in placements}
    remaining = {p.region_id for p in placements}
    ordered: list[Placement] = []
    while remaining:
        progressed = False
        for rid in list(remaining):
            dep_ids = by_id[rid].depends_on
            if all(d not in remaining for d in dep_ids):
                ordered.append(by_id[rid])
                remaining.remove(rid)
                progressed = True
        if not progressed:
            # Cycle / missing dep — append rest in original order.
            ordered.extend(by_id[rid] for rid in list(remaining))
            break
    placements = ordered
    devices = tuple(dict.fromkeys(p.device for p in placements))
    plan = ExecutionPlan(
        graph_name=program.graph_name,
        fingerprint=fingerprint or "bindings",
        objective=Objective.LATENCY,
        placements=placements,
        decisions=[],
        devices_used=devices,
        communication_backend="none",
        predicted_latency_s=0.0,
        strategy="bindings_schedule",
        notes=["schedule rebuilt from region bindings"],
    )
    residency = attach_residency_to_plan(plan, program)
    from tensortorrent.runtime.schedule.build import build_executable_schedule

    return build_executable_schedule(
        plan,
        residency,
        streaming=streaming,
        prefetch_distance=prefetch_distance,
        program=program,
    )
