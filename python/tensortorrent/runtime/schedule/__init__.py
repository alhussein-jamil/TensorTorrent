"""One executable plan format shared by planner, simulator, and runtime.

Instructions are explicit memory/compute ops. The simulator must not invent
schedules the runtime cannot perform: both consume :class:`ExecutableSchedule`.
"""

from __future__ import annotations

from tensortorrent.ir.graph import OpCode
from tensortorrent.runtime.schedule.build import (
    _state_tensors_without_later_use,
    _tier_for_device,
    build_executable_schedule,
    hoist_resident_parameter_transfers,
)
from tensortorrent.runtime.schedule.spill_plan import plan_activation_spills
from tensortorrent.runtime.schedule.streams import (
    default_stream_id,
    ensure_explicit_streams,
    with_explicit_streams,
    with_instruction_attributes,
)
from tensortorrent.runtime.schedule.types import (
    ExecutableSchedule,
    FrozenAttrs,
    MemoryTier,
    PlanInstruction,
    ScheduleValidationError,
)
from tensortorrent.runtime.schedule.validate import (
    assert_schedule_valid,
    schedule_from_bindings,
    schedule_matches_plan,
    validate_schedule,
    validate_schedule_resources,
    validate_schedule_tensor_sizes,
)

__all__ = [
    "ExecutableSchedule",
    "FrozenAttrs",
    "MemoryTier",
    "OpCode",
    "PlanInstruction",
    "ScheduleValidationError",
    "_state_tensors_without_later_use",
    "_tier_for_device",
    "assert_schedule_valid",
    "build_executable_schedule",
    "hoist_resident_parameter_transfers",
    "default_stream_id",
    "ensure_explicit_streams",
    "plan_activation_spills",
    "schedule_from_bindings",
    "schedule_matches_plan",
    "validate_schedule",
    "validate_schedule_resources",
    "validate_schedule_tensor_sizes",
    "with_explicit_streams",
    "with_instruction_attributes",
]
