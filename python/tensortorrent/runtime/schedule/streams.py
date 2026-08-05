"""Stream and instruction attribute helpers for executable schedules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from tensortorrent.ir.graph import OpCode
from tensortorrent.runtime.schedule.types import ExecutableSchedule, PlanInstruction


def default_stream_id(opcode: OpCode, resource: str) -> str:
    res = resource or "unknown"
    if opcode == OpCode.COMPUTE:
        return f"{res}::compute"
    if opcode in (OpCode.TRANSFER, OpCode.PREFETCH, OpCode.LOAD):
        return f"{res}::copy0"
    if opcode in (OpCode.RECORD_EVENT, OpCode.WAIT_EVENT):
        return f"{res}::sync"
    return f"{res}::lifetime"


def with_explicit_streams(inst: PlanInstruction) -> PlanInstruction:
    """Fill stream / copy-engine / link / I/O-queue ids when the planner omitted them."""
    stream_id = inst.stream_id or default_stream_id(inst.opcode, inst.resource)
    copy_engine_id = inst.copy_engine_id
    link_id = inst.link_id
    io_queue_id = inst.io_queue_id
    if inst.opcode in (OpCode.TRANSFER, OpCode.PREFETCH, OpCode.LOAD) and not copy_engine_id:
        copy_engine_id = f"{inst.resource or 'unknown'}::copy0"
    if inst.opcode == OpCode.TRANSFER and not link_id:
        src = inst.source or "unknown"
        dst = inst.destination or inst.resource or "unknown"
        link_id = f"{src}->{dst}"
    if inst.opcode in (OpCode.PREFETCH, OpCode.LOAD) and not io_queue_id:
        io_queue_id = f"{inst.resource or 'unknown'}::io0"
    if (
        stream_id == inst.stream_id
        and copy_engine_id == inst.copy_engine_id
        and link_id == inst.link_id
        and io_queue_id == inst.io_queue_id
    ):
        return inst
    return replace(
        inst,
        stream_id=stream_id,
        copy_engine_id=copy_engine_id,
        link_id=link_id,
        io_queue_id=io_queue_id,
    )


def ensure_explicit_streams(schedule: ExecutableSchedule) -> ExecutableSchedule:
    """Return schedule with every instruction carrying explicit stream resources."""
    new_insts = tuple(with_explicit_streams(i) for i in schedule.instructions)
    if new_insts == schedule.instructions:
        return schedule
    return replace(schedule, instructions=new_insts)


def with_instruction_attributes(
    schedule: ExecutableSchedule,
    updates: Mapping[str, Mapping[str, Any]],
) -> ExecutableSchedule:
    """Return a new schedule with merged instruction attributes (tests / tooling)."""
    if not updates:
        return schedule
    new_insts: list[PlanInstruction] = []
    for inst in schedule.instructions:
        patch = updates.get(inst.name)
        if patch is None:
            new_insts.append(inst)
            continue
        merged = {**dict(inst.attributes), **dict(patch)}
        new_insts.append(replace(inst, attributes=merged))
    return replace(schedule, instructions=tuple(new_insts))
