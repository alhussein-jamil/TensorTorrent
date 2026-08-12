"""Pinned-host memory policy derived from executable schedules and hardware."""

from __future__ import annotations

import torch

from tensortorrent.closed import InstructionKind


def _schedule_opcode_name(inst: object) -> str:
    opcode = getattr(inst, "opcode", None)
    return str(getattr(opcode, "value", opcode))


def schedule_uses_pinned_staging(schedule: object | None) -> bool:
    """Return whether the schedule materializes parameters into pinned host memory."""
    if schedule is None:
        return False
    for inst in getattr(schedule, "instructions", ()) or ():
        if _schedule_opcode_name(inst) != "Load":
            continue
        attrs = getattr(inst, "attributes", None) or {}
        if attrs.get("kind") != InstructionKind.PARAMETER_MATERIALIZE:
            continue
        destination = str(getattr(inst, "destination", None) or getattr(inst, "resource", "") or "")
        if "pinned" in destination.lower():
            return True
    return False


def schedule_needs_host_pin(schedule: object | None) -> bool:
    """Return whether host weights should be page-locked for DMA H2D."""
    if schedule is None:
        return False
    if schedule_uses_pinned_staging(schedule):
        return True
    for inst in getattr(schedule, "instructions", ()) or ():
        if _schedule_opcode_name(inst) != "Transfer":
            continue
        attrs = getattr(inst, "attributes", None) or {}
        if attrs.get("kind") == InstructionKind.PARAMETER_HOST_TO_DEVICE:
            return True
    return False


def pinned_host_allocatable_bytes(machine: object | None) -> int | None:
    """Return the smallest discovered pinned-host allocatable pool, if any."""
    if machine is None:
        return None
    memory = getattr(machine, "memory", None)
    if not isinstance(memory, dict) or not memory:
        return None

    from tensortorrent.ir.resource_graph import MemoryClass

    capacities: list[int] = []
    for name, mem in memory.items():
        memory_class = getattr(mem, "memory_class", None)
        class_name = str(getattr(memory_class, "value", memory_class) or "").lower()
        is_pinned = memory_class == MemoryClass.PINNED_HOST or "pinned" in class_name or "pinned" in str(name).lower()
        if not is_pinned:
            continue
        allocatable = int(getattr(mem, "allocatable_bytes", 0) or 0)
        capacity = int(getattr(mem, "capacity_bytes", 0) or 0)
        value = allocatable if allocatable > 0 else capacity
        if value > 0:
            capacities.append(value)
    return min(capacities) if capacities else None


def resolve_parameter_pin(
    *,
    wants_pin: bool,
    state_bytes: int,
    machine: object | None,
    streaming: bool,
    allow_training: bool,
) -> bool:
    """Decide whether the parameter store should page-lock host tensors."""
    if not wants_pin or allow_training or not torch.cuda.is_available():
        return False
    if streaming:
        # Streaming pins one region at a time rather than the full model.
        return True
    pinned = pinned_host_allocatable_bytes(machine)
    return pinned is None or int(state_bytes) <= int(pinned)


def should_pin_parameter_store(
    schedule: object | None,
    *,
    state_bytes: int,
    machine: object | None = None,
    streaming: bool = False,
    allow_training: bool = False,
) -> bool:
    """Return whether schedule intent and capacity make parameter pinning safe."""
    return resolve_parameter_pin(
        wants_pin=schedule_needs_host_pin(schedule),
        state_bytes=state_bytes,
        machine=machine,
        streaming=streaming,
        allow_training=allow_training,
    )
