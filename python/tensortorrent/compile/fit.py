"""Compile-time fit policy for residency, streaming, region budgets, and fusion.

Keep all memory-fit decisions here so compile, specialization, and runtime
provisioning use the same capacity model.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from tensortorrent.config import CompileConfig
from tensortorrent.ir.resource_graph import ResourceGraph

logger = logging.getLogger(__name__)

# Fraction of accelerator allocatable bytes available to one region's state.
# The remainder is reserved for activations, outputs, allocator fragmentation,
# staging, and backend workspace.
ACCELERATOR_REGION_STATE_FRACTION = 0.70


def needs_parameter_streaming(config: CompileConfig, *, state_bytes: int) -> bool:
    """Return whether the runtime must use a disk-backed parameter store."""
    budget = config.ram_budget_bytes
    if budget is None or int(state_bytes) <= int(budget):
        return False
    return bool(config.allow_nvme_streaming)


def accelerator_vram_capacity_bytes(
    config: CompileConfig,
    machine: ResourceGraph | None,
) -> int | None:
    """Return the smallest eligible accelerator capacity after any VRAM cap.

    Using the minimum keeps shards safe on heterogeneous hosts: a region chosen
    for an accelerator must fit the tightest eligible device.
    """
    if not bool(getattr(config, "allow_gpu", True)):
        return None

    budget = getattr(config, "vram_budget_bytes", None)
    if machine is None:
        return int(budget) if budget is not None else None

    from tensortorrent.ir.resource_graph import ComputeClass

    allow_igpu = bool(getattr(config, "allow_integrated_gpu", True))
    candidates: list[int] = []
    for device in machine.compute.values():
        if device.compute_class not in {
            ComputeClass.DISCRETE_GPU,
            ComputeClass.INTEGRATED_GPU,
            ComputeClass.ACCELERATOR,
        }:
            continue
        if device.compute_class == ComputeClass.INTEGRATED_GPU and not allow_igpu:
            continue

        capacity = sum(
            max(0, int(machine.memory[name].allocatable_bytes))
            for name in device.memory_affinity
            if name in machine.memory
        )
        if budget is not None:
            cap = int(budget)
            capacity = min(capacity, cap) if capacity > 0 else cap
        if capacity > 0:
            candidates.append(capacity)

    return min(candidates) if candidates else None


def accelerator_region_state_budget_bytes(
    config: CompileConfig,
    machine: ResourceGraph | None,
) -> int | None:
    """Return accelerator bytes available to a single region's parameters."""
    capacity = accelerator_vram_capacity_bytes(config, machine)
    if capacity is None:
        return None
    return max(1, int(capacity * ACCELERATOR_REGION_STATE_FRACTION))


def should_hoist_resident_parameters(
    config: CompileConfig,
    *,
    state_bytes: int,
    machine: ResourceGraph | None = None,
) -> bool:
    """Keep device parameter copies only when the full state fits with headroom."""
    if config.allow_training:
        return False
    budget = accelerator_region_state_budget_bytes(config, machine)
    if budget is None:
        return True
    return int(state_bytes) <= budget


def exported_parameter_bytes(exported: Any) -> int:
    """Best-effort byte count of parameters and constants in an ExportedProgram."""
    total = 0
    state = getattr(exported, "state_dict", None)
    if callable(state):
        try:
            state = state()
        except Exception:  # noqa: BLE001 - estimate only; disables a fit optimization
            state = None
    if isinstance(state, dict):
        for value in state.values():
            if torch.is_tensor(value):
                total += int(value.numel()) * int(value.element_size())
        if total > 0:
            return total

    constants = getattr(exported, "constants", None)
    if isinstance(constants, dict):
        for value in constants.values():
            if torch.is_tensor(value):
                total += int(value.numel()) * int(value.element_size())
    return total


def streaming_region_budget(
    config: CompileConfig,
    *,
    parameter_bytes: int | None = None,
) -> int | None:
    """Return the per-region state budget implied by bounded host RAM.

    Prefetch may retain the current region plus ``prefetch_distance`` successors,
    so each region receives at most ``ram_budget / (1 + prefetch_distance)``.
    If the whole model already fits in the configured RAM budget, the resident
    store path is used and host RAM alone should not force extra partitioning.
    """
    budget = config.ram_budget_bytes
    if budget is None:
        return None
    if (
        parameter_bytes is not None
        and parameter_bytes > 0
        and parameter_bytes <= int(budget)
    ):
        return None
    divisor = max(1, 1 + max(0, int(config.prefetch_distance)))
    return max(1, int(budget) // divisor)


def region_state_budget(
    config: CompileConfig,
    machine: ResourceGraph | None,
    *,
    parameter_bytes: int | None = None,
) -> int | None:
    """Return the strictest per-region budget across RAM and accelerator memory."""
    candidates = [
        budget
        for budget in (
            streaming_region_budget(config, parameter_bytes=parameter_bytes),
            accelerator_region_state_budget_bytes(config, machine),
        )
        if budget is not None
    ]
    return min(candidates) if candidates else None


def exceeds_accelerator_region_budget(
    config: CompileConfig,
    machine: ResourceGraph | None,
    *,
    parameter_bytes: int,
) -> bool:
    """Return whether full parameters exceed the accelerator-only region budget."""
    budget = accelerator_region_state_budget_bytes(config, machine)
    return budget is not None and int(parameter_bytes) > budget


def should_force_single_region(
    config: CompileConfig,
    machine: ResourceGraph,
    *,
    parameter_bytes: int,
) -> bool:
    """Return whether lowering should collapse the graph into one region."""
    if config.allow_training:
        return False
    if streaming_region_budget(config, parameter_bytes=parameter_bytes) is not None:
        return False
    if config.allow_concurrent_regions and config.max_concurrent_regions != 1:
        return False
    if exceeds_accelerator_region_budget(
        config,
        machine,
        parameter_bytes=parameter_bytes,
    ):
        logger.info(
            "skip force_single_region: parameters exceed accelerator region budget "
            "(keep partitions so accelerators remain placeable)"
        )
        return False
    return True
