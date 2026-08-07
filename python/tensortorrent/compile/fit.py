"""Compile-time fit policy: streaming vs resident, region state budgets, fusion.

Single source of truth for decisions that used to be re-derived (and drift) in
``entry``, ``specialize``, and ``provisioning``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from tensortorrent.config import CompileConfig
from tensortorrent.ir.resource_graph import ResourceGraph

logger = logging.getLogger(__name__)

# Fraction of accelerator allocatable bytes reserved for one region's parameters.
# Remainder covers activations, outputs, allocator fragmentation, and workspace.
ACCELERATOR_REGION_STATE_FRACTION = 0.70


def needs_parameter_streaming(config: CompileConfig, *, state_bytes: int) -> bool:
    """True when the runtime must use a disk-backed StreamingParameterStore.

    Matches :func:`tensortorrent.runtime.provisioning.build_parameter_store`:
    stream only when a RAM budget is set and total state exceeds it.
    """
    budget = config.ram_budget_bytes
    if budget is None or int(state_bytes) <= int(budget):
        return False
    return bool(config.allow_nvme_streaming)


def should_hoist_resident_parameters(config: CompileConfig, *, state_bytes: int) -> bool:
    """Keep device parameter copies across forwards only when state fits VRAM."""
    if config.allow_training:
        return False
    vram = config.vram_budget_bytes
    return vram is None or int(state_bytes) <= int(vram)


def exported_parameter_bytes(exported: Any) -> int:
    """Best-effort byte count of parameters/buffers in an ExportedProgram."""
    total = 0
    state = getattr(exported, "state_dict", None)
    if callable(state):
        try:
            state = state()
        except Exception:  # noqa: BLE001 - estimate only; miss → allow fusion
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


def streaming_region_budget(config: CompileConfig, *, parameter_bytes: int | None = None) -> int | None:
    """Per-region parameter budget implied by the host RAM budget.

    With prefetching, the runtime may hold the current region's pins plus up to
    ``prefetch_distance`` successors → cap ``budget / (1 + prefetch_distance)``.

    When ``parameter_bytes`` is known and already fits in ``ram_budget_bytes``,
    return ``None``: the runtime keeps a resident store, so a RAM budget alone
    must not force multi-region shards / disable fusion.
    """
    if config.ram_budget_bytes is None:
        return None
    if parameter_bytes is not None and parameter_bytes > 0 and parameter_bytes <= int(config.ram_budget_bytes):
        return None
    divisor = max(1, 1 + max(0, int(config.prefetch_distance)))
    return max(1, int(config.ram_budget_bytes) // divisor)


def region_state_budget(
    config: CompileConfig,
    machine: ResourceGraph | None,
    *,
    parameter_bytes: int | None = None,
) -> int | None:
    """State cap so regions stay executable under bounded RAM and VRAM.

    ``parameter_bytes`` is forwarded to :func:`streaming_region_budget` so a
    RAM budget that already covers the full model does not force multi-region
    shards (resident store path).
    """
    candidates: list[int] = []
    streaming = streaming_region_budget(config, parameter_bytes=parameter_bytes)
    if streaming is not None:
        candidates.append(streaming)

    if machine is not None and config.allow_gpu:
        from tensortorrent.ir.resource_graph import ComputeClass

        for device in machine.compute.values():
            if device.compute_class not in {
                ComputeClass.DISCRETE_GPU,
                ComputeClass.INTEGRATED_GPU,
                ComputeClass.ACCELERATOR,
            }:
                continue
            if device.compute_class == ComputeClass.INTEGRATED_GPU and not config.allow_integrated_gpu:
                continue
            capacity = sum(
                max(0, int(machine.memory[name].allocatable_bytes))
                for name in device.memory_affinity
                if name in machine.memory
            )
            if config.vram_budget_bytes is not None:
                capacity = min(capacity, config.vram_budget_bytes) if capacity > 0 else config.vram_budget_bytes
            if capacity > 0:
                candidates.append(max(1, int(capacity * ACCELERATOR_REGION_STATE_FRACTION)))
    elif config.vram_budget_bytes is not None and config.allow_gpu:
        candidates.append(max(1, int(config.vram_budget_bytes * ACCELERATOR_REGION_STATE_FRACTION)))

    return min(candidates) if candidates else None


def exceeds_accelerator_region_budget(
    config: CompileConfig,
    machine: ResourceGraph,
    *,
    parameter_bytes: int,
) -> bool:
    """True when full parameters cannot fit the per-region accelerator budget."""
    budget = region_state_budget(config, machine, parameter_bytes=parameter_bytes)
    if budget is None:
        return False
    return int(parameter_bytes) > int(budget)


def should_force_single_region(
    config: CompileConfig,
    machine: ResourceGraph,
    *,
    parameter_bytes: int,
) -> bool:
    """Whether lowering should collapse the graph into one region.

    Single-region fusion is the fast path when concurrency is off. It must not
    run when training needs a multi-piece schedule, host RAM streaming needs
    per-region shards, concurrent multi-region is enabled, or the full model
    exceeds the accelerator region state budget (one fused region → GPU
    placement infeasible → CPU-only).
    """
    if config.allow_training:
        return False
    if streaming_region_budget(config, parameter_bytes=parameter_bytes) is not None:
        return False
    if config.allow_concurrent_regions and config.max_concurrent_regions != 1:
        return False
    if exceeds_accelerator_region_budget(config, machine, parameter_bytes=parameter_bytes):
        logger.info(
            "skip force_single_region: parameters exceed accelerator region budget "
            "(keep partitions so accelerators remain placeable)"
        )
        return False
    return True
