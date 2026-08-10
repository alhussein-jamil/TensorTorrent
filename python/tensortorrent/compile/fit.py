"""Compile-time fit policy for residency, streaming, region budgets, and fusion.

Keep all memory-fit decisions here so compile, specialization, and runtime
provisioning use the same capacity model.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch

from tensortorrent.config import CompileConfig
from tensortorrent.ir.resource_graph import ResourceGraph

logger = logging.getLogger(__name__)

# Fraction of accelerator allocatable bytes available to one region's state.
# The remainder is reserved for activations, outputs, allocator fragmentation,
# staging, and backend workspace.
ACCELERATOR_REGION_STATE_FRACTION = 0.70

# Workspace / allocator / driver safety floor when sizing a residency budget.
# Prefer ``usable - activation_reserve - safety`` over a fixed hoist fraction so
# activation-heavy and larger-batch workloads keep headroom automatically.
ACCELERATOR_HOIST_SAFETY_FRACTION = 0.08
ACCELERATOR_HOIST_SAFETY_MIN_BYTES = 256 << 20  # 256 MiB

# Pre-CUDA optimistic free fraction for export-free Auto selection. Matches a
# typical idle / display reservation without calling ``torch.cuda.mem_get_info``
# (which would initialize the CUDA context and poison multi-GiB host GEMM).
ASSUMED_IDLE_VRAM_FREE_FRACTION = 0.90

_CUDA_DEVICE_INDEX_RE = re.compile(r"cuda(?:_gpu)?_(\d+)$")


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


def accelerator_hoist_safety_bytes(capacity: int) -> int:
    """Allocator / driver / workspace reserve deducted from usable VRAM."""
    raw = max(ACCELERATOR_HOIST_SAFETY_MIN_BYTES, int(capacity * ACCELERATOR_HOIST_SAFETY_FRACTION))
    # Never let the floor dominate tiny test budgets / mocked capacities.
    return int(min(raw, max(1, capacity // 8)))


def accelerator_activation_reserve_bytes(
    config: CompileConfig,
    capacity: int,
    *,
    activation_bytes: int | None = None,
) -> int:
    """Bytes reserved for activations / workspace beyond the safety floor.

    Prefer an explicit ``CompileConfig.activation_budget_bytes``, then a caller
    estimate (example activations / attention working set). When neither is
    known, reserve 0 here — :func:`accelerator_hoist_safety_bytes` already
    keeps allocator headroom so fit@0.8–0.9× sequential models stay resident.
    """
    del capacity  # sizing uses explicit/estimate only
    explicit = getattr(config, "activation_budget_bytes", None)
    if explicit is not None:
        return max(0, int(explicit))
    if activation_bytes is not None and int(activation_bytes) > 0:
        return max(0, int(activation_bytes))
    return 0


def accelerator_hoist_budget_bytes(
    config: CompileConfig,
    machine: ResourceGraph | None,
    *,
    activation_bytes: int | None = None,
) -> int | None:
    """Persistent parameter residency budget: usable VRAM − activation − safety.

    Falls back to transfer/evict when state exceeds this budget; callers may
    still keep a prefix of parameters resident via
    :func:`select_persistent_parameter_ids`.
    """
    capacity = accelerator_vram_capacity_bytes(config, machine)
    if capacity is None:
        return None
    safety = accelerator_hoist_safety_bytes(capacity)
    activation = accelerator_activation_reserve_bytes(config, capacity, activation_bytes=activation_bytes)
    # Keep at least 1 byte of budget so empty devices do not claim infinite hoist.
    return max(1, int(capacity) - safety - activation)


def cuda_device_index_from_resource(resource: str) -> int | None:
    """Parse ``cuda_0`` / ``cuda_gpu_0`` style resource ids to a device index."""
    match = _CUDA_DEVICE_INDEX_RE.search(str(resource or ""))
    return int(match.group(1)) if match is not None else None


def optimistic_hoist_budget_without_cuda(config: CompileConfig, vram_bytes: int) -> int:
    """Hoist budget for pre-CUDA Auto selection (no ``mem_get_info``).

    Uses the same safety model as live clamping, with
    :data:`ASSUMED_IDLE_VRAM_FREE_FRACTION` standing in for free VRAM so export-
    free CPU selection stays honest without initializing CUDA.
    """
    vram = max(1, int(vram_bytes))
    budget = accelerator_hoist_budget_bytes(config, None)
    if budget is None:
        budget = max(1, vram - accelerator_hoist_safety_bytes(vram))
    assumed_free = max(1, int(vram * ASSUMED_IDLE_VRAM_FREE_FRACTION))
    live = max(0, assumed_free - accelerator_hoist_safety_bytes(assumed_free))
    if live > 0:
        return max(1, min(int(budget), live))
    return max(1, int(budget))


def clamp_hoist_budget_to_live_vram(
    budget_bytes: int,
    *,
    device_indices: set[int] | None = None,
    synchronize: bool = False,
) -> int:
    """Clamp a hoist budget to currently free CUDA VRAM (shared authority).

    ``device_indices`` defaults to ``{0}`` when empty/None. When ``synchronize``
    is True, collect/sync/empty_cache on each target before ``mem_get_info`` so
    compile scratch does not falsely collapse the budget. Bakeoff may pass
    ``synchronize=False`` when the CUDA context is already warm and timing
    matters.
    """
    budget = max(1, int(budget_bytes))
    if not torch.cuda.is_available():
        return budget
    targets = set(device_indices or ())
    if not targets:
        targets.add(0)
    if synchronize:
        import gc

        gc.collect()
    live_limits: list[int] = []
    for index in sorted(targets):
        try:
            if synchronize:
                with torch.cuda.device(index):
                    torch.cuda.synchronize(index)
                    torch.cuda.empty_cache()
            free_b, _total_b = torch.cuda.mem_get_info(index)
        except (RuntimeError, ValueError):
            continue
        live_limits.append(max(0, int(free_b) - accelerator_hoist_safety_bytes(max(1, int(free_b)))))
    if not live_limits:
        return budget
    return max(1, min(budget, min(live_limits)))


def live_hoist_budget_bytes(
    config: CompileConfig,
    machine: ResourceGraph | None = None,
    *,
    device_indices: set[int] | None = None,
    synchronize: bool = False,
    activation_bytes: int | None = None,
) -> int | None:
    """Configured hoist budget clamped to live free VRAM (one call site)."""
    budget = accelerator_hoist_budget_bytes(config, machine, activation_bytes=activation_bytes)
    if budget is None:
        return None
    return clamp_hoist_budget_to_live_vram(
        int(budget),
        device_indices=device_indices,
        synchronize=synchronize,
    )


def should_hoist_resident_parameters(
    config: CompileConfig,
    *,
    state_bytes: int,
    machine: ResourceGraph | None = None,
    activation_bytes: int | None = None,
) -> bool:
    """Keep *all* device parameter copies when full state fits the hoist budget.

    Uses :func:`accelerator_hoist_budget_bytes` (usable − activation − safety),
    not the tighter per-region partition fraction. When this returns False,
    :func:`select_persistent_parameter_ids` may still hoist a subset.
    """
    if config.allow_training:
        return False
    budget = accelerator_hoist_budget_bytes(config, machine, activation_bytes=activation_bytes)
    if budget is None:
        return True
    return int(state_bytes) <= max(1, int(budget))


def select_persistent_parameter_ids(
    parameter_nbytes: dict[str, int],
    *,
    budget_bytes: int,
    transfer_groups: list[tuple[str, ...]] | None = None,
) -> set[str]:
    """Greedy largest-first selection of parameters that fit ``budget_bytes``.

    Remaining ids must continue to transfer/evict each forward. Empty when the
    budget cannot hold even the smallest tensor (safe fallback = full stream).

    When ``transfer_groups`` is provided (coalesced H2D payloads), the selection
    is trimmed so every group's non-resident remainder still fits in
    ``budget_bytes - resident_bytes``. Without that headroom, residents fill the
    device and the next streamed region OOMs.
    """
    if budget_bytes <= 0 or not parameter_nbytes:
        return set()
    ordered = sorted(
        ((max(0, int(nbytes)), name) for name, nbytes in parameter_nbytes.items()),
        key=lambda item: (-item[0], item[1]),
    )
    selected: set[str] = set()
    used = 0
    for nbytes, name in ordered:
        if nbytes <= 0:
            continue
        if used + nbytes > budget_bytes:
            continue
        selected.add(name)
        used += nbytes

    if not transfer_groups or not selected:
        return selected

    sizes = {name: max(0, int(nbytes)) for name, nbytes in parameter_nbytes.items()}

    def _fits(sel: set[str]) -> bool:
        resident = sum(sizes.get(n, 0) for n in sel)
        room = int(budget_bytes) - resident
        if room < 0:
            return False
        for group in transfer_groups:
            rem = sum(sizes.get(str(n), 0) for n in group if str(n) not in sel)
            if rem > room:
                return False
        return True

    while selected and not _fits(selected):
        victim = min(selected, key=lambda n: (sizes.get(n, 0), n))
        selected.remove(victim)
    return selected


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
    if parameter_bytes is not None and parameter_bytes > 0 and parameter_bytes <= int(budget):
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
