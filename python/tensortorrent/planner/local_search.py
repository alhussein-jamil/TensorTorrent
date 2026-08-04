"""Post-search refinements that preserve backend/device correctness."""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

from tensortorrent.planner.maximal import ExecutionPlan


def refine_prefetch_distance(
    plan: ExecutionPlan,
    distance: int = 1,
    *,
    adaptive: bool = False,
    ram_budget_bytes: int | None = None,
    storage_bytes_per_s: float | None = None,
    max_distance: int = 8,
) -> ExecutionPlan:
    """Choose a budget-feasible prefetch distance from compute/I/O overlap.

    The chosen depth is the number of average region compute windows required to
    hide the largest state read, bounded by the number of state working sets the
    host RAM budget can hold.  No latency improvement is fabricated: the schedule
    simulator and runtime telemetry remain the sources of predicted/measured gain.
    """
    requested = max(0, int(distance))
    chosen = requested
    reason = "configured"

    state_sizes = [max(0, int(placement.state_bytes)) for placement in plan.placements if placement.state_bytes > 0]
    compute_times = [
        max(1e-9, float(placement.estimated_latency_s))
        for placement in plan.placements
        if placement.estimated_latency_s > 0
    ]

    if adaptive and state_sizes and compute_times and ram_budget_bytes is not None:
        largest_state = max(state_sizes)
        slots = max(1, int(ram_budget_bytes) // max(1, largest_state))
        max_fit = max(0, slots - 1)  # one slot is occupied by the current region
        bandwidth = max(1.0, float(storage_bytes_per_s or 2.5e9))
        io_window = largest_state / bandwidth
        compute_window = statistics.median(compute_times)
        hide_depth = max(1, int(math.ceil(io_window / max(compute_window, 1e-9))))
        chosen = min(max(0, int(max_distance)), max_fit, max(requested, hide_depth))
        reason = (
            f"adaptive largest_state={largest_state} ram_slots={slots} "
            f"io_window_s={io_window:.6g} median_compute_s={compute_window:.6g}"
        )
    elif adaptive and ram_budget_bytes is not None and not state_sizes:
        chosen = 0
        reason = "adaptive no_streamed_state"

    notes = [note for note in plan.notes if not note.startswith("prefetch_distance=")]
    notes.append(f"prefetch_distance={chosen}")
    notes.append(f"prefetch_rationale={reason}")
    return replace(
        plan,
        placements=[replace(placement) for placement in plan.placements],
        decisions=[replace(decision) for decision in plan.decisions],
        predicted_peak_bytes=dict(plan.predicted_peak_bytes),
        search_statistics=dict(plan.search_statistics),
        prefetch_distance=chosen,
        notes=notes,
    )


def rebalance_partitions(plan: ExecutionPlan) -> ExecutionPlan:
    """Compatibility no-op: joint search already performs valid rebalancing.

    The previous implementation changed only ``Placement.device`` while retaining
    the old backend/kernel/dtype, which could create impossible CPU/CUDA or
    CUDA/ROCm combinations.  Rebalancing now belongs inside candidate-aware joint
    search; this function only clones the immutable published plan.
    """
    notes = [note for note in plan.notes if note != "local_search:unsafe_device_only_rebalance_removed"]
    notes.append("local_search:joint_planner_owns_rebalancing")
    return replace(
        plan,
        placements=[replace(placement) for placement in plan.placements],
        decisions=[replace(decision) for decision in plan.decisions],
        predicted_peak_bytes=dict(plan.predicted_peak_bytes),
        search_statistics=dict(plan.search_statistics),
        devices_used=tuple(sorted({placement.device for placement in plan.placements})),
        notes=notes,
    )
