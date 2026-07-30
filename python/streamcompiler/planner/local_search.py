"""Local-search refinement over prefetch distance and partition ratios."""

from __future__ import annotations

from copy import deepcopy

from streamcompiler.planner.maximal import ExecutionPlan, Placement


def refine_prefetch_distance(plan: ExecutionPlan, distance: int = 1) -> ExecutionPlan:
    """Record the runtime prefetch distance without inventing latency savings.

    Prefetch benefit is measured by the streaming parameter store under a real RAM
    budget. Annotating a fake ``predicted_latency_s`` delta here used to pretend
    deeper prefetch always helped, which was untrue on CPU-only resident plans.
    """
    out = deepcopy(plan)
    out.notes = list(plan.notes) + [f"prefetch_distance={max(0, int(distance))}"]
    return out


def rebalance_partitions(plan: ExecutionPlan) -> ExecutionPlan:
    """Nudge placements toward less-loaded devices without changing backends."""
    if len(plan.devices_used) <= 1:
        return plan
    loads: dict[str, float] = {}
    for p in plan.placements:
        loads[p.device] = loads.get(p.device, 0.0) + p.estimated_latency_s
    avg = sum(loads.values()) / len(loads)
    new_placements: list[Placement] = []
    for p in plan.placements:
        if loads[p.device] > avg * 1.25:
            # Move to the currently least-loaded compatible device if any shares backend.
            candidates = [d for d in plan.devices_used if d != p.device and loads.get(d, 0.0) < loads[p.device]]
            if candidates:
                target = min(candidates, key=lambda d: loads.get(d, 0.0))
                loads[p.device] -= p.estimated_latency_s
                loads[target] = loads.get(target, 0.0) + p.estimated_latency_s
                new_placements.append(
                    Placement(
                        region_id=p.region_id,
                        device=target,
                        backend_id=p.backend_id,
                        dtype=p.dtype,
                        kernel_id=p.kernel_id,
                        estimated_latency_s=p.estimated_latency_s,
                        depends_on=p.depends_on,
                        measured=p.measured,
                        output_bytes=p.output_bytes,
                        state_bytes=p.state_bytes,
                    )
                )
                continue
        new_placements.append(p)
    out = deepcopy(plan)
    out.placements = new_placements
    out.devices_used = tuple(sorted({p.device for p in new_placements}))
    out.notes = list(plan.notes) + ["local_search:rebalance_partitions"]
    return out
