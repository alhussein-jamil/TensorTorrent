"""Discrete-event schedule simulator for heterogeneous plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streamcompiler.cost_model.contention import concurrent_slowdown
from streamcompiler.ir.resource_graph import ResourceGraph
from streamcompiler.planner.maximal import ExecutionPlan


@dataclass
class SimulationResult:
    makespan_s: float
    peak_bytes: dict[str, int]
    timeline: list[dict[str, Any]]
    exposed_transfer_latency_s: float
    resource_busy_s: dict[str, float]


def simulate_plan(plan: ExecutionPlan, machine: ResourceGraph) -> SimulationResult:
    """Schedule placements onto independent resource timelines.

    - Device resource constraints always apply.
    - Explicit ``depends_on`` edges enforce data dependencies.
    - If ``depends_on`` is empty, regions may run concurrently on different devices.
    """
    resource_free_at: dict[str, float] = {name: 0.0 for name in machine.compute}
    resource_busy: dict[str, float] = {name: 0.0 for name in machine.compute}
    peak: dict[str, int] = {name: 0 for name in machine.memory}
    timeline: list[dict[str, Any]] = []
    region_end: dict[str, float] = {}
    exposed = 0.0
    makespan = 0.0

    # Default linear dependence only when every placement leaves depends_on empty
    # and strategy looks like a single-device chain; otherwise honor explicit edges.
    use_implicit_chain = all(not p.depends_on for p in plan.placements) and len(plan.devices_used) <= 1

    prev_id: str | None = None
    for placement in plan.placements:
        deps = list(placement.depends_on)
        if use_implicit_chain and prev_id is not None:
            deps.append(prev_id)
        elif (
            not placement.depends_on
            and prev_id is not None
            and len(plan.devices_used) > 1
            and placement.device == plan.placements[0].device
        ):
            # Keep per-device ordering stable without forcing cross-device serialization.
            pass

        dep_ready = 0.0
        for dep in deps:
            if dep in region_end:
                dep_ready = max(dep_ready, region_end[dep])
                if dep != placement.region_id:
                    # Cross-device dependency pays a small sync/transfer exposure.
                    producer = next((p for p in plan.placements if p.region_id == dep), None)
                    if producer is not None and producer.device != placement.device:
                        exposed += 0.0002
                        dep_ready += 0.0002

        start = max(resource_free_at.get(placement.device, 0.0), dep_ready)
        # Apply mild contention when multiple devices are busy at start time.
        active = sum(1 for t in resource_free_at.values() if t > start)
        factors = concurrent_slowdown(
            active_compute=max(1, active),
            active_transfers=1 if deps else 0,
            active_storage=0,
        )
        dur = float(placement.estimated_latency_s) * factors.compute
        end = start + dur
        resource_free_at[placement.device] = end
        resource_busy[placement.device] = resource_busy.get(placement.device, 0.0) + dur
        region_end[placement.region_id] = end
        timeline.append(
            {
                "region": placement.region_id,
                "device": placement.device,
                "backend": placement.backend_id,
                "dtype": placement.dtype,
                "start_s": start,
                "end_s": end,
            }
        )
        device = machine.compute.get(placement.device)
        if device is not None:
            for mem_name in device.memory_affinity:
                peak[mem_name] = peak.get(mem_name, 0) + 1_048_576
        prev_id = placement.region_id
        makespan = max(makespan, end)

    return SimulationResult(
        makespan_s=makespan,
        peak_bytes=peak,
        timeline=timeline,
        exposed_transfer_latency_s=exposed,
        resource_busy_s=resource_busy,
    )
