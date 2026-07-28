"""Discrete-event schedule simulator for heterogeneous plans.

Peak memory and cross-device transfer costs come from placement byte counts and
the machine's transfer links. The simulator remains analytic (it does not execute
kernels), but it no longer invents a flat 1 MiB per region or a fixed 200 µs hop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streamcompiler.cost_model.contention import concurrent_slowdown
from streamcompiler.cost_model.transfer import TransferModel, transfer_time
from streamcompiler.ir.resource_graph import ResourceGraph, TransferLink
from streamcompiler.planner.maximal import ExecutionPlan, Placement


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
    - Cross-device edges pay a transfer cost derived from the producer's output
      bytes and the best available link between the two devices' memories.
    """
    resource_free_at: dict[str, float] = {name: 0.0 for name in machine.compute}
    resource_busy: dict[str, float] = {name: 0.0 for name in machine.compute}
    peak: dict[str, int] = {name: 0 for name in machine.memory}
    resident: dict[str, int] = {name: 0 for name in machine.memory}
    timeline: list[dict[str, Any]] = []
    region_end: dict[str, float] = {}
    by_id = {p.region_id: p for p in plan.placements}
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

        dep_ready = 0.0
        for dep in deps:
            if dep not in region_end:
                continue
            ready = region_end[dep]
            producer = by_id.get(dep)
            if producer is not None and producer.device != placement.device:
                hop = _transfer_latency_s(machine, producer, placement)
                exposed += hop
                ready += hop
            dep_ready = max(dep_ready, ready)

        start = max(resource_free_at.get(placement.device, 0.0), dep_ready)
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
                "working_set_bytes": placement.working_set_bytes,
            }
        )
        _account_peak(machine, placement, resident, peak)
        prev_id = placement.region_id
        makespan = max(makespan, end)

    return SimulationResult(
        makespan_s=makespan,
        peak_bytes=peak,
        timeline=timeline,
        exposed_transfer_latency_s=exposed,
        resource_busy_s=resource_busy,
    )


def _account_peak(
    machine: ResourceGraph,
    placement: Placement,
    resident: dict[str, int],
    peak: dict[str, int],
) -> None:
    """Add this region's working set to its device memories and track the peak."""
    device = machine.compute.get(placement.device)
    if device is None:
        return
    working = max(0, placement.working_set_bytes)
    for mem_name in device.memory_affinity:
        if mem_name not in resident:
            continue
        resident[mem_name] += working
        peak[mem_name] = max(peak.get(mem_name, 0), resident[mem_name])


def _transfer_latency_s(machine: ResourceGraph, producer: Placement, consumer: Placement) -> float:
    """Time to move the producer's outputs onto the consumer's device."""
    nbytes = max(0, producer.output_bytes)
    if nbytes == 0:
        # No sized outputs recorded; keep a small sync prior so the edge is not free.
        return 1e-6
    link = _best_link(machine, producer.device, consumer.device)
    if link is None:
        return transfer_time(None, producer.device, consumer.device, nbytes)
    model = TransferModel(
        source=link.source,
        destination=link.destination,
        alpha_s=float(link.latency_s or 0.0),
        beta_bytes_per_s=link.bytes_per_s,
        contention_factor=float(link.contention_factor),
        measured=bool(link.measured),
    )
    return transfer_time(model, link.source, link.destination, nbytes)


def _best_link(machine: ResourceGraph, source_device: str, destination_device: str) -> TransferLink | None:
    """Pick the fastest memory-to-memory link between two compute devices.

    Prefers measured peer-to-peer links, then any direct link between the devices'
    memories, then a host-staged path. Returns ``None`` when nothing is known.
    """
    src = machine.compute.get(source_device)
    dst = machine.compute.get(destination_device)
    if src is None or dst is None:
        return None
    candidates: list[TransferLink] = []
    for s_mem in src.memory_affinity:
        for d_mem in dst.memory_affinity:
            link = machine.link_between(s_mem, d_mem)
            if link is not None:
                candidates.append(link)
            reverse = machine.link_between(d_mem, s_mem)
            if reverse is not None and reverse.bidirectional:
                candidates.append(reverse)
    if not candidates:
        # Fall back to any host-staged link touching either device's memories.
        for link in machine.links.values():
            if link.link_class.value != "host_staged":
                continue
            if link.source in src.memory_affinity or link.destination in dst.memory_affinity:
                candidates.append(link)
    if not candidates:
        return None

    def rank(link: TransferLink) -> tuple[int, float]:
        # Higher is better: measured P2P first, then by bandwidth.
        score = (2 if link.peer_to_peer else 0) + (1 if link.measured else 0)
        bandwidth = float(link.bytes_per_s or 0.0)
        return (score, bandwidth)

    return max(candidates, key=rank)
