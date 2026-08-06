"""Joint heterogeneous placement search.

The planner explores concrete ``(device, backend, kernel, dtype)`` choices while
modeling the instruction critical path, transfer engines, and time-dependent
memory pressure.  It deliberately operates on backend-neutral candidates and
resource-graph links: vendor-specific rules remain behind backend discovery.

The search is a bounded beam search rather than an exponential exhaustive walk.
For each partial assignment it keeps enough state to make the important costs
non-local:

* compute resources serialize their own kernels,
* transfer paths serialize their own copies,
* dependency edges pay size-aware transfer costs,
* streamed state is charged only while its region executes,
* activations remain live until their last graph consumer is assigned,
* per-device capacity is checked before a state enters the beam.

This is not a MILP solver, but unlike the former greedy pass it jointly evaluates
placement and communication and can backtrack away from locally attractive but
globally poor assignments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tensortorrent.backends.base import KernelCandidate
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.ir.graph import HeterogeneousGraph
from tensortorrent.ir.resource_graph import ComputeClass, LinkClass, ResourceGraph, TransferLink

if TYPE_CHECKING:
    from tensortorrent.planner.maximal import Placement


_LINK_PRIORS: dict[LinkClass, tuple[float, float]] = {
    # (fixed latency seconds, payload bytes/second). Priors are used only when a
    # measured link coefficient is absent and are clearly surfaced in plan notes.
    LinkClass.CPU_LOCAL: (0.4e-6, 80e9),
    LinkClass.SHARED_MEMORY: (0.7e-6, 120e9),
    LinkClass.NUMA_INTERCONNECT: (1.5e-6, 35e9),
    LinkClass.NVLINK: (2.0e-6, 100e9),
    LinkClass.INFINITY_FABRIC: (2.5e-6, 80e9),
    LinkClass.CXL: (3.0e-6, 40e9),
    LinkClass.PCIE: (8.0e-6, 12e9),
    LinkClass.HOST_STAGED: (16.0e-6, 6e9),
    LinkClass.STORAGE: (35.0e-6, 2.5e9),
    LinkClass.NETWORK: (20.0e-6, 10e9),
    LinkClass.UNKNOWN: (15.0e-6, 8e9),
}


@dataclass(frozen=True)
class TransferEstimate:
    duration_s: float
    resource: str
    measured: bool
    host_staged: bool


@dataclass
class SearchState:
    placements: tuple[Placement, ...] = ()
    finish: dict[str, float] = field(default_factory=dict)
    device_free: dict[str, float] = field(default_factory=dict)
    link_free: dict[str, float] = field(default_factory=dict)
    device_busy: dict[str, float] = field(default_factory=dict)
    link_busy: dict[str, float] = field(default_factory=dict)
    live_activation_bytes: dict[str, int] = field(default_factory=dict)
    peak_bytes: dict[str, int] = field(default_factory=dict)
    output_device: dict[str, str] = field(default_factory=dict)
    output_bytes: dict[str, int] = field(default_factory=dict)
    remaining_consumers: dict[str, int] = field(default_factory=dict)
    transfer_bytes: int = 0
    transfer_latency_s: float = 0.0
    unmeasured_transfer_count: int = 0
    host_staged_transfer_count: int = 0

    @property
    def makespan_s(self) -> float:
        return max(self.finish.values(), default=0.0)

    @property
    def initiation_interval_s(self) -> float:
        """Steady-state lower bound for one request.

        At saturation a resource must provide all service demanded by one graph.
        The busiest compute or transfer resource therefore bounds the initiation
        interval, independently of single-request dependency latency.
        """
        return max(
            max(self.device_busy.values(), default=0.0),
            max(self.link_busy.values(), default=0.0),
            1e-12,
        )


@dataclass(frozen=True)
class SearchResult:
    placements: tuple[Placement, ...]
    latency_s: float
    throughput_per_s: float
    peak_bytes: dict[str, int]
    transfer_bytes: int
    transfer_latency_s: float
    unmeasured_transfer_count: int
    host_staged_transfer_count: int
    states_expanded: int
    states_pruned: int
    beam_width: int
    local_improvements: int


def _device_memory_name(machine: ResourceGraph, device_name: str) -> str:
    device = machine.compute.get(device_name)
    if device is None:
        return device_name
    for name in device.memory_affinity:
        if name in machine.memory:
            return name
    return device_name


def device_capacity_bytes(
    machine: ResourceGraph,
    device_name: str,
    *,
    vram_budget_bytes: int | None,
) -> int:
    device = machine.compute.get(device_name)
    if device is None:
        return 0
    total = sum(
        max(0, int(machine.memory[name].allocatable_bytes)) for name in device.memory_affinity if name in machine.memory
    )
    if (
        device.compute_class
        in {
            ComputeClass.DISCRETE_GPU,
            ComputeClass.INTEGRATED_GPU,
            ComputeClass.ACCELERATOR,
        }
        and vram_budget_bytes is not None
    ):
        return min(total, vram_budget_bytes) if total > 0 else vram_budget_bytes
    return total


def _link_duration(link: TransferLink, nbytes: int) -> tuple[float, bool]:
    prior_latency, prior_bandwidth = _LINK_PRIORS.get(link.link_class, _LINK_PRIORS[LinkClass.UNKNOWN])
    latency = float(link.latency_s) if link.latency_s is not None and link.latency_s >= 0 else prior_latency
    bandwidth = float(link.bytes_per_s) if link.bytes_per_s is not None and link.bytes_per_s > 0 else prior_bandwidth
    contention = max(1.0, float(link.contention_factor or 1.0))
    duration = (latency + max(0, nbytes) / max(1.0, bandwidth)) * contention
    measured = bool(link.measured and link.latency_s is not None and link.bytes_per_s is not None)
    return max(0.0, duration), measured


def estimate_transfer(
    machine: ResourceGraph,
    source_device: str,
    destination_device: str,
    nbytes: int,
    *,
    allow_host_staged: bool,
) -> TransferEstimate | None:
    if source_device == destination_device or nbytes <= 0:
        return TransferEstimate(0.0, f"local::{source_device}", True, False)

    source_memory = _device_memory_name(machine, source_device)
    destination_memory = _device_memory_name(machine, destination_device)
    direct = machine.link_between(source_memory, destination_memory)
    if direct is not None:
        duration, measured = _link_duration(direct, nbytes)
        host_staged = direct.link_class == LinkClass.HOST_STAGED
        if host_staged and not allow_host_staged:
            return None
        return TransferEstimate(duration, direct.id.name, measured, host_staged)

    # Discovery installs explicit staged pseudo-links on many machines. Their id
    # is not source->destination, so locate them by endpoint attributes.
    staged = next(
        (
            link
            for link in machine.links.values()
            if link.link_class == LinkClass.HOST_STAGED
            and link.source == source_memory
            and link.destination == destination_memory
        ),
        None,
    )
    if staged is not None:
        if not allow_host_staged:
            return None
        duration, measured = _link_duration(staged, nbytes)
        return TransferEstimate(duration, staged.id.name, measured, True)

    if not allow_host_staged:
        return None

    # Last-resort explicit host-staged prior. This is intentionally expensive and
    # marked unmeasured so a direct measured path always wins when available.
    latency, bandwidth = _LINK_PRIORS[LinkClass.HOST_STAGED]
    return TransferEstimate(
        duration_s=latency + max(0, nbytes) / bandwidth,
        resource=f"host_staged::{source_memory}->{destination_memory}",
        measured=False,
        host_staged=True,
    )


def dependency_edge_bytes(graph: HeterogeneousGraph) -> dict[tuple[str, str], int]:
    """Exact bytes transferred on each producer->consumer region edge."""
    producer: dict[str, str] = {}
    for region in graph.compute_regions():
        for output in region.outputs:
            producer[output] = region.name

    edges: dict[tuple[str, str], int] = {}
    for region in graph.compute_regions():
        for input_name in region.inputs:
            source = producer.get(input_name)
            if source is None or source == region.name:
                continue
            meta = graph.tensors.get(input_name)
            edges[(source, region.name)] = edges.get((source, region.name), 0) + max(
                0, int(getattr(meta, "size_bytes", 0) or 0)
            )
    return edges


def _consumer_counts(graph: HeterogeneousGraph) -> dict[str, int]:
    counts = {region.name: 0 for region in graph.compute_regions()}
    for region in graph.compute_regions():
        for dep in tuple(str(x) for x in region.attributes.get("depends_on", ())):
            if dep in counts:
                counts[dep] += 1
    # Graph outputs must remain live after the last region. Give their producer a
    # sentinel consumer so the search does not release them prematurely.
    producer: dict[str, str] = {}
    for region in graph.compute_regions():
        for output in region.outputs:
            producer[output] = region.name
    for output in graph.outputs:
        source = producer.get(output)
        if source is not None:
            counts[source] = counts.get(source, 0) + 1
    return counts


def _candidate_pool(
    candidates: list[KernelCandidate],
    *,
    allowed_devices: set[str],
    per_device: int,
) -> list[KernelCandidate]:
    by_device: dict[str, list[KernelCandidate]] = {}
    for candidate in candidates:
        if candidate.device not in allowed_devices:
            continue
        by_device.setdefault(candidate.device, []).append(candidate)
    selected: list[KernelCandidate] = []
    for device in sorted(by_device):
        ranked = sorted(
            by_device[device],
            key=lambda candidate: (
                float(candidate.estimated_latency_s or math.inf),
                int(candidate.workspace_bytes),
                candidate.dtype,
                candidate.kernel_id,
            ),
        )
        selected.extend(ranked[: max(1, per_device)])
    return selected


def _memory_pressure(state: SearchState, capacities: dict[str, int]) -> float:
    pressure = 0.0
    for device, peak in state.peak_bytes.items():
        capacity = capacities.get(device, 0)
        if capacity > 0:
            pressure += peak / capacity
        elif peak > 0:
            pressure += 1.0
    return pressure


def _state_score(state: SearchState, config: CompileConfig, capacities: dict[str, int]) -> float:
    latency = state.makespan_s
    cycle = state.initiation_interval_s
    peak = float(sum(state.peak_bytes.values()))
    pressure = _memory_pressure(state, capacities)

    if config.objective == Objective.LATENCY:
        return latency + 1e-6 * cycle + 1e-9 * pressure
    if config.objective == Objective.THROUGHPUT:
        return cycle + 1e-3 * latency + 1e-9 * pressure
    if config.objective == Objective.MEMORY:
        return peak + 1e-3 * pressure + 1e-9 * latency
    if config.objective == Objective.BALANCED:
        # Dimensionless pressure keeps memory relevant without letting byte units
        # dwarf seconds. Latency and cycle share the same time unit.
        return latency + cycle + 0.05 * pressure

    weights = config.objective_weights
    return (
        weights.get("latency", 0.0) * latency
        + weights.get("throughput", 0.0) * cycle
        + weights.get("memory", 0.0) * pressure
    )


def _state_signature(state: SearchState) -> tuple[Any, ...]:
    """Dominance key that preserves future-relevant resource state.

    Placement history alone is insufficient: two states can place the same
    regions on the same devices while selecting kernels with different finish
    times or copy-path occupancy. Collapsing those states can discard the only
    assignment that leaves a critical transfer engine free for a later branch.
    Rounded clocks keep the key deterministic without relying on exact float
    bit patterns.
    """
    devices = tuple(placement.device for placement in state.placements)
    live = tuple(sorted(state.live_activation_bytes.items()))
    device_free = tuple(sorted((name, round(value, 12)) for name, value in state.device_free.items()))
    link_free = tuple(sorted((name, round(value, 12)) for name, value in state.link_free.items()))
    return devices, live, device_free, link_free


def _build_prefix_states(
    assignment: list[KernelCandidate],
    *,
    order: list[str],
    dependencies: dict[str, tuple[str, ...]],
    byte_counts: dict[str, tuple[int, int]],
    edge_bytes: dict[tuple[str, str], int],
    initial_consumers: dict[str, int],
    machine: ResourceGraph,
    capacities: dict[str, int],
    allow_host_staged: bool,
) -> list[SearchState]:
    """``prefix_states[i]`` is the state before placing ``order[i]``; last entry is final."""
    states: list[SearchState] = [SearchState(remaining_consumers=dict(initial_consumers))]
    for region_id, candidate in zip(order, assignment, strict=True):
        next_state = _extend_state(
            states[-1],
            region_id=region_id,
            candidate=candidate,
            dependencies=dependencies.get(region_id, ()),
            output_bytes=byte_counts.get(region_id, (0, 0))[0],
            state_bytes=byte_counts.get(region_id, (0, 0))[1],
            edge_bytes=edge_bytes,
            machine=machine,
            capacities=capacities,
            allow_host_staged=allow_host_staged,
        )
        if next_state is None:  # pragma: no cover - assignment came from a feasible beam
            return states
        states.append(next_state)
    return states


def _incremental_evaluate_assignment(
    assignment: list[KernelCandidate],
    *,
    change_index: int,
    alternate: KernelCandidate,
    prefix_states: list[SearchState],
    order: list[str],
    dependencies: dict[str, tuple[str, ...]],
    byte_counts: dict[str, tuple[int, int]],
    edge_bytes: dict[tuple[str, str], int],
    machine: ResourceGraph,
    capacities: dict[str, int],
    allow_host_staged: bool,
) -> SearchState | None:
    """Replay an assignment with one alternate, reusing prefix state before ``change_index``."""
    state: SearchState = prefix_states[change_index]
    region_id = order[change_index]
    output_bytes, state_bytes = byte_counts.get(region_id, (0, 0))
    extended = _extend_state(
        state,
        region_id=region_id,
        candidate=alternate,
        dependencies=dependencies.get(region_id, ()),
        output_bytes=output_bytes,
        state_bytes=state_bytes,
        edge_bytes=edge_bytes,
        machine=machine,
        capacities=capacities,
        allow_host_staged=allow_host_staged,
    )
    if extended is None:
        return None
    state = extended
    for j in range(change_index + 1, len(order)):
        region_id = order[j]
        candidate = assignment[j]
        output_bytes, state_bytes = byte_counts.get(region_id, (0, 0))
        extended = _extend_state(
            state,
            region_id=region_id,
            candidate=candidate,
            dependencies=dependencies.get(region_id, ()),
            output_bytes=output_bytes,
            state_bytes=state_bytes,
            edge_bytes=edge_bytes,
            machine=machine,
            capacities=capacities,
            allow_host_staged=allow_host_staged,
        )
        if extended is None:
            return None
        state = extended
    return state


def _evaluate_assignment(
    assignment: list[KernelCandidate],
    *,
    order: list[str],
    dependencies: dict[str, tuple[str, ...]],
    byte_counts: dict[str, tuple[int, int]],
    edge_bytes: dict[tuple[str, str], int],
    initial_consumers: dict[str, int],
    machine: ResourceGraph,
    capacities: dict[str, int],
    allow_host_staged: bool,
) -> SearchState | None:
    """Replay a complete concrete assignment through the same cost model."""
    state = SearchState(remaining_consumers=dict(initial_consumers))
    for region_id, candidate in zip(order, assignment, strict=True):
        output_bytes, state_bytes = byte_counts.get(region_id, (0, 0))
        next_state = _extend_state(
            state,
            region_id=region_id,
            candidate=candidate,
            dependencies=dependencies.get(region_id, ()),
            output_bytes=output_bytes,
            state_bytes=state_bytes,
            edge_bytes=edge_bytes,
            machine=machine,
            capacities=capacities,
            allow_host_staged=allow_host_staged,
        )
        if next_state is None:
            return None
        state = next_state
    return state


def _extend_state(
    state: SearchState,
    *,
    region_id: str,
    candidate: KernelCandidate,
    dependencies: tuple[str, ...],
    output_bytes: int,
    state_bytes: int,
    edge_bytes: dict[tuple[str, str], int],
    machine: ResourceGraph,
    capacities: dict[str, int],
    allow_host_staged: bool,
) -> SearchState | None:
    from tensortorrent.planner.maximal import Placement

    device = candidate.device
    finish = dict(state.finish)
    device_free = dict(state.device_free)
    link_free = dict(state.link_free)
    device_busy = dict(state.device_busy)
    link_busy = dict(state.link_busy)
    live = dict(state.live_activation_bytes)
    peak = dict(state.peak_bytes)
    output_device = dict(state.output_device)
    output_size = dict(state.output_bytes)
    remaining = dict(state.remaining_consumers)

    ready = 0.0
    incoming_remote = 0
    transfer_bytes = state.transfer_bytes
    transfer_latency = state.transfer_latency_s
    unmeasured = state.unmeasured_transfer_count
    host_staged = state.host_staged_transfer_count

    for dep in dependencies:
        dep_finish = finish.get(dep)
        dep_device = output_device.get(dep)
        if dep_finish is None or dep_device is None:
            return None
        nbytes = max(0, int(edge_bytes.get((dep, region_id), output_size.get(dep, 0))))
        if dep_device == device:
            ready = max(ready, dep_finish)
            continue
        estimate = estimate_transfer(
            machine,
            dep_device,
            device,
            nbytes,
            allow_host_staged=allow_host_staged,
        )
        if estimate is None:
            return None
        start = max(dep_finish, link_free.get(estimate.resource, 0.0))
        end = start + estimate.duration_s
        link_free[estimate.resource] = end
        link_busy[estimate.resource] = link_busy.get(estimate.resource, 0.0) + estimate.duration_s
        ready = max(ready, end)
        incoming_remote += nbytes
        transfer_bytes += nbytes
        transfer_latency += estimate.duration_s
        if not estimate.measured:
            unmeasured += 1
        if estimate.host_staged:
            host_staged += 1

    latency = max(1e-12, float(candidate.estimated_latency_s or 0.0))
    start = max(ready, device_free.get(device, 0.0))
    end = start + latency

    workspace = max(0, int(candidate.workspace_bytes or 0))
    current_live = max(0, int(live.get(device, 0)))
    instantaneous = current_live + incoming_remote + max(0, state_bytes) + max(0, output_bytes) + workspace
    capacity = capacities.get(device, 0)
    if capacity > 0 and instantaneous > capacity:
        return None

    peak[device] = max(peak.get(device, 0), instantaneous)
    live[device] = current_live + max(0, output_bytes)
    output_device[region_id] = device
    output_size[region_id] = max(0, output_bytes)
    if remaining.get(region_id, 0) <= 0:
        # Dead output: charge it during the kernel but do not retain it for later regions.
        live[device] = max(0, live.get(device, 0) - max(0, output_bytes))

    # Once this consumer is assigned, producer output lifetime can end after the
    # corresponding transfer/compute frontier. The byte accounting is a peak
    # estimate, so releasing here is correct for subsequent regions in topo order.
    for dep in dependencies:
        if dep not in remaining:
            continue
        remaining[dep] -= 1
        if remaining[dep] <= 0:
            source_device = output_device.get(dep)
            if source_device is not None:
                live[source_device] = max(0, live.get(source_device, 0) - output_size.get(dep, 0))

    finish[region_id] = end
    device_free[device] = end
    device_busy[device] = device_busy.get(device, 0.0) + latency

    placement = Placement(
        region_id=region_id,
        device=device,
        backend_id=candidate.backend_id,
        dtype=candidate.dtype,
        kernel_id=candidate.kernel_id,
        estimated_latency_s=latency,
        depends_on=dependencies,
        measured=bool(candidate.attributes.get("measured", False)),
        output_bytes=max(0, output_bytes),
        state_bytes=max(0, state_bytes),
        workspace_bytes=workspace,
    )
    return SearchState(
        placements=(*state.placements, placement),
        finish=finish,
        device_free=device_free,
        link_free=link_free,
        device_busy=device_busy,
        link_busy=link_busy,
        live_activation_bytes=live,
        peak_bytes=peak,
        output_device=output_device,
        output_bytes=output_size,
        remaining_consumers=remaining,
        transfer_bytes=transfer_bytes,
        transfer_latency_s=transfer_latency,
        unmeasured_transfer_count=unmeasured,
        host_staged_transfer_count=host_staged,
    )


def search_placements(
    graph: HeterogeneousGraph,
    machine: ResourceGraph,
    region_candidates: dict[str, list[KernelCandidate]],
    allowed_devices: set[str],
    byte_counts: dict[str, tuple[int, int]],
    config: CompileConfig,
) -> SearchResult | None:
    regions = graph.compute_regions()
    if not regions:
        return None
    declared_order = [region.name for region in regions]
    known = set(declared_order)
    dependencies = {
        region.name: tuple(str(dep) for dep in region.attributes.get("depends_on", ()) if str(dep) in known)
        for region in regions
    }
    # Stable Kahn topological order. Portable IR is normally already ordered, but
    # planning must not silently treat a shuffled independent list as executable.
    remaining = set(declared_order)
    order: list[str] = []
    while remaining:
        ready = [
            region_id
            for region_id in declared_order
            if region_id in remaining and all(dep not in remaining for dep in dependencies[region_id])
        ]
        if not ready:
            return None
        for region_id in ready:
            order.append(region_id)
            remaining.remove(region_id)
    edge_bytes = dependency_edge_bytes(graph)
    initial_consumers = _consumer_counts(graph)
    capacities = {
        device: device_capacity_bytes(machine, device, vram_budget_bytes=config.vram_budget_bytes)
        for device in allowed_devices
    }

    pools: dict[str, list[KernelCandidate]] = {}
    for region_id in order:
        pool = _candidate_pool(
            region_candidates.get(region_id, []),
            allowed_devices=allowed_devices,
            per_device=max(1, int(config.planner_candidates_per_device)),
        )
        if not pool:
            return None
        pools[region_id] = pool

    beam: list[SearchState] = [SearchState(remaining_consumers=dict(initial_consumers))]
    states_expanded = 0
    states_pruned = 0
    beam_width = max(1, int(config.planner_beam_width))

    for region_id in order:
        pool = pools[region_id]
        output_bytes, state_bytes = byte_counts.get(region_id, (0, 0))
        next_states: list[SearchState] = []
        for state in beam:
            for candidate in pool:
                states_expanded += 1
                extended = _extend_state(
                    state,
                    region_id=region_id,
                    candidate=candidate,
                    dependencies=dependencies.get(region_id, ()),
                    output_bytes=output_bytes,
                    state_bytes=state_bytes,
                    edge_bytes=edge_bytes,
                    machine=machine,
                    capacities=capacities,
                    allow_host_staged=config.allow_host_staged_transfers,
                )
                if extended is None:
                    states_pruned += 1
                    continue
                next_states.append(extended)
        if not next_states:
            return None

        # Dominance pruning: for the same placement/live-byte signature retain the
        # state with the best objective score. This preserves useful diversity while
        # preventing the beam from filling with kernel variants that lead to the
        # same resource state.
        dominant: dict[tuple[Any, ...], SearchState] = {}
        for candidate_state in next_states:
            signature = _state_signature(candidate_state)
            previous = dominant.get(signature)
            if previous is None or _state_score(candidate_state, config, capacities) < _state_score(
                previous, config, capacities
            ):
                dominant[signature] = candidate_state
        ranked = sorted(dominant.values(), key=lambda state: _state_score(state, config, capacities))
        if len(ranked) > beam_width:
            states_pruned += len(ranked) - beam_width
        beam = ranked[:beam_width]

    best = min(beam, key=lambda state: _state_score(state, config, capacities))

    # Bounded coordinate descent over the full concrete candidate set. Unlike
    # the removed device-only rebalance, every mutation is replayed through
    # backend/kernel compatibility, transfers, lifetimes, and capacity checks.
    assignment: list[KernelCandidate] = []
    for region_id, placement in zip(order, best.placements, strict=True):
        exact = next(
            (
                candidate
                for candidate in pools[region_id]
                if candidate.device == placement.device
                and candidate.backend_id == placement.backend_id
                and candidate.kernel_id == placement.kernel_id
                and candidate.dtype == placement.dtype
            ),
            None,
        )
        if exact is None:  # pragma: no cover - placements originate from these pools
            return None
        assignment.append(exact)

    prefix_states = _build_prefix_states(
        assignment,
        order=order,
        dependencies=dependencies,
        byte_counts=byte_counts,
        edge_bytes=edge_bytes,
        initial_consumers=initial_consumers,
        machine=machine,
        capacities=capacities,
        allow_host_staged=config.allow_host_staged_transfers,
    )

    local_improvements = 0
    for _ in range(max(0, int(config.planner_local_search_iters))):
        improved_this_pass = False
        for index, region_id in enumerate(order):
            incumbent_candidate = assignment[index]
            incumbent_state = best
            incumbent_score = _state_score(best, config, capacities)
            for alternate in pools[region_id]:
                if alternate == incumbent_candidate:
                    continue
                states_expanded += len(order) - index
                evaluated = _incremental_evaluate_assignment(
                    assignment,
                    change_index=index,
                    alternate=alternate,
                    prefix_states=prefix_states,
                    order=order,
                    dependencies=dependencies,
                    byte_counts=byte_counts,
                    edge_bytes=edge_bytes,
                    machine=machine,
                    capacities=capacities,
                    allow_host_staged=config.allow_host_staged_transfers,
                )
                if evaluated is None:
                    states_pruned += 1
                    continue
                score = _state_score(evaluated, config, capacities)
                if score + 1e-15 < incumbent_score:
                    incumbent_candidate = alternate
                    incumbent_state = evaluated
                    incumbent_score = score
            if incumbent_candidate != assignment[index]:
                assignment[index] = incumbent_candidate
                best = incumbent_state
                local_improvements += 1
                improved_this_pass = True
                prefix_states = _build_prefix_states(
                    assignment,
                    order=order,
                    dependencies=dependencies,
                    byte_counts=byte_counts,
                    edge_bytes=edge_bytes,
                    initial_consumers=initial_consumers,
                    machine=machine,
                    capacities=capacities,
                    allow_host_staged=config.allow_host_staged_transfers,
                )
        if not improved_this_pass:
            break
    saturation_throughput = 1.0 / max(best.initiation_interval_s, 1e-12)
    closed_queue_throughput = max(1, int(config.target_inflight_requests)) / max(best.makespan_s, 1e-12)
    throughput = min(saturation_throughput, closed_queue_throughput)
    return SearchResult(
        placements=best.placements,
        latency_s=best.makespan_s,
        throughput_per_s=throughput,
        peak_bytes=dict(best.peak_bytes),
        transfer_bytes=best.transfer_bytes,
        transfer_latency_s=best.transfer_latency_s,
        unmeasured_transfer_count=best.unmeasured_transfer_count,
        host_staged_transfer_count=best.host_staged_transfer_count,
        states_expanded=states_expanded,
        states_pruned=states_pruned,
        beam_width=beam_width,
        local_improvements=local_improvements,
    )
