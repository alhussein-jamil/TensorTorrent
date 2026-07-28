"""Discrete-event schedule simulator for heterogeneous plans.

Analytic only: kernels are not executed. Makespan, transfer exposure, peak
memory, and contention come from placement byte counts, explicit dependencies,
and the machine's transfer links. Tensor residency follows producer/consumer
lifetimes: outputs allocate at region end and release after the last consumer;
parameter/state bytes stay resident for each region's ``[start, end]`` interval.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from streamcompiler.cost_model.contention import concurrent_slowdown
from streamcompiler.cost_model.transfer import TransferModel, transfer_time
from streamcompiler.ir.resource_graph import ResourceGraph, TransferLink
from streamcompiler.planner.maximal import ExecutionPlan, Placement

# Imported lazily in simulate_schedule to avoid a runtime↔simulator cycle at module load.


@dataclass
class SimulationResult:
    makespan_s: float
    peak_bytes: dict[str, int]
    timeline: list[dict[str, Any]]
    exposed_transfer_latency_s: float
    resource_busy_s: dict[str, float]
    transfer_events: list[dict[str, Any]] = field(default_factory=list)
    release_events: list[dict[str, Any]] = field(default_factory=list)
    simulated: bool = True
    """Always True: this path never executes kernels."""
    critical_path: list[str] = field(default_factory=list)
    bytes_read: int = 0
    bytes_transferred: int = 0
    instruction_count: int = 0
    resource_utilization: dict[str, float] = field(default_factory=dict)
    """Busy time / makespan per compute resource (0..1+ under oversub)."""


def simulate_plan(plan: ExecutionPlan, machine: ResourceGraph) -> SimulationResult:
    """Schedule placements onto independent resource timelines.

    - Device resource constraints always apply.
    - Explicit ``depends_on`` edges enforce data dependencies.
    - Cross-device edges pay a transfer cost and briefly occupy both memories.
    - Activation bytes allocate when a region finishes and free after the last
      dependent region that consumes them finishes (liveness).
    - Parameter/state bytes stay resident for the region interval ``[start, end]``
      so overlapping peers that share a memory pool contribute to peak together.
    - Prefetch distance from plan notes only annotates timeline events; it does
      not invent measured I/O overlap.
    """
    resource_free_at: dict[str, float] = {name: 0.0 for name in machine.compute}
    link_free_at: dict[str, float] = {name: 0.0 for name in machine.links}
    resource_busy: dict[str, float] = {name: 0.0 for name in machine.compute}
    peak: dict[str, int] = {name: 0 for name in machine.memory}
    resident: dict[str, int] = {name: 0 for name in machine.memory}
    # Live output tensors: region_id -> (memory_name, nbytes, remaining_consumers)
    live_outputs: dict[str, tuple[str, int, int]] = {}
    consumer_count = _consumer_counts(plan)
    timeline: list[dict[str, Any]] = []
    transfer_events: list[dict[str, Any]] = []
    release_events: list[dict[str, Any]] = []
    region_end: dict[str, float] = {}
    by_id = {p.region_id: p for p in plan.placements}
    # Dest-side copies of transferred activations: consumer_region -> (mem, nbytes)
    dest_copies: dict[str, list[tuple[str, int]]] = defaultdict(list)
    # State pinned for [start, end]: (end_s, mem, nbytes, region_id)
    state_leases: list[tuple[float, str, int, str]] = []
    exposed = 0.0
    makespan = 0.0
    prefetch_distance = _prefetch_distance(plan)

    use_implicit_chain = all(not p.depends_on for p in plan.placements) and len(plan.devices_used) <= 1

    def _bump_resident(mem_name: str, nbytes: int, *, at_s: float, reason: str) -> None:
        if nbytes <= 0:
            return
        resident[mem_name] = resident.get(mem_name, 0) + nbytes
        peak[mem_name] = max(peak.get(mem_name, 0), resident[mem_name])
        mem = machine.memory.get(mem_name)
        capacity = int(mem.allocatable_bytes) if mem is not None else 0
        if capacity > 0 and resident[mem_name] > capacity:
            timeline.append(
                {
                    "event": "eviction_pressure",
                    "memory": mem_name,
                    "resident_bytes": resident[mem_name],
                    "allocatable_bytes": capacity,
                    "at_s": at_s,
                    "reason": reason,
                    "simulated": True,
                    "validated": False,
                }
            )

    def _sync_state_leases(at_s: float) -> None:
        """Drop state whose region already finished before ``at_s``."""
        kept: list[tuple[float, str, int, str]] = []
        for end_s, mem_name, nbytes, region_id in state_leases:
            if end_s > at_s:
                kept.append((end_s, mem_name, nbytes, region_id))
                continue
            resident[mem_name] = max(0, resident.get(mem_name, 0) - nbytes)
            event = {
                "event": "release",
                "region": region_id,
                "memory": mem_name,
                "nbytes": nbytes,
                "at_s": end_s,
                "kind": "region_state",
                "simulated": True,
            }
            release_events.append(event)
            timeline.append(event)
        state_leases[:] = kept

    prev_id: str | None = None
    for index, placement in enumerate(plan.placements):
        deps = list(placement.depends_on)
        if use_implicit_chain and prev_id is not None:
            deps.append(prev_id)

        dep_ready = 0.0
        active_transfers = 0
        for dep in deps:
            if dep not in region_end:
                continue
            ready = region_end[dep]
            producer = by_id.get(dep)
            if producer is not None and producer.device != placement.device:
                active_on_links = sum(1 for t in link_free_at.values() if t > ready)
                xfer_factor = concurrent_slowdown(
                    active_compute=max(
                        1,
                        sum(1 for t in resource_free_at.values() if t > ready),
                    ),
                    active_transfers=max(1, active_on_links),
                    active_storage=1 if prefetch_distance > 0 else 0,
                ).transfer
                hop, transfer_meta = _schedule_transfer(
                    machine,
                    producer,
                    placement,
                    ready_at=ready,
                    link_free_at=link_free_at,
                    contention_factor=xfer_factor,
                )
                exposed += hop
                ready = transfer_meta["end_s"]
                active_transfers += 1
                transfer_events.append(transfer_meta)
                dest_mem = _primary_memory(machine, placement.device)
                nbytes = max(0, producer.output_bytes)
                if dest_mem is not None and nbytes > 0:
                    _bump_resident(
                        dest_mem,
                        nbytes,
                        at_s=transfer_meta["end_s"],
                        reason="transfer_landed",
                    )
                    dest_copies[placement.region_id].append((dest_mem, nbytes))
                    timeline.append(
                        {
                            "event": "transfer_landed",
                            "memory": dest_mem,
                            "nbytes": nbytes,
                            "at_s": transfer_meta["end_s"],
                            "source_region": producer.region_id,
                            "destination_region": placement.region_id,
                            "simulated": True,
                        }
                    )
            dep_ready = max(dep_ready, ready)

        if prefetch_distance > 0 and index + 1 < len(plan.placements):
            nxt = plan.placements[index + 1]
            timeline.append(
                {
                    "event": "prefetch_hint",
                    "region": nxt.region_id,
                    "device": nxt.device,
                    "after": placement.region_id,
                    "distance": prefetch_distance,
                    "simulated": True,
                }
            )

        start = max(resource_free_at.get(placement.device, 0.0), dep_ready)
        # Free state that finished before this start so overlapping peers still count.
        _sync_state_leases(start)
        active = sum(1 for t in resource_free_at.values() if t > start)
        factors = concurrent_slowdown(
            active_compute=max(1, active),
            active_transfers=active_transfers,
            active_storage=1 if prefetch_distance > 0 else 0,
        )
        dur = float(placement.estimated_latency_s) * factors.compute
        end = start + dur
        resource_free_at[placement.device] = end
        resource_busy[placement.device] = resource_busy.get(placement.device, 0.0) + dur
        region_end[placement.region_id] = end

        mem_name = _primary_memory(machine, placement.device)
        state = max(0, placement.state_bytes)
        if mem_name is not None and state > 0:
            _bump_resident(mem_name, state, at_s=start, reason="region_state")
            state_leases.append((end, mem_name, state, placement.region_id))

        # Allocate outputs at end while this region's state is still resident.
        out_bytes = max(0, placement.output_bytes)
        if mem_name is not None and out_bytes > 0:
            remaining = consumer_count.get(placement.region_id, 0)
            if remaining == 0:
                # No recorded consumer: treat as live through makespan (e.g. final output).
                remaining = 1
            live_outputs[placement.region_id] = (mem_name, out_bytes, remaining)
            _bump_resident(mem_name, out_bytes, at_s=end, reason="region_output")

        timeline.append(
            {
                "event": "compute",
                "region": placement.region_id,
                "device": placement.device,
                "backend": placement.backend_id,
                "dtype": placement.dtype,
                "start_s": start,
                "end_s": end,
                "working_set_bytes": placement.working_set_bytes,
                "output_bytes": out_bytes,
                "state_bytes": state,
                "contention": {
                    "compute": factors.compute,
                    "transfer": factors.transfer,
                    "storage": factors.storage,
                },
                "simulated": True,
            }
        )

        # Release consumed producers after this consumer finishes.
        for dep in deps:
            meta = live_outputs.get(dep)
            if meta is None:
                continue
            mem, nbytes, remaining = meta
            remaining -= 1
            if remaining > 0:
                live_outputs[dep] = (mem, nbytes, remaining)
                continue
            live_outputs.pop(dep, None)
            resident[mem] = max(0, resident.get(mem, 0) - nbytes)
            release_events.append(
                {
                    "event": "release",
                    "region": dep,
                    "memory": mem,
                    "nbytes": nbytes,
                    "at_s": end,
                    "simulated": True,
                }
            )
            timeline.append(release_events[-1])

        # Release destination-side copies of transferred inputs.
        for mem, nbytes in dest_copies.pop(placement.region_id, []):
            resident[mem] = max(0, resident.get(mem, 0) - nbytes)
            release_events.append(
                {
                    "event": "release",
                    "region": placement.region_id,
                    "memory": mem,
                    "nbytes": nbytes,
                    "at_s": end,
                    "kind": "transfer_copy",
                    "simulated": True,
                }
            )
            timeline.append(release_events[-1])

        prev_id = placement.region_id
        makespan = max(makespan, end)

    _sync_state_leases(makespan)
    utilization = {name: (busy / makespan if makespan > 0 else 0.0) for name, busy in resource_busy.items()}
    return SimulationResult(
        makespan_s=makespan,
        peak_bytes=peak,
        timeline=timeline,
        exposed_transfer_latency_s=exposed,
        resource_busy_s=resource_busy,
        transfer_events=transfer_events,
        release_events=release_events,
        simulated=True,
        resource_utilization=utilization,
    )


def _consumer_counts(plan: ExecutionPlan) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for placement in plan.placements:
        for dep in placement.depends_on:
            counts[dep] += 1
    return dict(counts)


def _prefetch_distance(plan: ExecutionPlan) -> int:
    for note in plan.notes:
        if note.startswith("prefetch_distance="):
            try:
                return max(0, int(note.split("=", 1)[1]))
            except ValueError:
                return 0
    return 0


def _primary_memory(machine: ResourceGraph, device_name: str) -> str | None:
    device = machine.compute.get(device_name)
    if device is None or not device.memory_affinity:
        return None
    return device.memory_affinity[0]


def _schedule_transfer(
    machine: ResourceGraph,
    producer: Placement,
    consumer: Placement,
    *,
    ready_at: float,
    link_free_at: dict[str, float],
    contention_factor: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    hop = _transfer_latency_s(machine, producer, consumer) * max(1.0, float(contention_factor))
    link = _best_link(machine, producer.device, consumer.device)
    link_id = link.id.name if link is not None else f"{producer.device}->{consumer.device}"
    start = max(ready_at, link_free_at.get(link_id, 0.0))
    end = start + hop
    link_free_at[link_id] = end
    return hop, {
        "event": "transfer",
        "source_region": producer.region_id,
        "destination_region": consumer.region_id,
        "source_device": producer.device,
        "destination_device": consumer.device,
        "nbytes": max(0, producer.output_bytes),
        "link": link_id,
        "start_s": start,
        "end_s": end,
        "latency_s": hop,
        "contention_factor": float(contention_factor),
        "simulated": True,
    }


def _transfer_latency_s(machine: ResourceGraph, producer: Placement, consumer: Placement) -> float:
    """Time to move the producer's outputs onto the consumer's device."""
    nbytes = max(0, producer.output_bytes)
    if nbytes == 0:
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
    """Pick the fastest memory-to-memory link between two compute devices."""
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
        for link in machine.links.values():
            if link.link_class.value != "host_staged":
                continue
            if link.source in src.memory_affinity or link.destination in dst.memory_affinity:
                candidates.append(link)
    if not candidates:
        return None

    def rank(link: TransferLink) -> tuple[int, float]:
        score = (2 if link.peer_to_peer else 0) + (1 if link.measured else 0)
        bandwidth = float(link.bytes_per_s or 0.0)
        return (score, bandwidth)

    return max(candidates, key=rank)


def simulate_schedule(schedule: Any, machine: ResourceGraph) -> SimulationResult:
    """Simulate an :class:`ExecutableSchedule` instruction dependency DAG directly.

    Walks Prefetch/Load/Transfer/RecordEvent/WaitEvent/Compute/Evict/Release with
    the same instruction IDs the runtime uses. Does not reconstruct placements or
    invent transfers absent from the schedule. Analytic only — no kernel execution.
    """
    from collections import defaultdict, deque

    from streamcompiler.ir.graph import OpCode
    from streamcompiler.runtime.schedule import ExecutableSchedule

    if not isinstance(schedule, ExecutableSchedule):
        raise TypeError(f"simulate_schedule expects ExecutableSchedule, got {type(schedule).__name__}")

    by_name = {i.name: i for i in schedule.instructions}
    remaining: dict[str, set[str]] = {i.name: set(i.depends_on) for i in schedule.instructions}
    dependents: dict[str, list[str]] = defaultdict(list)
    for inst in schedule.instructions:
        for dep in inst.depends_on:
            dependents[dep].append(inst.name)

    ready: deque[str] = deque(n for n, deps in remaining.items() if not deps)
    compute_free: dict[str, float] = {name: 0.0 for name in machine.compute}
    copy_free: dict[str, float] = defaultdict(float)
    io_free = 0.0
    resource_busy: dict[str, float] = {name: 0.0 for name in machine.compute}
    peak: dict[str, int] = {name: 0 for name in machine.memory}
    resident: dict[str, int] = {name: 0 for name in machine.memory}
    copies: dict[tuple[str, str], int] = {}
    event_ready_at: dict[str, float] = {}
    inst_end: dict[str, float] = {}
    timeline: list[dict[str, Any]] = []
    transfer_events: list[dict[str, Any]] = []
    release_events: list[dict[str, Any]] = []
    exposed = 0.0
    bytes_read = 0
    bytes_transferred = 0
    cp_pred: dict[str, str | None] = {}
    cp_finish: dict[str, float] = {}

    def _mem_for(resource: str) -> str:
        if resource in machine.memory:
            return resource
        compute = machine.compute.get(resource)
        if compute is not None and compute.memory_affinity:
            return compute.memory_affinity[0]
        for name in machine.memory:
            if any(tok in name.lower() for tok in ("ram", "host", "numa")):
                return name
        return next(iter(machine.memory), "system_ram")

    def _bump(mem: str, nbytes: int, at_s: float, reason: str) -> None:
        if nbytes <= 0:
            return
        resident[mem] = resident.get(mem, 0) + nbytes
        peak[mem] = max(peak.get(mem, 0), resident[mem])
        capacity = int(machine.memory[mem].allocatable_bytes) if mem in machine.memory else 0
        if capacity > 0 and resident[mem] > capacity:
            timeline.append(
                {
                    "event": "eviction_pressure",
                    "instruction": reason,
                    "memory": mem,
                    "resident_bytes": resident[mem],
                    "allocatable_bytes": capacity,
                    "at_s": at_s,
                    "simulated": True,
                    "validated": False,
                }
            )

    def _free_mem(mem: str, nbytes: int) -> None:
        if nbytes > 0:
            resident[mem] = max(0, resident.get(mem, 0) - nbytes)

    def _dep_ready(name: str) -> float:
        deps = by_name[name].depends_on
        return max((inst_end.get(d, 0.0) for d in deps), default=0.0)

    def _duration(inst: Any) -> float:
        attrs = inst.attributes or {}
        if inst.opcode == OpCode.COMPUTE:
            delay = float(attrs.get("mock_compute_delay_s", 0.0))
            return delay if delay > 0 else max(1e-9, float(inst.predicted_duration_s or 1e-6))
        if inst.opcode == OpCode.TRANSFER:
            delay = float(attrs.get("mock_transfer_delay_s", 0.0))
            if delay > 0:
                return delay
            link = _pick_link(machine, str(inst.source or ""), str(inst.destination or ""))
            if link is not None:
                model = TransferModel(
                    source=link.source,
                    destination=link.destination,
                    alpha_s=float(link.latency_s or 0.0),
                    beta_bytes_per_s=float(link.bytes_per_s) if link.bytes_per_s else None,
                    measured=bool(link.measured),
                )
                return transfer_time(model, link.source, link.destination, max(1, inst.nbytes))
            return max(1e-6, (inst.nbytes or 1) / 4e9)
        if inst.opcode in (OpCode.PREFETCH, OpCode.LOAD):
            return max(1e-6, float(inst.nbytes or 1) / (500 * (1 << 20)))
        return max(0.0, float(inst.predicted_duration_s or 0.0))

    while ready:
        name = ready.popleft()
        inst = by_name[name]
        dep_t = _dep_ready(name)
        duration = _duration(inst)
        opcode = inst.opcode
        start = dep_t
        end = dep_t

        if opcode == OpCode.PREFETCH:
            start = max(dep_t, io_free)
            end = start + duration
            io_free = end
            bytes_read += max(0, inst.nbytes)
            timeline.append(
                {
                    "event": "Prefetch",
                    "instruction": name,
                    "opcode": opcode.value,
                    "resource": inst.resource,
                    "start_s": start,
                    "end_s": end,
                    "nbytes": inst.nbytes,
                    "simulated": True,
                }
            )
        elif opcode == OpCode.LOAD:
            start = max(dep_t, io_free)
            end = start + duration
            io_free = end
            bytes_read += max(0, inst.nbytes)
            dest = str(inst.destination or inst.resource)
            dmem = _mem_for(dest)
            for tensor in inst.outputs or inst.inputs:
                copies[(tensor, dest)] = max(1, inst.nbytes)
            _bump(dmem, max(0, inst.nbytes), end, name)
            timeline.append(
                {
                    "event": "Load",
                    "instruction": name,
                    "opcode": opcode.value,
                    "resource": dest,
                    "start_s": start,
                    "end_s": end,
                    "nbytes": inst.nbytes,
                    "simulated": True,
                    "notes": "disk→RAM",
                }
            )
        elif opcode == OpCode.TRANSFER:
            src = str(inst.source or "")
            dst = str(inst.destination or "")
            engine = str(inst.resource)
            start = max(dep_t, copy_free[engine])
            end = start + duration
            copy_free[engine] = end
            bytes_transferred += max(0, inst.nbytes)
            dmem = _mem_for(dst)
            for tensor in inst.outputs or inst.inputs:
                copies[(tensor, dst)] = max(1, inst.nbytes)
            _bump(dmem, max(0, inst.nbytes), end, name)
            tev = {
                "event": "Transfer",
                "instruction": name,
                "opcode": opcode.value,
                "source": src,
                "destination": dst,
                "start_s": start,
                "end_s": end,
                "nbytes": inst.nbytes,
                "simulated": True,
            }
            transfer_events.append(tev)
            timeline.append(tev)
        elif opcode == OpCode.RECORD_EVENT:
            start = end = dep_t
            pred = inst.depends_on[0] if inst.depends_on else name
            event_ready_at[name] = inst_end.get(pred, dep_t)
            timeline.append(
                {
                    "event": "RecordEvent",
                    "instruction": name,
                    "opcode": opcode.value,
                    "resource": inst.resource,
                    "start_s": start,
                    "end_s": end,
                    "ready_s": event_ready_at[name],
                    "simulated": True,
                }
            )
        elif opcode == OpCode.WAIT_EVENT:
            waits_for = str((inst.attributes or {}).get("waits_for") or (inst.depends_on[0] if inst.depends_on else ""))
            ready_s = event_ready_at.get(waits_for, dep_t)
            for d in inst.depends_on:
                if d in event_ready_at:
                    ready_s = max(ready_s, event_ready_at[d])
            start = dep_t
            stall = max(0.0, ready_s - dep_t)
            exposed += stall
            end = max(dep_t, ready_s)
            timeline.append(
                {
                    "event": "WaitEvent",
                    "instruction": name,
                    "opcode": opcode.value,
                    "resource": inst.resource,
                    "start_s": start,
                    "end_s": end,
                    "exposed_stall_s": stall,
                    "waits_for": waits_for,
                    "simulated": True,
                }
            )
        elif opcode == OpCode.COMPUTE:
            device = str(inst.resource)
            start = max(dep_t, compute_free.get(device, 0.0))
            end = start + duration
            compute_free[device] = end
            resource_busy[device] = resource_busy.get(device, 0.0) + duration
            for tensor in inst.outputs:
                copies[(tensor, device)] = max(1, inst.nbytes)
            _bump(_mem_for(device), max(0, inst.nbytes), end, name)
            timeline.append(
                {
                    "event": "Compute",
                    "instruction": name,
                    "opcode": opcode.value,
                    "resource": device,
                    "start_s": start,
                    "end_s": end,
                    "nbytes": inst.nbytes,
                    "executable_ref": inst.executable_ref,
                    "simulated": True,
                }
            )
        elif opcode in (OpCode.EVICT, OpCode.RELEASE):
            start = end = dep_t
            resource = str(inst.destination or inst.resource)
            freed = 0
            for tensor in inst.inputs:
                if opcode == OpCode.EVICT:
                    keys = [(tensor, resource)] if (tensor, resource) in copies else []
                else:
                    keys = [k for k in list(copies) if k[0] == tensor]
                for key in keys:
                    nbytes = copies.pop(key, 0)
                    freed += nbytes
                    _free_mem(_mem_for(key[1]), nbytes)
            rev = {
                "event": opcode.value,
                "instruction": name,
                "opcode": opcode.value,
                "resource": resource,
                "start_s": start,
                "end_s": end,
                "nbytes": freed,
                "simulated": True,
            }
            release_events.append(rev)
            timeline.append(rev)
        else:
            start = dep_t
            end = dep_t + duration
            timeline.append(
                {
                    "event": opcode.value,
                    "instruction": name,
                    "opcode": opcode.value,
                    "resource": inst.resource,
                    "start_s": start,
                    "end_s": end,
                    "simulated": True,
                }
            )

        inst_end[name] = end
        best_pred: str | None = None
        best_t = -1.0
        for d in inst.depends_on:
            t = inst_end.get(d, 0.0)
            if t >= best_t:
                best_t = t
                best_pred = d
        cp_pred[name] = best_pred
        cp_finish[name] = end

        for child in dependents.get(name, ()):
            remaining[child].discard(name)
            if not remaining[child] and child not in inst_end and child not in ready:
                ready.append(child)

    if len(inst_end) != len(by_name):
        missing = sorted(set(by_name) - set(inst_end))
        raise ValueError(f"simulate_schedule could not schedule instructions (cycle?): {missing}")

    makespan = max(inst_end.values()) if inst_end else 0.0
    critical: list[str] = []
    if cp_finish:
        node: str | None = max(cp_finish, key=lambda n: cp_finish[n])
        seen: set[str] = set()
        while node is not None and node not in seen:
            critical.append(node)
            seen.add(node)
            node = cp_pred.get(node)
        critical.reverse()

    utilization = {name: (busy / makespan if makespan > 0 else 0.0) for name, busy in resource_busy.items()}
    return SimulationResult(
        makespan_s=makespan,
        peak_bytes=dict(peak),
        timeline=timeline,
        exposed_transfer_latency_s=exposed,
        resource_busy_s=dict(resource_busy),
        transfer_events=transfer_events,
        release_events=release_events,
        simulated=True,
        critical_path=critical,
        bytes_read=bytes_read,
        bytes_transferred=bytes_transferred,
        instruction_count=len(by_name),
        resource_utilization=utilization,
    )


def _pick_link(machine: ResourceGraph, src: str, dst: str) -> TransferLink | None:
    if not src or not dst:
        return None
    direct = machine.link_between(src, dst)
    if direct is not None:
        return direct
    candidates = [
        link
        for link in machine.links.values()
        if src in (link.source, link.destination) or dst in (link.source, link.destination)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda link: (1 if link.measured else 0, float(link.bytes_per_s or 0.0)))
