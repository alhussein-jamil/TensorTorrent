"""Discrete-event simulator for :class:`ExecutableSchedule` instruction DAGs.

Analytic only: kernels are not executed. Makespan, transfer exposure, peak
memory, and contention come from schedule instruction costs, explicit
dependencies, and the machine's transfer links. ``simulate_plan`` is a thin
wrapper that lowers an ``ExecutionPlan`` to an executable schedule first —
the simulator never invents transfers absent from that schedule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from streamcompiler.cost_model.transfer import TransferModel, transfer_time
from streamcompiler.ir.resource_graph import ResourceGraph, TransferLink
from streamcompiler.planner.maximal import ExecutionPlan


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


def simulate_plan(
    plan: ExecutionPlan,
    machine: ResourceGraph,
    *,
    residency: Any | None = None,
    streaming: bool = False,
    prefetch_distance: int = 1,
    program: Any | None = None,
) -> SimulationResult:
    """Lower ``plan`` to an :class:`ExecutableSchedule`, then simulate that DAG.

    Does **not** infer transfers inside the simulator. Cross-device movement must
    appear as Transfer instructions (via residency / schedule builder). Kept as a
    thin compatibility entry for tests that still hand-build ``ExecutionPlan``s.
    """
    from streamcompiler.runtime.residency import build_residency_schedule
    from streamcompiler.runtime.schedule import build_executable_schedule

    if residency is None:
        residency = build_residency_schedule(plan, program)
    if streaming is False:
        for note in plan.notes:
            if note.startswith("prefetch_distance="):
                try:
                    prefetch_distance = max(0, int(note.split("=", 1)[1]))
                    streaming = prefetch_distance > 0
                except ValueError:
                    pass
                break
    schedule = build_executable_schedule(
        plan,
        residency,
        streaming=streaming,
        prefetch_distance=prefetch_distance,
        program=program,
    )
    return simulate_schedule(schedule, machine)


def simulate_schedule(schedule: Any, machine: ResourceGraph) -> SimulationResult:
    """Simulate an :class:`ExecutableSchedule` instruction dependency DAG directly.

    Walks Prefetch/Load/Transfer/RecordEvent/WaitEvent/Compute/Evict/Release with
    the same instruction IDs the runtime uses. Does not reconstruct placements or
    invent transfers absent from the schedule. Analytic only — no kernel execution.
    """
    from collections import deque

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
    state_leases: list[tuple[float, str, int]] = []  # (end_s, mem, nbytes)
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

    def _release_state_due(at_s: float) -> None:
        kept: list[tuple[float, str, int]] = []
        for end_s, mem, nbytes in state_leases:
            if end_s <= at_s + 1e-15:
                _free_mem(mem, nbytes)
            else:
                kept.append((end_s, mem, nbytes))
        state_leases[:] = kept

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
        _release_state_due(dep_t)

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
                "source_device": src,
                "destination_device": dst,
                "source_region": (inst.attributes or {}).get("after_region"),
                "destination_region": (inst.attributes or {}).get("before_region"),
                "start_s": start,
                "end_s": end,
                "nbytes": inst.nbytes,
                "contention_factor": 1.0,
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
            attrs = inst.attributes or {}
            start = max(dep_t, compute_free.get(device, 0.0))
            _release_state_due(start)
            end = start + duration
            compute_free[device] = end
            resource_busy[device] = resource_busy.get(device, 0.0) + duration
            mem = _mem_for(device)
            state = max(0, int(attrs.get("state_bytes", 0) or 0))
            # Load already accounted for parameter residency when present.
            state_already = any(
                (tensor, device) in copies and (tensor.startswith("state::") or "state" in tensor)
                for tensor in inst.inputs
            )
            if state > 0 and not state_already:
                _bump(mem, state, start, name)
                state_leases.append((end, mem, state))
            for tensor in inst.outputs:
                copies[(tensor, device)] = max(1, inst.nbytes)
            _bump(mem, max(0, inst.nbytes), end, name)
            timeline.append(
                {
                    "event": "Compute",
                    "instruction": name,
                    "opcode": opcode.value,
                    "resource": device,
                    "start_s": start,
                    "end_s": end,
                    "nbytes": inst.nbytes,
                    "state_bytes": state,
                    "executable_ref": inst.executable_ref,
                    "simulated": True,
                }
            )
        elif opcode in (OpCode.EVICT, OpCode.RELEASE):
            start = end = dep_t
            attrs = inst.attributes or {}
            resource = str(attrs.get("release_resource") or inst.destination or inst.resource)
            freed = 0
            for tensor in inst.inputs:
                key = (tensor, resource)
                if key not in copies:
                    # Strict: do not drop sibling resource copies.
                    continue
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
    _release_state_due(makespan)
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
    """Pick the best memory-to-memory link for compute or memory resource ids."""
    if not src or not dst:
        return None
    direct = machine.link_between(src, dst)
    if direct is not None:
        return direct

    def _mems(name: str) -> tuple[str, ...]:
        if name in machine.memory:
            return (name,)
        compute = machine.compute.get(name)
        if compute is not None and compute.memory_affinity:
            return tuple(compute.memory_affinity)
        return (name,)

    candidates: list[TransferLink] = []
    for s_mem in _mems(src):
        for d_mem in _mems(dst):
            link = machine.link_between(s_mem, d_mem)
            if link is not None:
                candidates.append(link)
            reverse = machine.link_between(d_mem, s_mem)
            if reverse is not None and reverse.bidirectional:
                candidates.append(reverse)
    if not candidates:
        candidates = [
            link
            for link in machine.links.values()
            if src in (link.source, link.destination) or dst in (link.source, link.destination)
        ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda link: (
            (1 if link.peer_to_peer else 0) + (1 if link.measured else 0),
            float(link.bytes_per_s or 0.0),
        ),
    )
