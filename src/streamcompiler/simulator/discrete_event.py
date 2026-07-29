"""Discrete-event simulator for :class:`ExecutableSchedule` instruction DAGs.

Analytic only: kernels are not executed. Makespan, transfer exposure, peak
memory, and contention come from schedule instruction costs, explicit
dependencies, and the machine's transfer links. ``simulate_plan`` is a thin
wrapper that lowers an ``ExecutionPlan`` to an executable schedule first —
the simulator never invents transfers absent from that schedule.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
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
    activation_peak_bytes: int = 0
    """Peak bytes of distinct live physical activation allocations."""


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

    Shares the Rust schedule model via typed bindings. The analytic Python
    discrete-event walk remains the default planner oracle until the Rust
    simulator reaches bit-level agreement on peak-memory tests. Set
    ``STREAMCOMPILER_NATIVE_SIM=1`` to force the Rust simulator (always
    labelled ``simulated=True``).
    """
    import os

    from streamcompiler.native import native_available, require_native
    from streamcompiler.runtime.schedule import ExecutableSchedule

    if not isinstance(schedule, ExecutableSchedule):
        raise TypeError(f"simulate_schedule expects ExecutableSchedule, got {type(schedule).__name__}")

    use_native = os.environ.get("STREAMCOMPILER_NATIVE_SIM", "").strip() in {"1", "true", "yes"}
    if use_native and native_available():
        native = require_native()
        raw = native.simulate_schedule(schedule, machine)
        timeline = list(raw.get("timeline") or [])
        return SimulationResult(
            makespan_s=float(raw.get("makespan_s") or 0.0),
            peak_bytes={str(k): int(v) for k, v in dict(raw.get("peak_bytes") or {}).items()},
            timeline=timeline,
            exposed_transfer_latency_s=float(raw.get("exposed_transfer_latency_s") or 0.0),
            resource_busy_s={str(k): float(v) for k, v in dict(raw.get("resource_busy_s") or {}).items()},
            simulated=True,
            critical_path=[str(x) for x in list(raw.get("critical_path") or [])],
            bytes_read=int(raw.get("bytes_read") or 0),
            bytes_transferred=int(raw.get("bytes_transferred") or 0),
            instruction_count=int(raw.get("instruction_count") or len(schedule.instructions)),
            activation_peak_bytes=int(raw.get("activation_peak_bytes") or 0),
        )

    return _simulate_schedule_python(schedule, machine)


def _simulate_schedule_python(schedule: Any, machine: ResourceGraph) -> SimulationResult:
    """Python discrete-event walk (planner oracle during Rust sim parity work)."""
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
    copies: dict[tuple[str, str], str] = {}
    allocations: dict[str, tuple[str, int, int]] = {}  # allocation_id -> (memory, capacity, refs)
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
    # Implicit resource-order predecessors for critical-path reconstruction.
    last_on_compute: dict[str, str] = {}
    last_on_copy: dict[str, str] = {}
    last_on_io: str | None = None

    def _mem_for(resource: str) -> str:
        if resource in machine.memory:
            return resource
        compute = machine.compute.get(resource)
        if compute is not None and compute.memory_affinity:
            return compute.memory_affinity[0]
        name = str(resource).lower()
        # Host-side Load destinations must never alias into device VRAM peaks.
        if any(tok in name for tok in ("cpu", "host", "numa", "pinned", "system_ram")) or name == "disk":
            for mem_name, mem in machine.memory.items():
                cls = str(getattr(mem.memory_class, "value", mem.memory_class)).lower()
                if "vram" in mem_name.lower() or "device" in cls:
                    continue
                if any(tok in mem_name.lower() for tok in ("ram", "host", "numa")):
                    return mem_name
            return "host_ram"
        for mem_name in machine.memory:
            if any(tok in mem_name.lower() for tok in ("ram", "host", "numa")):
                return mem_name
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

    def _allocation_id(tensor: str, resource: str, inst: Any) -> str:
        raw = inst.attributes.get("allocation_ids")
        if isinstance(raw, Mapping) and tensor in raw:
            return str(raw[tensor])
        return f"sim::{resource}::{tensor}"

    def _install_copy(tensor: str, resource: str, nbytes: int, inst: Any, at_s: float) -> None:
        key = (tensor, resource)
        new_alloc = _allocation_id(tensor, resource, inst)
        previous = copies.get(key)
        if previous == new_alloc:
            return
        if previous is not None:
            _drop_copy(tensor, resource)
        mem = _mem_for(resource)
        rec = allocations.get(new_alloc)
        if rec is None:
            allocations[new_alloc] = (mem, max(0, int(nbytes)), 1)
            _bump(mem, max(0, int(nbytes)), at_s, inst.name)
        else:
            old_mem, capacity, refs = rec
            allocations[new_alloc] = (old_mem, max(capacity, int(nbytes)), refs + 1)
        copies[key] = new_alloc

    def _drop_copy(tensor: str, resource: str) -> int:
        key = (tensor, resource)
        alloc_id = copies.pop(key, None)
        if alloc_id is None:
            return 0
        rec = allocations.get(alloc_id)
        if rec is None:
            return 0
        mem, capacity, refs = rec
        if refs > 1:
            allocations[alloc_id] = (mem, capacity, refs - 1)
            return 0
        del allocations[alloc_id]
        _free_mem(mem, capacity)
        return capacity

    def _release_state_due(at_s: float) -> None:
        kept: list[tuple[float, str, int]] = []
        for end_s, mem, nbytes in state_leases:
            if end_s <= at_s + 1e-15:
                _free_mem(mem, nbytes)
            else:
                kept.append((end_s, mem, nbytes))
        state_leases[:] = kept

    def _tensor_nbytes(inst: Any, tensor: str) -> int:
        """Exact per-tensor size — never equal-split aggregate nbytes."""
        raw = inst.attributes.get("tensor_nbytes")
        if isinstance(raw, Mapping) and tensor in raw:
            return max(0, int(raw[tensor] or 0))
        for key in ("input_bytes", "output_bytes"):
            block = inst.attributes.get(key)
            if isinstance(block, Mapping) and tensor in block:
                return max(0, int(block[tensor] or 0))
        tensors = tuple(inst.outputs or inst.inputs or ())
        if len(tensors) == 1 and tensors[0] == tensor:
            return max(0, int(inst.nbytes or 0))
        # Fail soft with zero rather than inventing equal-split bytes.
        return max(0, int(inst.nbytes or 0)) if not tensors else 0

    activation_peak = 0
    activation_ids: set[str] = set()

    def _resync_activation() -> None:
        """Count distinct physical activation allocations across resources."""
        nonlocal activation_peak
        active_allocs: set[str] = set()
        for (tid, rid), alloc_id in copies.items():
            if rid == "disk" or tid not in activation_ids:
                continue
            active_allocs.add(alloc_id)
        live = int(sum(allocations[a][1] for a in active_allocs if a in allocations))
        activation_peak = max(activation_peak, live)

    def _dep_ready(name: str) -> float:
        deps = by_name[name].depends_on
        return max((inst_end.get(d, 0.0) for d in deps), default=0.0)

    def _resource_preds(inst: Any) -> list[str]:
        preds: list[str] = []
        if inst.opcode == OpCode.COMPUTE:
            prev = last_on_compute.get(str(inst.resource))
            if prev:
                preds.append(prev)
        elif inst.opcode == OpCode.TRANSFER:
            engine = str(inst.resource)
            prev = last_on_copy.get(engine)
            if prev:
                preds.append(prev)
        elif inst.opcode in (OpCode.PREFETCH, OpCode.LOAD) or (
            inst.opcode == OpCode.EVICT and str(inst.attributes.get("kind") or "") == "activation_spill"
        ):
            if last_on_io:
                preds.append(last_on_io)
        return preds

    def _duration(inst: Any) -> float:
        attrs: Mapping[str, Any] = inst.attributes
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
            # Load is always storage → host-accessible RAM (never device VRAM).
            if any(tok in dest.lower() for tok in ("mock", "cuda", "rocm", "gpu", "xpu", "mps", "vram")):
                dest = "cpu"
            dmem = _mem_for(dest)
            kind = str(inst.attributes.get("kind") or "")
            for tensor in inst.outputs or inst.inputs:
                # Keep disk copy after activation_reload — runtime shares one
                # spill file across parallel consumers (delete=False).
                n = _tensor_nbytes(inst, tensor)
                _install_copy(tensor, dest, n, inst, end)
                if kind == "activation_reload":
                    activation_ids.add(tensor)
            if kind == "activation_reload":
                _resync_activation()
            del dmem  # copy installation performs exact physical accounting
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
                    "notes": "disk→RAM" + (" activation_reload" if kind == "activation_reload" else ""),
                    "activation_bytes_read": inst.nbytes if kind == "activation_reload" else 0,
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
                _install_copy(tensor, dst, _tensor_nbytes(inst, tensor), inst, end)
            _resync_activation()
            del dmem  # copy installation performs exact physical accounting
            tev = {
                "event": "Transfer",
                "instruction": name,
                "opcode": opcode.value,
                "source": src,
                "destination": dst,
                "source_device": src,
                "destination_device": dst,
                "source_region": inst.attributes.get("after_region"),
                "destination_region": inst.attributes.get("before_region"),
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
            waits_for = str(inst.attributes.get("waits_for") or (inst.depends_on[0] if inst.depends_on else ""))
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
            attrs: Mapping[str, Any] = inst.attributes
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
                n = _tensor_nbytes(inst, tensor)
                _install_copy(tensor, device, n, inst, end)
                activation_ids.add(tensor)
            _resync_activation()
            del mem  # output copy installation performs exact physical accounting
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
            kind = str(inst.attributes.get("kind") or "")
            if opcode == OpCode.EVICT and kind == "activation_spill":
                start = max(dep_t, io_free)
                end = start + max(1e-6, float(inst.nbytes or 1) / (500 * (1 << 20)))
                io_free = end
                resource = str(inst.attributes.get("spill_resource") or inst.source or inst.resource)
                freed = 0
                written = 0
                for tensor in inst.inputs:
                    key = (tensor, resource)
                    nbytes = _tensor_nbytes(inst, tensor) or max(1, int(inst.nbytes or 1))
                    freed += _drop_copy(tensor, resource)
                    written += nbytes
                    _install_copy(tensor, "disk", nbytes, inst, end)
                _resync_activation()
                bytes_read += 0  # spill is a write; tracked on timeline
                rev = {
                    "event": "Evict",
                    "instruction": name,
                    "opcode": opcode.value,
                    "resource": resource,
                    "start_s": start,
                    "end_s": end,
                    "nbytes": freed,
                    "simulated": True,
                    "notes": "activation_spill RAM→disk",
                    "activation_bytes_written": written,
                }
                release_events.append(rev)
                timeline.append(rev)
            else:
                resource = str(inst.attributes.get("release_resource") or inst.destination or inst.resource)
                freed = 0
                for tensor in inst.inputs:
                    key = (tensor, resource)
                    if key not in copies:
                        key = (tensor, "disk")
                        if key not in copies:
                            continue
                    freed += _drop_copy(tensor, key[1])
                _resync_activation()
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
        # Critical-path predecessor: latest among explicit deps and resource order.
        candidates = list(inst.depends_on) + _resource_preds(inst)
        # Update resource-order chains after using prior values.
        if opcode == OpCode.COMPUTE:
            last_on_compute[str(inst.resource)] = name
        elif opcode == OpCode.TRANSFER:
            last_on_copy[str(inst.resource)] = name
        elif opcode in (OpCode.PREFETCH, OpCode.LOAD) or (
            opcode == OpCode.EVICT and str(inst.attributes.get("kind") or "") == "activation_spill"
        ):
            last_on_io = name
        best_pred: str | None = None
        best_t = -1.0
        for d in candidates:
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
        activation_peak_bytes=activation_peak,
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
