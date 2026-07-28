"""One executable plan format shared by planner, simulator, and runtime.

Instructions are explicit memory/compute ops. The simulator must not invent
schedules the runtime cannot perform: both consume :class:`ExecutableSchedule`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from streamcompiler.ir.graph import OpCode
from streamcompiler.planner.maximal import ExecutionPlan, Placement
from streamcompiler.runtime.residency import ResidencySchedule, ScheduledTransfer


class MemoryTier(str, Enum):
    DISK = "disk"
    SYSTEM_RAM = "system_ram"
    PINNED_RAM = "pinned_ram"
    DEVICE = "device"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PlanInstruction:
    """One scheduled op the runtime can execute and the simulator can cost."""

    opcode: OpCode
    name: str
    resource: str
    depends_on: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    nbytes: int = 0
    memory_tier: MemoryTier = MemoryTier.UNKNOWN
    predicted_duration_s: float = 0.0
    executable_ref: str | None = None
    """Region id or compiled-region key when opcode is Compute."""
    source: str | None = None
    destination: str | None = None
    backend_id: str | None = None
    transfer_backend: str | None = None
    sync_required: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["opcode"] = self.opcode.value
        payload["memory_tier"] = self.memory_tier.value
        return payload


@dataclass
class ExecutableSchedule:
    """Linearized executable plan: same object for plan explain, sim, and run."""

    graph_name: str
    fingerprint: str
    instructions: list[PlanInstruction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def compute_ops(self) -> list[PlanInstruction]:
        return [i for i in self.instructions if i.opcode == OpCode.COMPUTE]

    def transfer_ops(self) -> list[PlanInstruction]:
        return [i for i in self.instructions if i.opcode in (OpCode.TRANSFER, OpCode.PREFETCH, OpCode.LOAD)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "fingerprint": self.fingerprint,
            "instructions": [i.as_dict() for i in self.instructions],
            "notes": list(self.notes),
        }


def _tier_for_device(device: str) -> MemoryTier:
    name = device.lower()
    if "disk" in name or "nvme" in name or "pack" in name:
        return MemoryTier.DISK
    if "pinned" in name:
        return MemoryTier.PINNED_RAM
    if any(tok in name for tok in ("cuda", "rocm", "gpu", "xpu", "mps", "vram")):
        return MemoryTier.DEVICE
    if any(tok in name for tok in ("cpu", "numa", "ram", "host")):
        return MemoryTier.SYSTEM_RAM
    return MemoryTier.UNKNOWN


def _transfer_backend(src: str, dst: str) -> str:
    src_tier = _tier_for_device(src)
    dst_tier = _tier_for_device(dst)
    if src_tier == MemoryTier.DISK or dst_tier == MemoryTier.DISK:
        return "disk_pread"
    if src_tier == MemoryTier.DEVICE and dst_tier == MemoryTier.DEVICE:
        return "device_p2p_or_host_staged"
    if MemoryTier.DEVICE in (src_tier, dst_tier):
        return "host_device_copy"
    return "host_memcpy"


def build_executable_schedule(
    plan: ExecutionPlan,
    residency: ResidencySchedule | None = None,
    *,
    streaming: bool = False,
    prefetch_distance: int = 1,
    program: Any | None = None,
) -> ExecutableSchedule:
    """Lower placements + residency into an ordered executable instruction list.

    CPU-only single-device plans emit Compute + Release. Cross-device edges emit
    explicit Transfer ops before the consumer Compute. Streaming plans emit
    Prefetch/Load before Compute when ``streaming`` is True.

    When ``program`` is provided, Compute inputs/outputs and Releases use real
    region tensor ids (not synthetic ``activation::`` names).
    """
    by_id = {p.region_id: p for p in plan.placements}
    transfers = list(residency.transfers) if residency is not None else []
    transfer_before: dict[str, list[ScheduledTransfer]] = {}
    for transfer in transfers:
        transfer_before.setdefault(transfer.before_region, []).append(transfer)

    region_io: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {}
    if program is not None:
        for region in program.regions:
            region_io[region.region_id] = (region.inputs, region.outputs, region.state_inputs)

    instructions: list[PlanInstruction] = []
    last_compute: dict[str, str] = {}
    last_load: dict[str, str] = {}
    state_evicts: list[str | None] = []
    notes = list(plan.notes)
    # Unique transfers emit once; every consumer on the destination waits.
    wait_for_value_dest: dict[tuple[str, str], str] = {}
    emitted_transfers: set[str] = set()

    def _ensure_transfer_ops(transfer: ScheduledTransfer) -> str:
        tname = f"transfer::{transfer.after_region}->{transfer.before_region}:{transfer.value_name}"
        key = (transfer.value_name, transfer.destination_device)
        if key in wait_for_value_dest:
            return wait_for_value_dest[key]
        if tname in emitted_transfers:
            wait_name = f"wait::{tname}"
            wait_for_value_dest[key] = wait_name
            return wait_name
        emitted_transfers.add(tname)
        producer_compute = last_compute.get(transfer.after_region)
        tdeps = (producer_compute,) if producer_compute else ()
        instructions.append(
            PlanInstruction(
                opcode=OpCode.TRANSFER,
                name=tname,
                resource=_transfer_resource(transfer.source_device, transfer.destination_device),
                depends_on=tdeps,
                inputs=(transfer.value_name,),
                outputs=(transfer.value_name,),
                nbytes=transfer.nbytes,
                memory_tier=_tier_for_device(transfer.destination_device),
                predicted_duration_s=0.0,
                source=transfer.source_device,
                destination=transfer.destination_device,
                backend_id="transfer",
                transfer_backend=_transfer_backend(transfer.source_device, transfer.destination_device),
                sync_required=False,
                attributes={
                    "after_region": transfer.after_region,
                    "before_region": transfer.before_region,
                    "simulated_until_validated": True,
                    "mock_transfer_delay_s": 0.08 if "mock" in transfer.destination_device else 0.0,
                },
            )
        )
        record_name = f"record::{tname}"
        instructions.append(
            PlanInstruction(
                opcode=OpCode.RECORD_EVENT,
                name=record_name,
                resource=transfer.source_device,
                depends_on=(tname,),
                inputs=(transfer.value_name,),
                sync_required=False,
                attributes={"pairs_with_wait": f"wait::{tname}", "simulated_until_validated": True},
            )
        )
        wait_name = f"wait::{tname}"
        instructions.append(
            PlanInstruction(
                opcode=OpCode.WAIT_EVENT,
                name=wait_name,
                resource=transfer.destination_device,
                depends_on=(record_name,),
                inputs=(transfer.value_name,),
                sync_required=True,
                attributes={"waits_for": record_name, "simulated_until_validated": True},
            )
        )
        wait_for_value_dest[key] = wait_name
        return wait_name

    # Index transfers by destination consumer and by (value, dest) for sharing.
    transfer_by_value_dest: dict[tuple[str, str], ScheduledTransfer] = {}
    for transfer in transfers:
        transfer_by_value_dest.setdefault((transfer.value_name, transfer.destination_device), transfer)

    for index, placement in enumerate(plan.placements):
        compute_name = f"compute::{placement.region_id}"
        deps: list[str] = []
        # Region-level deps still serialize producers before consumers when no
        # transfer wait is present; waits added below replace same-device edges.
        for dep in placement.depends_on:
            if dep in last_compute and placement.device == by_id[dep].device:
                deps.append(last_compute[dep])

        inputs_t, outputs_t, state_t = region_io.get(
            placement.region_id,
            (
                tuple(f"activation::{d}" for d in placement.depends_on),
                (f"activation::{placement.region_id}",),
                ((f"state::{placement.region_id}",) if placement.state_bytes else ()),
            ),
        )
        if not state_t and placement.state_bytes > 0:
            state_t = (f"state::{placement.region_id}",)

        if placement.state_bytes > 0:
            state_inputs = state_t or (f"state::{placement.region_id}",)
            load_deps: list[str] = list(deps)
            if streaming and prefetch_distance > 0:
                prefetch_name = f"prefetch::{placement.region_id}"
                prefetch_deps_list: list[str] = []
                # Do not race Prefetch ahead of the previous region's Load (would
                # steal staging budget under a single-region RAM cap). After that
                # Load, Prefetch i may overlap Compute i-1 when the budget fits both.
                if index >= 1:
                    prev_id = plan.placements[index - 1].region_id
                    prev_load = last_load.get(prev_id)
                    if prev_load is not None:
                        prefetch_deps_list.append(prev_load)
                # When prefetch_distance > 1, also wait for older Evicts to free slots.
                evict_lead = index - prefetch_distance - 1
                if evict_lead >= 0 and evict_lead < len(state_evicts) and state_evicts[evict_lead]:
                    prefetch_deps_list.append(str(state_evicts[evict_lead]))
                lead = index - prefetch_distance - 1
                if lead >= 0 and plan.placements[lead].region_id in last_compute:
                    prefetch_deps_list.append(last_compute[plan.placements[lead].region_id])
                instructions.append(
                    PlanInstruction(
                        opcode=OpCode.PREFETCH,
                        name=prefetch_name,
                        resource="nvme_or_pack",
                        depends_on=tuple(dict.fromkeys(prefetch_deps_list)),
                        inputs=state_inputs,
                        outputs=state_inputs,
                        nbytes=placement.state_bytes,
                        memory_tier=MemoryTier.DISK,
                        predicted_duration_s=0.0,
                        source="disk",
                        destination=placement.device,
                        backend_id="cpu",
                        transfer_backend="disk_pread",
                        sync_required=False,
                        attributes={"region_id": placement.region_id, "kind": "parameter_prefetch"},
                    )
                )
                load_deps = [prefetch_name]
            load_name = f"load::{placement.region_id}"
            # When streaming, Load waits for previous Evict so only one live set is required.
            if streaming and index >= 1 and index - 1 < len(state_evicts) and state_evicts[index - 1]:
                load_deps.append(str(state_evicts[index - 1]))
            instructions.append(
                PlanInstruction(
                    opcode=OpCode.LOAD,
                    name=load_name,
                    resource=placement.device,
                    depends_on=tuple(dict.fromkeys(load_deps)),
                    inputs=state_inputs,
                    outputs=state_inputs,
                    nbytes=placement.state_bytes,
                    memory_tier=MemoryTier.SYSTEM_RAM,
                    predicted_duration_s=0.0,
                    source="disk",
                    destination=placement.device,
                    backend_id=placement.backend_id,
                    transfer_backend="disk_pread",
                    sync_required=True,
                    attributes={"region_id": placement.region_id, "kind": "parameter_materialize"},
                )
            )
            last_load[placement.region_id] = load_name
            deps.append(load_name)

        # Transfers listed for this consumer, plus any shared (value, dest) copy.
        pending_transfers = list(transfer_before.get(placement.region_id, ()))
        for value_name in inputs_t:
            shared = transfer_by_value_dest.get((value_name, placement.device))
            if shared is not None and shared not in pending_transfers:
                pending_transfers.append(shared)
        for transfer in pending_transfers:
            deps.append(_ensure_transfer_ops(transfer))

        compute_inputs = tuple(n for n in inputs_t if True)
        if state_t:
            # Ensure state ids appear once.
            compute_inputs = tuple(dict.fromkeys(list(compute_inputs) + list(state_t)))

        instructions.append(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name=compute_name,
                resource=placement.device,
                depends_on=tuple(dict.fromkeys(deps)),
                inputs=compute_inputs,
                outputs=outputs_t,
                nbytes=placement.output_bytes,
                memory_tier=_tier_for_device(placement.device),
                predicted_duration_s=placement.estimated_latency_s,
                executable_ref=placement.region_id,
                backend_id=placement.backend_id,
                attributes={
                    "dtype": placement.dtype,
                    "kernel_id": placement.kernel_id,
                    "measured": placement.measured,
                    "state_bytes": placement.state_bytes,
                    "working_set_bytes": placement.working_set_bytes,
                    "mock_compute_delay_s": 0.05 if "mock" in placement.device else 0.0,
                },
            )
        )
        last_compute[placement.region_id] = compute_name

        if streaming and placement.state_bytes > 0 and state_t:
            evict_name = f"evict::state::{placement.region_id}"
            instructions.append(
                PlanInstruction(
                    opcode=OpCode.EVICT,
                    name=evict_name,
                    resource=placement.device,
                    depends_on=(compute_name,),
                    inputs=state_t,
                    outputs=(),
                    nbytes=placement.state_bytes,
                    memory_tier=MemoryTier.SYSTEM_RAM,
                    predicted_duration_s=0.0,
                    destination=placement.device,
                    attributes={"kind": "parameter_evict", "region_id": placement.region_id},
                )
            )
            while len(state_evicts) < index:
                state_evicts.append(None)
            state_evicts.append(evict_name)
        else:
            while len(state_evicts) <= index:
                state_evicts.append(None)

    # Releases: after every consumer of a tensor has computed.
    for producer in plan.placements:
        _, outputs_t, _ = region_io.get(
            producer.region_id,
            ((), (f"activation::{producer.region_id}",), ()),
        )
        for out_name in outputs_t:
            consumers = [p for p in plan.placements if out_name in region_io.get(p.region_id, ((), (), ()))[0]]
            if not consumers and program is None:
                consumers = [p for p in plan.placements if producer.region_id in p.depends_on]
            if not consumers:
                continue
            instructions.append(
                PlanInstruction(
                    opcode=OpCode.RELEASE,
                    name=f"release::{out_name}",
                    resource=producer.device,
                    depends_on=tuple(f"compute::{p.region_id}" for p in consumers),
                    inputs=(out_name,),
                    outputs=(),
                    nbytes=max(0, producer.output_bytes // max(1, len(outputs_t))),
                    memory_tier=_tier_for_device(producer.device),
                    predicted_duration_s=0.0,
                    attributes={
                        "kind": "activation",
                        "producer_region": producer.region_id,
                        "consumer_count": len(consumers),
                        "release_resource": producer.device,
                    },
                )
            )

    if residency is not None:
        notes.extend(n for n in residency.notes if n not in notes)

    schedule = ExecutableSchedule(
        graph_name=plan.graph_name,
        fingerprint=plan.fingerprint,
        instructions=instructions,
        notes=notes,
    )
    assert_schedule_valid(schedule)
    return schedule


class ScheduleValidationError(ValueError):
    """Raised when an :class:`ExecutableSchedule` violates a structural invariant."""


def validate_schedule(schedule: ExecutableSchedule) -> list[str]:
    """Check structural invariants a runtime/simulator must be able to rely on.

    Returns a list of human-readable violations; empty means the schedule is
    safe to simulate or execute. Never silently drops or reorders instructions
    to "fix" a bad schedule -- planner bugs must surface, not be papered over.
    """
    errors: list[str] = []

    by_name: dict[str, PlanInstruction] = {}
    for inst in schedule.instructions:
        if inst.name in by_name:
            errors.append(f"duplicate instruction id: {inst.name!r}")
        else:
            by_name[inst.name] = inst

    known_opcodes = {
        OpCode.PREFETCH,
        OpCode.LOAD,
        OpCode.TRANSFER,
        OpCode.RECORD_EVENT,
        OpCode.WAIT_EVENT,
        OpCode.COMPUTE,
        OpCode.EVICT,
        OpCode.RELEASE,
    }
    recorded_events: set[str] = set()
    for inst in schedule.instructions:
        if inst.opcode not in known_opcodes:
            errors.append(f"unknown instruction opcode {inst.opcode!r} on {inst.name!r}")
        for dep in inst.depends_on:
            if dep not in by_name:
                errors.append(f"{inst.name!r} depends on unknown instruction {dep!r}")
        if inst.opcode == OpCode.RECORD_EVENT:
            recorded_events.add(inst.name)
        if inst.opcode == OpCode.TRANSFER:
            if not inst.source or not inst.destination:
                errors.append(f"transfer {inst.name!r} missing source or destination")
            if inst.source and inst.destination and inst.source == inst.destination:
                # Same-resource transfer is allowed only as an explicit no-op path.
                pass
            if not (inst.inputs or inst.outputs):
                errors.append(f"transfer {inst.name!r} references no tensors")
            for tid in (*inst.inputs, *inst.outputs):
                if not tid:
                    errors.append(f"transfer {inst.name!r} has empty tensor id")
        if inst.opcode == OpCode.COMPUTE:
            if not inst.resource:
                errors.append(f"compute {inst.name!r} missing resource")
            if not inst.executable_ref:
                errors.append(f"compute {inst.name!r} missing executable_ref")
            for tid in (*inst.inputs, *inst.outputs):
                if not tid:
                    errors.append(f"compute {inst.name!r} has empty tensor id")
        if inst.opcode in (OpCode.LOAD, OpCode.PREFETCH, OpCode.EVICT, OpCode.RELEASE):
            for tid in (*inst.inputs, *inst.outputs):
                if tid == "":
                    errors.append(f"{inst.opcode.value} {inst.name!r} has empty tensor id")
            if inst.opcode == OpCode.LOAD and not (inst.inputs or inst.outputs):
                errors.append(f"load {inst.name!r} references no tensors")
            if inst.opcode == OpCode.RELEASE and not inst.inputs:
                errors.append(f"release {inst.name!r} references no tensors")
            if inst.opcode == OpCode.EVICT and not inst.inputs:
                errors.append(f"evict {inst.name!r} references no tensors")

    # Kahn's algorithm: any node left unresolved after removing satisfiable
    # nodes is part of (or depends on) a cycle.
    indegree = {name: len(inst.depends_on) for name, inst in by_name.items()}
    dependents: dict[str, list[str]] = {name: [] for name in by_name}
    for name, inst in by_name.items():
        for dep in inst.depends_on:
            if dep in dependents:
                dependents[dep].append(name)
    ready = deque(name for name, deg in indegree.items() if deg == 0)
    order: list[str] = []
    remaining_indegree = dict(indegree)
    while ready:
        name = ready.popleft()
        order.append(name)
        for nxt in dependents[name]:
            remaining_indegree[nxt] -= 1
            if remaining_indegree[nxt] == 0:
                ready.append(nxt)
    if len(order) != len(by_name):
        cyclic = sorted(set(by_name) - set(order))
        errors.append(f"dependency cycle involves: {cyclic}")
        # Cycle makes ancestor/order-based checks below meaningless; stop here.
        return errors

    position = {name: i for i, name in enumerate(order)}

    def ancestors(name: str) -> set[str]:
        seen: set[str] = set()
        stack = list(by_name[name].depends_on)
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in by_name:
                continue
            seen.add(cur)
            stack.extend(by_name[cur].depends_on)
        return seen

    # Key by (tensor, destination resource): a GPU transfer must not force a CPU
    # consumer of the same logical tensor to wait (multi-copy residency).
    # Transfers without a destination apply to every reader of the tensor.
    transfer_completion_for: dict[tuple[str, str], str] = {}
    for name, inst in by_name.items():
        if inst.opcode == OpCode.WAIT_EVENT:
            dest = inst.resource
            for out in inst.inputs:
                transfer_completion_for.setdefault((out, dest), name)
        elif inst.opcode == OpCode.TRANSFER:
            dest = str(inst.destination or "")
            for out in inst.outputs or inst.inputs:
                transfer_completion_for.setdefault((out, dest), name)

    consumers_by_tensor: dict[str, list[str]] = {}
    for name, inst in by_name.items():
        for value in inst.inputs:
            consumers_by_tensor.setdefault(value, []).append(name)

    for name, inst in by_name.items():
        if inst.opcode == OpCode.WAIT_EVENT:
            waits_for = str((inst.attributes or {}).get("waits_for") or "")
            if not waits_for and inst.depends_on:
                # Convention: first dependency is the RecordEvent.
                candidate = inst.depends_on[0]
                if by_name.get(candidate) and by_name[candidate].opcode == OpCode.RECORD_EVENT:
                    waits_for = candidate
            if waits_for and waits_for not in recorded_events and waits_for not in by_name:
                errors.append(f"wait {name!r} references unknown event {waits_for!r}")
            elif waits_for and waits_for in by_name and by_name[waits_for].opcode != OpCode.RECORD_EVENT:
                errors.append(f"wait {name!r} waits for non-RecordEvent {waits_for!r}")
            elif waits_for and waits_for not in recorded_events:
                errors.append(f"wait {name!r} for event that is never recorded: {waits_for!r}")
            elif not waits_for:
                errors.append(f"wait {name!r} has no RecordEvent target")
        if inst.opcode == OpCode.RELEASE:
            for value in inst.inputs:
                for consumer in consumers_by_tensor.get(value, ()):
                    if consumer == name:
                        continue
                    if consumer not in ancestors(name) and position[consumer] > position[name]:
                        errors.append(f"release {name!r} of {value!r} happens before consumer {consumer!r}")
        if inst.opcode == OpCode.COMPUTE:
            for value in inst.inputs:
                completion = transfer_completion_for.get((value, inst.resource))
                if completion is None:
                    completion = transfer_completion_for.get((value, ""))
                if completion is not None and completion not in ancestors(name):
                    errors.append(
                        f"compute {name!r} reads {value!r} without depending on transfer completion {completion!r}"
                    )
                # Strict residency for activations: a Compute-produced tensor must
                # have a local producer or an inbound Transfer/Load to this resource.
                # User inputs (no Compute producer) may be seeded on the host.
                producers = [
                    pname
                    for pname, pinst in by_name.items()
                    if value in (pinst.outputs or ())
                    or (pinst.opcode == OpCode.TRANSFER and value in (pinst.outputs or pinst.inputs))
                    or (pinst.opcode == OpCode.LOAD and value in (pinst.outputs or pinst.inputs))
                ]
                activation_producers = [pname for pname in producers if by_name[pname].opcode == OpCode.COMPUTE]
                if not activation_producers:
                    continue
                local_ok = False
                for pname in producers:
                    pinst = by_name[pname]
                    if pinst.opcode == OpCode.COMPUTE and pinst.resource == inst.resource:
                        local_ok = True
                        break
                    if pinst.opcode == OpCode.TRANSFER and str(pinst.destination or "") == inst.resource:
                        local_ok = True
                        break
                    if pinst.opcode == OpCode.LOAD and str(pinst.destination or pinst.resource) == inst.resource:
                        local_ok = True
                        break
                if not local_ok:
                    errors.append(
                        f"compute {name!r} requires copy of {value!r} on {inst.resource!r} "
                        f"but schedule only produces it elsewhere (no silent reuse)"
                    )

    return errors


def validate_schedule_resources(schedule: ExecutableSchedule, machine: Any) -> list[str]:
    """Check that every Compute instruction names a real compute resource.

    Transfer/Prefetch ``resource`` labels are synthetic engine identifiers
    (e.g. ``copy_engine:src->dst``), not resource-graph ids, so only Compute
    placements -- which must run on an actual discovered device -- are
    checked here. ``machine`` is a :class:`~streamcompiler.ir.resource_graph.ResourceGraph`.
    """
    errors: list[str] = []
    known = set(machine.compute)
    for inst in schedule.compute_ops():
        if inst.resource not in known:
            errors.append(f"compute {inst.name!r} references unknown compute resource {inst.resource!r}")
    return errors


def assert_schedule_valid(schedule: ExecutableSchedule) -> None:
    errors = validate_schedule(schedule)
    if errors:
        raise ScheduleValidationError(f"ExecutableSchedule {schedule.graph_name!r} failed validation: {errors}")


def _transfer_resource(source: str, destination: str) -> str:
    return f"copy_engine:{source}->{destination}"


def schedule_matches_plan(schedule: ExecutableSchedule, plan: ExecutionPlan) -> list[str]:
    """Return mismatch reasons; empty means schedule covers every placement."""
    compute_refs = {i.executable_ref for i in schedule.compute_ops()}
    errors: list[str] = []
    for placement in plan.placements:
        if placement.region_id not in compute_refs:
            errors.append(f"missing compute for {placement.region_id}")
    for inst in schedule.compute_ops():
        if inst.executable_ref not in {p.region_id for p in plan.placements}:
            errors.append(f"orphan compute {inst.name}")
    return errors


def placements_from_schedule(schedule: ExecutableSchedule) -> list[Placement]:
    """Rebuild placements from Compute ops (simulator/runtime plan consistency)."""
    by_name = {i.name: i for i in schedule.instructions}
    placements: list[Placement] = []
    for inst in schedule.compute_ops():
        depends: list[str] = []
        for dep_name in inst.depends_on:
            dep = by_name.get(dep_name)
            if dep is None:
                continue
            if dep.opcode == OpCode.COMPUTE and dep.executable_ref:
                depends.append(dep.executable_ref)
            elif dep.opcode in (OpCode.TRANSFER, OpCode.WAIT_EVENT, OpCode.RECORD_EVENT):
                after = dep.attributes.get("after_region")
                if after:
                    depends.append(str(after))
                # Walk one hop for wait→record→transfer.
                for nested_name in dep.depends_on:
                    nested = by_name.get(nested_name)
                    if nested is None:
                        continue
                    after = nested.attributes.get("after_region")
                    if after:
                        depends.append(str(after))
                    for nested2_name in nested.depends_on:
                        nested2 = by_name.get(nested2_name)
                        if nested2 is not None and nested2.attributes.get("after_region"):
                            depends.append(str(nested2.attributes["after_region"]))
            elif dep.opcode == OpCode.LOAD:
                continue
        attrs = inst.attributes
        placements.append(
            Placement(
                region_id=str(inst.executable_ref),
                device=inst.resource,
                backend_id=inst.backend_id or "cpu",
                dtype=str(attrs.get("dtype", "float32")),
                kernel_id=str(attrs.get("kernel_id", "unknown")),
                estimated_latency_s=inst.predicted_duration_s,
                depends_on=tuple(dict.fromkeys(depends)),
                measured=bool(attrs.get("measured", False)),
                output_bytes=inst.nbytes,
                state_bytes=int(attrs.get("state_bytes", 0)),
            )
        )
    return placements


def schedule_from_bindings(
    program: Any,
    bindings: dict[str, Any],
    *,
    streaming: bool = False,
    prefetch_distance: int = 0,
    fingerprint: str = "",
) -> ExecutableSchedule:
    """Build an ExecutableSchedule from region bindings when no plan schedule exists.

    Used so GraphExecutor never falls back to a region topo walker: every run
    goes through the instruction DAG.
    """
    from streamcompiler.runtime.residency import attach_residency_to_plan

    output_producer: dict[str, str] = {}
    for region in program.regions:
        for name in region.outputs:
            output_producer[name] = region.region_id

    placements: list[Placement] = []
    for region in program.regions:
        binding = bindings[region.region_id]
        deps: list[str] = []
        for inp in region.inputs:
            if inp in program.state_bindings:
                continue
            producer = output_producer.get(inp)
            if producer is not None and producer not in deps:
                deps.append(producer)
        state_bytes = 0
        state_map = program.state_tensors() if hasattr(program, "state_tensors") else {}
        for name in region.state_inputs:
            tensor = state_map.get(name) if isinstance(state_map, dict) else None
            if tensor is not None and hasattr(tensor, "numel"):
                state_bytes += int(tensor.numel() * tensor.element_size())
            else:
                # Unknown size: still force a Load so Compute never materializes weights.
                state_bytes = max(state_bytes, 1)
        if region.state_inputs and state_bytes <= 0:
            state_bytes = 1
        placements.append(
            Placement(
                region_id=region.region_id,
                device=binding.device,
                backend_id=binding.backend_id,
                dtype="float32",
                kernel_id="eager",
                estimated_latency_s=0.0,
                depends_on=tuple(deps),
                measured=False,
                output_bytes=0,
                state_bytes=state_bytes,
            )
        )
    # Placement list order must be producer-before-consumer so schedule emission
    # can resolve last_compute when wiring Transfer deps (region list may be shuffled).
    by_id = {p.region_id: p for p in placements}
    remaining = {p.region_id for p in placements}
    ordered: list[Placement] = []
    while remaining:
        progressed = False
        for rid in list(remaining):
            dep_ids = by_id[rid].depends_on
            if all(d not in remaining for d in dep_ids):
                ordered.append(by_id[rid])
                remaining.remove(rid)
                progressed = True
        if not progressed:
            # Cycle / missing dep — append rest in original order.
            ordered.extend(by_id[rid] for rid in list(remaining))
            break
    placements = ordered
    devices = tuple(dict.fromkeys(p.device for p in placements))
    plan = ExecutionPlan(
        graph_name=program.graph_name,
        fingerprint=fingerprint or "bindings",
        objective="latency",
        placements=placements,
        decisions=[],
        devices_used=devices,
        communication_backend="none",
        predicted_latency_s=0.0,
        strategy="bindings_schedule",
        notes=["schedule rebuilt from region bindings"],
    )
    residency = attach_residency_to_plan(plan, program)
    return build_executable_schedule(
        plan,
        residency,
        streaming=streaming,
        prefetch_distance=prefetch_distance,
        program=program,
    )
