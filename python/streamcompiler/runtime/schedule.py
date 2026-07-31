"""One executable plan format shared by planner, simulator, and runtime.

Instructions are explicit memory/compute ops. The simulator must not invent
schedules the runtime cannot perform: both consume :class:`ExecutableSchedule`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any

from streamcompiler.ir.graph import OpCode
from streamcompiler.planner.maximal import ExecutionPlan, Placement
from streamcompiler.runtime.residency import ResidencySchedule, ScheduledTransfer


class MemoryTier(str, Enum):
    DISK = "disk"
    SYSTEM_RAM = "system_ram"
    PINNED_RAM = "pinned_ram"
    NUMA_RAM = "numa_ram"
    DEVICE = "device"
    """Virtual accelerator or future GPU VRAM — never created by Load alone."""
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FrozenAttrs:
    """Picklable immutable string-key mapping for instruction attributes."""

    _items: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, attrs: Mapping[str, Any] | None) -> FrozenAttrs:
        if isinstance(attrs, FrozenAttrs):
            return attrs
        items = tuple(sorted(((str(k), v) for k, v in dict(attrs or {}).items()), key=lambda kv: kv[0]))
        return cls(items)

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self._items:
            if k == key:
                return v
        return default

    def __getitem__(self, key: str) -> Any:
        for k, v in self._items:
            if k == key:
                return v
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return any(k == key for k, _ in self._items)

    def __iter__(self) -> Iterator[str]:
        return (k for k, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self._items)

    def values(self) -> tuple[Any, ...]:
        return tuple(v for _, v in self._items)

    def items(self) -> tuple[tuple[str, Any], ...]:
        return self._items

    def as_dict(self) -> dict[str, Any]:
        return dict(self._items)


@dataclass(frozen=True)
class PlanInstruction:
    """One scheduled op the runtime can execute and the simulator can cost.

    Immutable: no futures, tensors, timestamps, or runtime handles may be stored
    in ``attributes``. Per-call state lives in :class:`ExecutionContext`.
    """

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
    stream_id: str | None = None
    """Ordered stream on resource (compute / copy / io / sync)."""
    copy_engine_id: str | None = None
    """Copy-engine identity for Transfer / Prefetch / Load."""
    link_id: str | None = None
    """Interconnect identity for Transfer."""
    io_queue_id: str | None = None
    """Disk / pack I/O queue identity for Prefetch / Load."""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "attributes", FrozenAttrs.from_mapping(self.attributes))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["opcode"] = self.opcode.value
        payload["memory_tier"] = self.memory_tier.value
        payload["attributes"] = dict(self.attributes)
        return payload


@dataclass(frozen=True)
class ExecutableSchedule:
    """Immutable executable plan: same object for plan explain, sim, and run."""

    graph_name: str
    fingerprint: str
    instructions: tuple[PlanInstruction, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "notes", tuple(self.notes))

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


def default_stream_id(opcode: OpCode, resource: str) -> str:
    res = resource or "unknown"
    if opcode == OpCode.COMPUTE:
        return f"{res}::compute"
    if opcode in (OpCode.TRANSFER, OpCode.PREFETCH, OpCode.LOAD):
        return f"{res}::copy0"
    if opcode in (OpCode.RECORD_EVENT, OpCode.WAIT_EVENT):
        return f"{res}::sync"
    return f"{res}::lifetime"


def with_explicit_streams(inst: PlanInstruction) -> PlanInstruction:
    """Fill stream / copy-engine / link / I/O-queue ids when the planner omitted them."""
    stream_id = inst.stream_id or default_stream_id(inst.opcode, inst.resource)
    copy_engine_id = inst.copy_engine_id
    link_id = inst.link_id
    io_queue_id = inst.io_queue_id
    if inst.opcode in (OpCode.TRANSFER, OpCode.PREFETCH, OpCode.LOAD) and not copy_engine_id:
        copy_engine_id = f"{inst.resource or 'unknown'}::copy0"
    if inst.opcode == OpCode.TRANSFER and not link_id:
        src = inst.source or "unknown"
        dst = inst.destination or inst.resource or "unknown"
        link_id = f"{src}->{dst}"
    if inst.opcode in (OpCode.PREFETCH, OpCode.LOAD) and not io_queue_id:
        io_queue_id = f"{inst.resource or 'unknown'}::io0"
    if (
        stream_id == inst.stream_id
        and copy_engine_id == inst.copy_engine_id
        and link_id == inst.link_id
        and io_queue_id == inst.io_queue_id
    ):
        return inst
    return replace(
        inst,
        stream_id=stream_id,
        copy_engine_id=copy_engine_id,
        link_id=link_id,
        io_queue_id=io_queue_id,
    )


def ensure_explicit_streams(schedule: ExecutableSchedule) -> ExecutableSchedule:
    """Return schedule with every instruction carrying explicit stream resources."""
    new_insts = tuple(with_explicit_streams(i) for i in schedule.instructions)
    if new_insts == schedule.instructions:
        return schedule
    return replace(schedule, instructions=new_insts)


def with_instruction_attributes(
    schedule: ExecutableSchedule,
    updates: Mapping[str, Mapping[str, Any]],
) -> ExecutableSchedule:
    """Return a new schedule with merged instruction attributes (tests / tooling)."""
    if not updates:
        return schedule
    new_insts: list[PlanInstruction] = []
    for inst in schedule.instructions:
        patch = updates.get(inst.name)
        if patch is None:
            new_insts.append(inst)
            continue
        merged = {**dict(inst.attributes), **dict(patch)}
        new_insts.append(replace(inst, attributes=merged))
    return replace(schedule, instructions=tuple(new_insts))


def _tier_for_device(device: str) -> MemoryTier:
    name = device.lower()
    if "disk" in name or "nvme" in name or "pack" in name:
        return MemoryTier.DISK
    if "pinned" in name:
        return MemoryTier.PINNED_RAM
    if any(tok in name for tok in ("cuda", "rocm", "gpu", "vram", "mock")):
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


def _first_pinned_host(machine: Any | None) -> str | None:
    """Return first PINNED_HOST memory resource name on ``machine``, if any."""
    if machine is None:
        return None
    memories = getattr(machine, "memory", None) or {}
    for name, mem in memories.items():
        mclass = getattr(mem, "memory_class", None)
        mname = getattr(mclass, "value", mclass)
        if str(mname) == "pinned_host" or "pinned" in str(name).lower():
            return str(name)
    return None


def _load_host_for_destination(dest: str, *, machine: Any | None = None) -> tuple[str, MemoryTier]:
    """Host staging resource for a Load that feeds ``dest``.

    Device destinations prefer pinned host RAM when the machine exposes it so
    host→device copies can use page-locked staging.
    """
    if _tier_for_device(dest) != MemoryTier.DEVICE:
        return dest, _tier_for_device(dest)
    pinned = _first_pinned_host(machine)
    if pinned is not None:
        return pinned, MemoryTier.PINNED_RAM
    return "cpu", MemoryTier.SYSTEM_RAM


def _transfer_is_simulated(source: str, destination: str) -> bool:
    """Mock/virtual/unknown paths stay simulated; known CPU/CUDA DMA is executable."""
    blob = f"{source}|{destination}".lower()
    if "mock" in blob or "virtual" in blob or "simulated" in blob:
        return True
    real_tokens = ("cuda_", "rocm_", "cpu", "numa", "pinned", "host", "vram")

    def _known_real(name: str) -> bool:
        lower = name.lower()
        return any(tok in lower for tok in real_tokens)

    return not (_known_real(source) and _known_real(destination))


def _state_tensors_without_later_use(
    state_names: tuple[str, ...],
    *,
    placements: list[Placement],
    start_index: int,
    region_io: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]],
) -> tuple[str, ...]:
    """Return state tensors that no later region lists in ``state_inputs``.

    Shared parameters/buffers stay resident until their last consumer so a
    subsequent Load/Compute is not left pointing at an already-evicted copy.
    """
    later: set[str] = set()
    for later_placement in placements[start_index + 1 :]:
        later.update(region_io.get(later_placement.region_id, ((), (), ()))[2])
    return tuple(name for name in state_names if name not in later)


def build_executable_schedule(
    plan: ExecutionPlan,
    residency: ResidencySchedule | None = None,
    *,
    streaming: bool = False,
    prefetch_distance: int = 1,
    program: Any | None = None,
    activation_budget_bytes: int | None = None,
    machine: Any | None = None,
) -> ExecutableSchedule:
    """Lower placements + residency into an ordered executable instruction list.

    CPU-only single-device plans emit Compute + Release. Cross-device edges emit
    explicit Transfer ops before the consumer Compute. Streaming plans emit
    Prefetch/Load before Compute when ``streaming`` is True.

    When ``program`` is provided, Compute inputs/outputs and Releases use real
    region tensor ids (not synthetic ``activation::`` names).

    When ``activation_budget_bytes`` is set, emit explicit activation Evict
    (RAM→disk) and Load (disk→RAM) instructions — never runtime-transparent spill.
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

    def _tensor_sizes(names: tuple[str, ...], total_bytes: int = 0) -> dict[str, int]:
        sizes: dict[str, int] = {}
        if program is not None:
            values = getattr(program, "values", {})
            for name in names:
                spec = values.get(name)
                nbytes = int(getattr(spec, "nbytes", 0) or 0) if spec is not None else 0
                if nbytes > 0:
                    sizes[str(name)] = nbytes
        missing = [name for name in names if str(name) not in sizes]
        if len(missing) == 1 and total_bytes > sum(sizes.values()):
            sizes[str(missing[0])] = int(total_bytes - sum(sizes.values()))
        return sizes

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
                    "simulated_until_validated": _transfer_is_simulated(
                        transfer.source_device, transfer.destination_device
                    ),
                    "mock_transfer_delay_s": 0.08 if "mock" in transfer.destination_device else 0.0,
                    "tensor_nbytes": {transfer.value_name: int(transfer.nbytes)},
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
                attributes={
                    "pairs_with_wait": f"wait::{tname}",
                    "simulated_until_validated": _transfer_is_simulated(
                        transfer.source_device, transfer.destination_device
                    ),
                },
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
                attributes={
                    "waits_for": record_name,
                    "simulated_until_validated": _transfer_is_simulated(
                        transfer.source_device, transfer.destination_device
                    ),
                },
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
        state_sizes = _tensor_sizes(state_t, placement.state_bytes)

        if placement.state_bytes > 0 and streaming:
            # Streaming: Prefetch/Load/Evict own RAM. Resident packs register
            # initial residency on the artifact — no fake runtime Load ops.
            state_inputs = state_t or (f"state::{placement.region_id}",)
            load_deps: list[str] = list(deps)
            if prefetch_distance > 0:
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
                        attributes={
                            "region_id": placement.region_id,
                            "kind": "parameter_prefetch",
                            "tensor_nbytes": state_sizes,
                        },
                    )
                )
                load_deps = [prefetch_name]
            load_name = f"load::{placement.region_id}"
            # When streaming, Load waits for previous Evict so only one live set is required.
            if streaming and index >= 1 and index - 1 < len(state_evicts) and state_evicts[index - 1]:
                load_deps.append(str(state_evicts[index - 1]))
            # Load is always disk → host RAM (prefer pinned when feeding a device).
            # Device residency needs an explicit Transfer after this Load.
            load_host, load_tier = _load_host_for_destination(placement.device, machine=machine)
            instructions.append(
                PlanInstruction(
                    opcode=OpCode.LOAD,
                    name=load_name,
                    resource=load_host,
                    depends_on=tuple(dict.fromkeys(load_deps)),
                    inputs=state_inputs,
                    outputs=state_inputs,
                    nbytes=placement.state_bytes,
                    memory_tier=load_tier,
                    predicted_duration_s=0.0,
                    source="disk",
                    destination=load_host,
                    backend_id=placement.backend_id,
                    transfer_backend="disk_pread",
                    sync_required=True,
                    attributes={
                        "region_id": placement.region_id,
                        "kind": "parameter_materialize",
                        "tensor_nbytes": state_sizes,
                    },
                )
            )
            last_load[placement.region_id] = load_name
            deps.append(load_name)
            if load_host != placement.device:
                for state_name in state_inputs:
                    tname = f"transfer::state::{state_name}->{placement.device}"
                    if tname not in emitted_transfers:
                        emitted_transfers.add(tname)
                        instructions.append(
                            PlanInstruction(
                                opcode=OpCode.TRANSFER,
                                name=tname,
                                resource=_transfer_resource(load_host, placement.device),
                                depends_on=(load_name,),
                                inputs=(state_name,),
                                outputs=(state_name,),
                                nbytes=max(1, int(state_sizes.get(state_name, 0) or 0)),
                                memory_tier=_tier_for_device(placement.device),
                                predicted_duration_s=0.0,
                                source=load_host,
                                destination=placement.device,
                                backend_id="transfer",
                                transfer_backend=_transfer_backend(load_host, placement.device),
                                sync_required=False,
                                attributes={
                                    "region_id": placement.region_id,
                                    "kind": "parameter_host_to_device",
                                    "simulated_until_validated": _transfer_is_simulated(load_host, placement.device),
                                    "mock_transfer_delay_s": 0.08 if "mock" in placement.device else 0.0,
                                    "tensor_nbytes": {state_name: max(1, int(state_sizes.get(state_name, 0) or 0))},
                                },
                            )
                        )
                    deps.append(tname)

        elif placement.state_bytes > 0 and _tier_for_device(placement.device) == MemoryTier.DEVICE:
            # Resident pack: weights already on host compute RAM — Transfer that
            # host→device (not pinned staging; nothing was Loaded onto pinned).
            state_inputs = state_t or (f"state::{placement.region_id}",)
            load_host = "cpu"
            for p in plan.placements:
                if any(tok in p.device.lower() for tok in ("cpu", "numa", "host")):
                    load_host = p.device
                    break
            for state_name in state_inputs:
                tname = f"transfer::state::{state_name}->{placement.device}"
                if tname not in emitted_transfers:
                    emitted_transfers.add(tname)
                    instructions.append(
                        PlanInstruction(
                            opcode=OpCode.TRANSFER,
                            name=tname,
                            resource=_transfer_resource(load_host, placement.device),
                            depends_on=(),
                            inputs=(state_name,),
                            outputs=(state_name,),
                            nbytes=max(1, int(state_sizes.get(state_name, 0) or 0)),
                            memory_tier=_tier_for_device(placement.device),
                            predicted_duration_s=0.0,
                            source=load_host,
                            destination=placement.device,
                            backend_id="transfer",
                            transfer_backend=_transfer_backend(load_host, placement.device),
                            sync_required=False,
                            attributes={
                                "region_id": placement.region_id,
                                "kind": "parameter_host_to_device",
                                "simulated_until_validated": _transfer_is_simulated(load_host, placement.device),
                                "mock_transfer_delay_s": 0.08 if "mock" in placement.device else 0.0,
                                "tensor_nbytes": {state_name: max(1, int(state_sizes.get(state_name, 0) or 0))},
                            },
                        )
                    )
                deps.append(tname)

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

        tensor_nbytes: dict[str, int] = {}
        if program is not None:
            for name in (*compute_inputs, *outputs_t):
                spec = getattr(program, "values", {}).get(name)
                n = int(getattr(spec, "nbytes", 0) or 0) if spec is not None else 0
                if n > 0:
                    tensor_nbytes[str(name)] = n
        input_bytes = {k: tensor_nbytes[k] for k in compute_inputs if k in tensor_nbytes}
        output_bytes = {k: tensor_nbytes[k] for k in outputs_t if k in tensor_nbytes}

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
                    "workspace_bytes": int(getattr(placement, "workspace_bytes", 0) or 0),
                    "staging_bytes": 0,
                    "input_bytes": input_bytes,
                    "output_bytes": output_bytes,
                    "mock_compute_delay_s": 0.05 if "mock" in placement.device else 0.0,
                    "tensor_nbytes": tensor_nbytes,
                },
            )
        )
        last_compute[placement.region_id] = compute_name

        if streaming and placement.state_bytes > 0 and state_t:
            evict_tensors = _state_tensors_without_later_use(
                state_t,
                placements=plan.placements,
                start_index=index,
                region_io=region_io,
            )
            if evict_tensors:
                evict_nbytes = {str(n): int(state_sizes.get(str(n), 0) or 0) for n in evict_tensors}
                evict_name = f"evict::state::{placement.region_id}"
                instructions.append(
                    PlanInstruction(
                        opcode=OpCode.EVICT,
                        name=evict_name,
                        resource=placement.device,
                        depends_on=(compute_name,),
                        inputs=evict_tensors,
                        outputs=(),
                        nbytes=sum(evict_nbytes.values()) or placement.state_bytes,
                        memory_tier=MemoryTier.SYSTEM_RAM,
                        predicted_duration_s=0.0,
                        destination=placement.device,
                        attributes={
                            "kind": "parameter_evict",
                            "region_id": placement.region_id,
                            "tensor_nbytes": evict_nbytes,
                        },
                    )
                )
                while len(state_evicts) < index:
                    state_evicts.append(None)
                state_evicts.append(evict_name)
            else:
                while len(state_evicts) <= index:
                    state_evicts.append(None)
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
            exact = 0
            if program is not None:
                spec = getattr(program, "values", {}).get(out_name)
                exact = int(getattr(spec, "nbytes", 0) or 0) if spec is not None else 0
            if exact <= 0:
                # Prefer producer Compute tensor_nbytes metadata over equal-split.
                compute_name = f"compute::{producer.region_id}"
                for existing in instructions:
                    if existing.name == compute_name:
                        raw = existing.attributes.get("tensor_nbytes") or existing.attributes.get("output_bytes")
                        if isinstance(raw, dict) and out_name in raw:
                            exact = int(raw[out_name] or 0)
                        break
            if exact <= 0:
                exact = max(0, int(producer.output_bytes or 0)) if len(outputs_t) == 1 else 0
            release_deps: list[str] = [f"compute::{p.region_id}" for p in consumers]
            # Async liveness: hold producer residency until Transfers that read it
            # finish (RecordEvent) and consumer Waits that consume it complete.
            for existing in instructions:
                if out_name not in existing.inputs and out_name not in existing.outputs:
                    continue
                if (
                    existing.opcode == OpCode.RECORD_EVENT
                    and out_name in existing.inputs
                    or existing.opcode == OpCode.WAIT_EVENT
                    and out_name in existing.inputs
                ):
                    release_deps.append(existing.name)
            instructions.append(
                PlanInstruction(
                    opcode=OpCode.RELEASE,
                    name=f"release::{out_name}",
                    resource=producer.device,
                    depends_on=tuple(dict.fromkeys(release_deps)),
                    inputs=(out_name,),
                    outputs=(),
                    nbytes=exact,
                    memory_tier=_tier_for_device(producer.device),
                    predicted_duration_s=0.0,
                    attributes={
                        "kind": "activation",
                        "producer_region": producer.region_id,
                        "consumer_count": len(consumers),
                        "release_resource": producer.device,
                        "tensor_nbytes": {out_name: exact} if exact > 0 else {},
                    },
                )
            )

    if residency is not None:
        notes.extend(n for n in residency.notes if n not in notes)

    schedule = ExecutableSchedule(
        graph_name=plan.graph_name,
        fingerprint=plan.fingerprint,
        instructions=tuple(instructions),
        notes=tuple(notes),
    )
    if activation_budget_bytes is not None and activation_budget_bytes >= 0:
        protected: set[str] = set()
        if program is not None:
            for kind, ref in getattr(program, "output_refs", ()):
                if kind == "value":
                    protected.add(str(ref))
            protected.update(getattr(program, "user_inputs", ()))
        schedule = plan_activation_spills(
            schedule,
            budget_bytes=int(activation_budget_bytes),
            protected_tensors=frozenset(protected),
            program=program,
            machine=machine,
        )
    from streamcompiler.ir.liveness import apply_schedule_liveness

    schedule = apply_schedule_liveness(schedule)
    schedule = ensure_explicit_streams(schedule)
    assert_schedule_valid(schedule)
    return schedule


def plan_activation_spills(
    schedule: ExecutableSchedule,
    *,
    budget_bytes: int,
    protected_tensors: frozenset[str] = frozenset(),
    program: Any | None = None,
    machine: Any | None = None,
) -> ExecutableSchedule:
    """Insert explicit activation Evict/Load ops so live RAM stays within budget.

    Spill = Evict RAM→disk. Reload = Load disk→host RAM. Simulator and runtime
    both execute these schedule ops; neither invents unscheduled I/O.
    """
    if budget_bytes < 0:
        return schedule

    # Next consumer index per tensor (instruction ordinal in topo-ish list order).
    use_sites: dict[str, list[int]] = {}
    for idx, inst in enumerate(schedule.instructions):
        if inst.opcode == OpCode.COMPUTE:
            for name in inst.inputs:
                use_sites.setdefault(name, []).append(idx)

    live: dict[str, tuple[str, int]] = {}  # tensor -> (resource, nbytes)
    spilled: set[str] = set()
    spill_op_for: dict[str, str] = {}  # tensor -> latest spill instruction name
    spill_nbytes: dict[str, int] = {}
    shared_reload: dict[str, str] = {}  # tensor -> Load op that restored host copy
    out: list[PlanInstruction] = []
    notes = list(schedule.notes)
    spill_count = 0
    reload_count = 0

    def _live_bytes() -> int:
        return sum(n for _, n in live.values())

    value_nbytes: dict[str, int] = {}
    if program is not None:
        for name, spec in getattr(program, "values", {}).items():
            n = int(getattr(spec, "nbytes", 0) or 0)
            if n > 0:
                value_nbytes[str(name)] = n

    def _tensor_nbytes(inst: PlanInstruction, tensor: str) -> int:
        if tensor in value_nbytes:
            return value_nbytes[tensor]
        raw = inst.attributes.get("tensor_nbytes") or inst.attributes.get("output_bytes")
        if isinstance(raw, dict) and tensor in raw:
            return max(1, int(raw[tensor] or 1))
        outs = inst.outputs or ()
        if len(outs) == 1 and outs[0] == tensor:
            return max(1, int(inst.nbytes or 1))
        return max(1, int(inst.nbytes or 1)) if not outs else max(1, int(value_nbytes.get(tensor, 1)))

    for idx, inst in enumerate(schedule.instructions):
        if inst.opcode == OpCode.COMPUTE:
            reload_deps: list[str] = []
            for tensor in inst.inputs:
                # Every post-spill consumer must wait on the shared reload —
                # "already live" alone is not a schedule edge under concurrency.
                if tensor in shared_reload:
                    reload_deps.append(shared_reload[tensor])
                    continue
                if tensor in live and tensor not in spilled:
                    continue
                if tensor not in spilled:
                    continue
                load_name = f"load::spill::{tensor}::{inst.name}"
                host, host_tier = _load_host_for_destination(str(inst.resource), machine=machine)
                nbytes = spill_nbytes.get(tensor, value_nbytes.get(tensor, max(1, int(inst.nbytes or 1))))
                spill_dep = spill_op_for.get(tensor)
                out.append(
                    PlanInstruction(
                        opcode=OpCode.LOAD,
                        name=load_name,
                        resource=host,
                        depends_on=(spill_dep,) if spill_dep else (),
                        inputs=(tensor,),
                        outputs=(tensor,),
                        nbytes=nbytes,
                        memory_tier=host_tier,
                        source="disk",
                        destination=host,
                        backend_id="cpu",
                        transfer_backend="disk_pread",
                        sync_required=True,
                        attributes={
                            "kind": "activation_reload",
                            "spill_resource": host,
                            "consumer": inst.name,
                            "tensor_nbytes": {tensor: nbytes},
                        },
                    )
                )
                reload_deps.append(load_name)
                shared_reload[tensor] = load_name
                spilled.discard(tensor)
                spill_op_for.pop(tensor, None)
                live[tensor] = (host, nbytes)
                reload_count += 1
                if host != str(inst.resource):
                    tname = f"transfer::spill::{tensor}->{inst.resource}::{inst.name}"
                    out.append(
                        PlanInstruction(
                            opcode=OpCode.TRANSFER,
                            name=tname,
                            resource=_transfer_resource(host, str(inst.resource)),
                            depends_on=(load_name,),
                            inputs=(tensor,),
                            outputs=(tensor,),
                            nbytes=nbytes,
                            memory_tier=_tier_for_device(str(inst.resource)),
                            source=host,
                            destination=str(inst.resource),
                            backend_id="transfer",
                            transfer_backend=_transfer_backend(host, str(inst.resource)),
                            sync_required=False,
                            attributes={
                                "kind": "activation_reload_transfer",
                                "before_region": str(inst.executable_ref or ""),
                                "simulated_until_validated": "mock" in str(inst.resource),
                                "tensor_nbytes": {tensor: nbytes},
                            },
                        )
                    )
                    reload_deps.append(tname)
                    live[tensor] = (str(inst.resource), nbytes)
                    shared_reload[tensor] = tname

            new_deps = tuple(dict.fromkeys(list(inst.depends_on) + reload_deps))
            compute_inst = replace(inst, depends_on=new_deps) if reload_deps else inst
            out.append(compute_inst)

            for tensor in compute_inst.outputs:
                nbytes = _tensor_nbytes(compute_inst, tensor)
                live[tensor] = (str(compute_inst.resource), nbytes)
                spilled.discard(tensor)
                spill_op_for.pop(tensor, None)

            while _live_bytes() > budget_bytes:
                candidates = [
                    (t, res, n) for t, (res, n) in live.items() if t not in protected_tensors and t not in spilled
                ]
                if not candidates:
                    break

                current_idx = idx

                def _score(item: tuple[str, str, int], *, _at: int = current_idx) -> tuple[int, int]:
                    t, _res, n = item
                    sites = use_sites.get(t, [])
                    next_use = next((s for s in sites if s > _at), 10**9)
                    return (next_use, -n)

                tensor, resource, nbytes = sorted(candidates, key=_score)[-1]
                spill_name = f"evict::spill::{tensor}::{compute_inst.name}"
                # Wait for every already-emitted consumer of this tensor so a
                # later spill cannot yank RAM from a concurrent sibling.
                prior_consumers = [prev.name for prev in out if prev.opcode == OpCode.COMPUTE and tensor in prev.inputs]
                spill_deps = tuple(dict.fromkeys([compute_inst.name, *prior_consumers]))
                out.append(
                    PlanInstruction(
                        opcode=OpCode.EVICT,
                        name=spill_name,
                        resource=resource,
                        depends_on=spill_deps,
                        inputs=(tensor,),
                        outputs=(),
                        nbytes=nbytes,
                        memory_tier=MemoryTier.DISK,
                        source=resource,
                        destination="disk",
                        transfer_backend="disk_pread",
                        sync_required=True,
                        attributes={
                            "kind": "activation_spill",
                            "spill_resource": resource,
                            "producer_compute": compute_inst.name,
                            "tensor_nbytes": {tensor: nbytes},
                        },
                    )
                )
                del live[tensor]
                spilled.add(tensor)
                spill_op_for[tensor] = spill_name
                spill_nbytes[tensor] = nbytes
                shared_reload.pop(tensor, None)
                spill_count += 1
            # Fail closed when a spillable tensor still exceeds the budget.
            leftover = [t for t in live if t not in protected_tensors]
            if leftover and _live_bytes() > budget_bytes:
                raise ScheduleValidationError(
                    f"activation budget {budget_bytes} bytes cannot be met; "
                    f"spillable live tensors={leftover} bytes={_live_bytes()}"
                )
            continue

        if inst.opcode == OpCode.RELEASE:
            # Release may target RAM or disk copy after spill.
            resource = str(inst.attributes.get("release_resource") or inst.resource)
            for tensor in inst.inputs:
                live.pop(tensor, None)
                spilled.discard(tensor)
            out.append(inst)
            continue

        if inst.opcode == OpCode.EVICT and inst.attributes.get("kind") != "activation_spill":
            for tensor in inst.inputs:
                live.pop(tensor, None)
            out.append(inst)
            continue

        out.append(inst)

    if spill_count:
        note = f"schedule activation spill: spills={spill_count} reloads={reload_count} budget_bytes={budget_bytes}"
        if note not in notes:
            notes.append(note)
    return replace(schedule, instructions=tuple(out), notes=tuple(notes))


class ScheduleValidationError(ValueError):
    """Raised when an :class:`ExecutableSchedule` violates a structural invariant."""


def validate_schedule(schedule: ExecutableSchedule) -> list[str]:
    """Check structural invariants a runtime/simulator must be able to rely on.

    Returns a list of human-readable violations; empty means the schedule is
    safe to simulate or execute. Never silently drops or reorders instructions
    to "fix" a bad schedule -- planner bugs must surface, not be papered over.
    """
    schedule = ensure_explicit_streams(schedule)
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
        if inst.opcode in (OpCode.COMPUTE, OpCode.TRANSFER, OpCode.LOAD, OpCode.PREFETCH) and not inst.stream_id:
            errors.append(f"{inst.opcode.value} {inst.name!r} missing stream_id")
        if inst.opcode in (OpCode.TRANSFER, OpCode.LOAD, OpCode.PREFETCH) and not inst.copy_engine_id:
            errors.append(f"{inst.opcode.value} {inst.name!r} missing copy_engine_id")
        if inst.opcode == OpCode.TRANSFER and not inst.link_id:
            errors.append(f"transfer {inst.name!r} missing link_id")
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
            waits_for = str(inst.attributes.get("waits_for") or "")
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


def validate_schedule_tensor_sizes(schedule: ExecutableSchedule) -> list[str]:
    """Require exact per-tensor sizes on a specialized executable schedule.

    Zero-byte scalar/control values are allowed only when the instruction itself
    reports zero bytes. Tensor-bearing movement and compute operations must not
    rely on aggregate equal-split guesses.
    """
    errors: list[str] = []
    for inst in schedule.instructions:
        tensors = tuple(dict.fromkeys((*inst.inputs, *inst.outputs)))
        if not tensors:
            continue
        if inst.opcode not in {
            OpCode.PREFETCH,
            OpCode.LOAD,
            OpCode.TRANSFER,
            OpCode.COMPUTE,
            OpCode.EVICT,
            OpCode.RELEASE,
        }:
            continue
        raw = inst.attributes.get("tensor_nbytes")
        sizes = dict(raw) if isinstance(raw, Mapping) else {}
        for tensor in tensors:
            if tensor in sizes:
                try:
                    size = int(sizes[tensor])
                except (TypeError, ValueError):
                    errors.append(f"instruction {inst.name!r} has invalid byte size for {tensor!r}")
                    continue
                if size < 0:
                    errors.append(f"instruction {inst.name!r} has negative byte size for {tensor!r}")
                continue
            if len(tensors) == 1 and int(inst.nbytes or 0) >= 0:
                continue
            errors.append(f"instruction {inst.name!r} lacks exact tensor_nbytes for {tensor!r}")
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
