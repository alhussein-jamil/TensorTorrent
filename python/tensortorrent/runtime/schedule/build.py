"""Build executable schedules from execution plans."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from tensortorrent.ir.graph import OpCode
from tensortorrent.planner.maximal import ExecutionPlan, Placement
from tensortorrent.runtime.residency import ResidencySchedule, ScheduledTransfer
from tensortorrent.runtime.resource_names import is_device_resource, is_host_resource
from tensortorrent.runtime.schedule.streams import ensure_explicit_streams
from tensortorrent.runtime.schedule.types import ExecutableSchedule, MemoryTier, PlanInstruction
from tensortorrent.runtime.schedule.validate import _transfer_resource


def _mock_delay_attrs(device: str, *, transfer: bool = False, compute: bool = False) -> dict[str, float]:
    """Mock-resource sleep attrs only. Omit zeros — Rust treats ``Some(0.0)`` as mock transfer."""
    if "mock" not in device:
        return {}
    out: dict[str, float] = {}
    if transfer:
        out["mock_transfer_delay_s"] = 0.08
    if compute:
        out["mock_compute_delay_s"] = 0.05
    return out


@dataclass
class _ParameterEvictGate:
    """Tracks per-placement parameter Evicts so later Loads/Transfers wait on real frees.

    Stateless (activation-only) placements call :meth:`skip`; :meth:`prior_deps`
    walks backward past those so H2D does not all become ready at t=0.
    """

    records: list[tuple[str | None, str | None]] = field(default_factory=list)

    def _pad_to(self, index: int) -> None:
        while len(self.records) <= index:
            self.records.append((None, None))

    def record(self, index: int, *, device: str | None, staging: str | None) -> None:
        self._pad_to(index)
        self.records[index] = (device, staging)

    def skip(self, index: int) -> None:
        self._pad_to(index)

    def prior_deps(self, index: int) -> list[str]:
        for i in range(min(index, len(self.records)) - 1, -1, -1):
            device, staging = self.records[i]
            deps = [name for name in (staging, device) if name]
            if deps:
                return deps
        return []

    def lead_gate(self, lead: int) -> str | None:
        if lead < 0:
            return None
        for i in range(min(lead, len(self.records) - 1), -1, -1):
            device, staging = self.records[i]
            gate = staging or device
            if gate:
                return gate
        return None


def _tier_for_device(device: str) -> MemoryTier:
    name = device.lower()
    if "disk" in name or "nvme" in name or "pack" in name:
        return MemoryTier.DISK
    if "pinned" in name:
        return MemoryTier.PINNED_RAM
    if is_device_resource(name):
        return MemoryTier.DEVICE
    if is_host_resource(name):
        return MemoryTier.SYSTEM_RAM
    from tensortorrent.backends import backend_id_for_resource

    if backend_id_for_resource(name) != "cpu":
        return MemoryTier.DEVICE
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


def _memory_allocatable_bytes(machine: Any | None, name: str) -> int | None:
    if machine is None:
        return None
    memories = getattr(machine, "memory", None) or {}
    mem = memories.get(name)
    if mem is None:
        return None
    try:
        return max(0, int(mem.allocatable_bytes))
    except (TypeError, ValueError, AttributeError):
        return None


def _first_numa_host(machine: Any | None) -> str | None:
    if machine is None:
        return None
    memories = getattr(machine, "memory", None) or {}
    for name, mem in memories.items():
        mclass = getattr(mem, "memory_class", None)
        mname = getattr(mclass, "value", mclass)
        if str(mname) in {"numa_ram", "system_ram"} or str(name).startswith("numa_ram"):
            return str(name)
    return None


def _load_host_for_destination(
    dest: str,
    *,
    machine: Any | None = None,
    nbytes: int = 0,
    force_pageable: bool = False,
) -> tuple[str, MemoryTier]:
    """Host staging for a Load feeding ``dest`` (pinned if it fits; else NUMA/pageable)."""
    if _tier_for_device(dest) != MemoryTier.DEVICE:
        return dest, _tier_for_device(dest)

    def _pageable() -> tuple[str, MemoryTier]:
        numa = _first_numa_host(machine)
        if numa is not None:
            return numa, MemoryTier.SYSTEM_RAM
        return "cpu", MemoryTier.SYSTEM_RAM

    if force_pageable:
        return _pageable()
    pinned = _first_pinned_host(machine)
    if pinned is not None:
        cap = _memory_allocatable_bytes(machine, pinned)
        need = max(0, int(nbytes))
        if cap is None or need <= 0 or need <= int(cap):
            return pinned, MemoryTier.PINNED_RAM
    return _pageable()


def _transfer_is_simulated(source: str, destination: str) -> bool:
    """Mock/virtual/unknown paths stay simulated; known CPU/CUDA DMA is executable."""
    blob = f"{source}|{destination}".lower()
    if "mock" in blob or "virtual" in blob or "simulated" in blob:
        return True

    def _known_real(name: str) -> bool:
        lower = name.lower()
        return is_host_resource(lower) or lower.startswith(("cuda_", "rocm_", "xpu_")) or "vram" in lower

    return not (_known_real(source) and _known_real(destination))


def _append_region_parameter_h2d(
    instructions: list[PlanInstruction],
    *,
    region_id: str,
    state_inputs: tuple[str, ...],
    state_sizes: dict[str, int],
    source: str,
    destination: str,
    depends_on: tuple[str, ...],
    emitted_transfers: set[str],
) -> str | None:
    """Emit one coalesced host→device Transfer for a region's parameters.

    Returns the instruction name when a new Transfer was appended, else ``None``.
    """
    needed = tuple(s for s in state_inputs if f"transfer::state::{s}->{destination}" not in emitted_transfers)
    if not needed:
        return None
    tname = f"transfer::state::{region_id}->{destination}"
    for state_name in needed:
        emitted_transfers.add(f"transfer::state::{state_name}->{destination}")
    tensor_nbytes = {str(s): max(1, int(state_sizes.get(s, 0) or 0)) for s in needed}
    instructions.append(
        PlanInstruction(
            opcode=OpCode.TRANSFER,
            name=tname,
            resource=_transfer_resource(source, destination),
            depends_on=depends_on,
            inputs=needed,
            outputs=needed,
            nbytes=max(1, sum(tensor_nbytes.values())),
            memory_tier=_tier_for_device(destination),
            predicted_duration_s=0.0,
            source=source,
            destination=destination,
            backend_id="transfer",
            transfer_backend=_transfer_backend(source, destination),
            sync_required=False,
            attributes={
                "region_id": region_id,
                "kind": "parameter_host_to_device",
                "simulated_until_validated": _transfer_is_simulated(source, destination),
                **_mock_delay_attrs(destination, transfer=True),
                "tensor_nbytes": tensor_nbytes,
            },
        )
    )
    return tname


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
    force_pageable_host_staging: bool = False,
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
    last_load_host: dict[str, tuple[str, MemoryTier]] = {}
    evict_gate = _ParameterEvictGate()
    notes = list(plan.notes)
    if force_pageable_host_staging:
        notes.append("host_staging=pageable")
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
                    **_mock_delay_attrs(transfer.destination_device, transfer=True),
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
            # Choose staging before Prefetch so IO timing targets the host that
            # Load will materialize into (never the accelerator device).
            state_nbytes = int(placement.state_bytes or 0)
            load_host, load_tier = _load_host_for_destination(
                placement.device,
                machine=machine,
                nbytes=state_nbytes,
                force_pageable=force_pageable_host_staging,
            )
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
                lead_evict = evict_gate.lead_gate(evict_lead)
                if lead_evict:
                    prefetch_deps_list.append(lead_evict)
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
                        destination=load_host,
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
            # When streaming, Load waits for previous Evicts so live staging + VRAM stay bounded.
            if streaming and index >= 1:
                load_deps.extend(evict_gate.prior_deps(index))
            # Load is always disk → host RAM (prefer pinned when feeding a device and it fits).
            # Device residency needs an explicit Transfer after this Load.
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
            last_load_host[placement.region_id] = (load_host, load_tier)
            deps.append(load_name)
            if load_host != placement.device:
                tname = _append_region_parameter_h2d(
                    instructions,
                    region_id=placement.region_id,
                    state_inputs=state_inputs,
                    state_sizes=state_sizes,
                    source=load_host,
                    destination=placement.device,
                    depends_on=(load_name,),
                    emitted_transfers=emitted_transfers,
                )
                if tname is not None:
                    deps.append(tname)

        elif placement.state_bytes > 0 and _tier_for_device(placement.device) == MemoryTier.DEVICE:
            # Resident pack: host-seeded weights → device Transfer (+ Evict after Compute).
            state_inputs = state_t or (f"state::{placement.region_id}",)
            load_host = "cpu"
            for p in plan.placements:
                if is_host_resource(p.device):
                    load_host = p.device
                    break
            tname = _append_region_parameter_h2d(
                instructions,
                region_id=placement.region_id,
                state_inputs=state_inputs,
                state_sizes=state_sizes,
                source=load_host,
                destination=placement.device,
                depends_on=tuple(dict.fromkeys(evict_gate.prior_deps(index))),
                emitted_transfers=emitted_transfers,
            )
            if tname is not None:
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
                    **_mock_delay_attrs(placement.device, compute=True),
                    "tensor_nbytes": tensor_nbytes,
                },
            )
        )
        last_compute[placement.region_id] = compute_name

        # Device Evict after accelerator Compute; staging Evict only for streaming Loads.
        on_accelerator = _tier_for_device(placement.device) == MemoryTier.DEVICE
        if placement.state_bytes > 0 and state_t and (streaming or on_accelerator):
            evict_tensors = _state_tensors_without_later_use(
                state_t,
                placements=plan.placements,
                start_index=index,
                region_io=region_io,
            )
            if not evict_tensors:
                evict_gate.skip(index)
            else:
                evict_nbytes = {str(n): int(state_sizes.get(str(n), 0) or 0) for n in evict_tensors}
                evict_total = sum(evict_nbytes.values()) or placement.state_bytes
                evict_name = f"evict::state::{placement.region_id}"
                instructions.append(
                    PlanInstruction(
                        opcode=OpCode.EVICT,
                        name=evict_name,
                        resource=placement.device,
                        depends_on=(compute_name,),
                        inputs=evict_tensors,
                        outputs=(),
                        nbytes=evict_total,
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
                staging_evict: str | None = None
                if streaming:
                    staging = last_load_host.get(placement.region_id)
                    if staging is not None:
                        staging_host, staging_tier = staging
                        if staging_host and staging_host != placement.device:
                            staging_evict = f"evict::state::staging::{placement.region_id}"
                            instructions.append(
                                PlanInstruction(
                                    opcode=OpCode.EVICT,
                                    name=staging_evict,
                                    resource=staging_host,
                                    depends_on=(compute_name,),
                                    inputs=evict_tensors,
                                    outputs=(),
                                    nbytes=evict_total,
                                    memory_tier=staging_tier,
                                    predicted_duration_s=0.0,
                                    destination=staging_host,
                                    attributes={
                                        "kind": "parameter_evict",
                                        "region_id": placement.region_id,
                                        "tensor_nbytes": evict_nbytes,
                                        "staging": True,
                                    },
                                )
                            )
                evict_gate.record(index, device=evict_name, staging=staging_evict)
        else:
            evict_gate.skip(index)

    # Releases: after every consumer of a tensor has computed. Graph outputs must
    # survive until output collection, even when another region also consumes them.
    graph_outputs = (
        {str(ref) for kind, ref in getattr(program, "output_refs", ()) if kind == "value"}
        if program is not None
        else set()
    )
    for producer in plan.placements:
        _, outputs_t, _ = region_io.get(
            producer.region_id,
            ((), (f"activation::{producer.region_id}",), ()),
        )
        for out_name in outputs_t:
            if out_name in graph_outputs:
                continue
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
        from tensortorrent.runtime.schedule.spill_plan import plan_activation_spills

        schedule = plan_activation_spills(
            schedule,
            budget_bytes=int(activation_budget_bytes),
            protected_tensors=frozenset(protected),
            program=program,
            machine=machine,
        )
    from tensortorrent.ir.liveness import apply_schedule_liveness

    schedule = apply_schedule_liveness(schedule)
    schedule = ensure_explicit_streams(schedule)
    from tensortorrent.runtime.schedule.validate import assert_schedule_valid

    assert_schedule_valid(schedule)
    return schedule


def hoist_resident_parameter_transfers(
    schedule: ExecutableSchedule,
    *,
    drop_parameter_evicts: bool = False,
) -> ExecutableSchedule:
    """Drop one-time host→device parameter copies from a repeated-inference view.

    Canonical schedules keep initialization Transfers for explain/validation.
    Steady-state runtime and DES ranking for non-streaming plans should score
    the post-hoist DAG so cold-start H2D does not overturn a faster device.

    When ``drop_parameter_evicts`` is True (DES ranking), also drop matching
    ``parameter_evict`` ops so validation does not require a producer Transfer
    that was hoisted away. Runtime keeps those Evicts; the executor seeds
    resident weights separately.
    """
    hoisted = {
        inst.name
        for inst in schedule.instructions
        if inst.opcode == OpCode.TRANSFER
        and str(inst.attributes.get("kind") or "") == "parameter_host_to_device"
        and "mock" not in str(inst.destination or inst.resource).lower()
    }
    if drop_parameter_evicts:
        hoisted |= {
            inst.name
            for inst in schedule.instructions
            if inst.opcode == OpCode.EVICT and str(inst.attributes.get("kind") or "") == "parameter_evict"
        }
    if not hoisted:
        return schedule
    n_xfer = sum(1 for inst in schedule.instructions if inst.name in hoisted and inst.opcode == OpCode.TRANSFER)
    instructions = tuple(
        replace(
            inst,
            depends_on=tuple(dep for dep in inst.depends_on if dep not in hoisted),
        )
        for inst in schedule.instructions
        if inst.name not in hoisted
    )
    note = f"hoisted_resident_parameter_transfers={n_xfer}"
    if drop_parameter_evicts:
        note += f";hoisted_parameter_evicts={len(hoisted) - n_xfer}"
    return ExecutableSchedule(
        graph_name=schedule.graph_name,
        fingerprint=schedule.fingerprint,
        instructions=instructions,
        notes=(*schedule.notes, note),
    )
