"""One executable plan format shared by planner, simulator, and runtime.

Instructions are explicit memory/compute ops. The simulator must not invent
schedules the runtime cannot perform: both consume :class:`ExecutableSchedule`.
"""

from __future__ import annotations

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
) -> ExecutableSchedule:
    """Lower placements + residency into an ordered executable instruction list.

    CPU-only single-device plans emit Compute + Release. Cross-device edges emit
    explicit Transfer ops before the consumer Compute. Streaming plans emit
    Prefetch/Load before Compute when ``streaming`` is True.
    """
    by_id = {p.region_id: p for p in plan.placements}
    transfers = list(residency.transfers) if residency is not None else []
    transfer_before: dict[str, list[ScheduledTransfer]] = {}
    for transfer in transfers:
        transfer_before.setdefault(transfer.before_region, []).append(transfer)

    instructions: list[PlanInstruction] = []
    last_compute: dict[str, str] = {}
    notes = list(plan.notes)

    for index, placement in enumerate(plan.placements):
        compute_name = f"compute::{placement.region_id}"
        deps: list[str] = []
        for dep in placement.depends_on:
            if dep in last_compute:
                deps.append(last_compute[dep])

        if streaming and placement.state_bytes > 0:
            prefetch_name = f"prefetch::{placement.region_id}"
            # Prefetch may start after the previous region's compute when distance allows.
            prefetch_deps: tuple[str, ...] = ()
            if index >= prefetch_distance and plan.placements[index - prefetch_distance].region_id in last_compute:
                prefetch_deps = (last_compute[plan.placements[index - prefetch_distance].region_id],)
            instructions.append(
                PlanInstruction(
                    opcode=OpCode.PREFETCH,
                    name=prefetch_name,
                    resource="nvme_or_pack",
                    depends_on=prefetch_deps,
                    inputs=(f"state::{placement.region_id}",),
                    outputs=(f"state::{placement.region_id}",),
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
            load_name = f"load::{placement.region_id}"
            instructions.append(
                PlanInstruction(
                    opcode=OpCode.LOAD,
                    name=load_name,
                    resource=placement.device,
                    depends_on=(prefetch_name,),
                    inputs=(f"state::{placement.region_id}",),
                    outputs=(f"state::{placement.region_id}",),
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
            deps.append(load_name)

        for transfer in transfer_before.get(placement.region_id, ()):
            tname = f"transfer::{transfer.after_region}->{transfer.before_region}:{transfer.value_name}"
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
                    sync_required=True,
                    attributes={
                        "after_region": transfer.after_region,
                        "before_region": transfer.before_region,
                        "simulated_until_validated": True,
                    },
                )
            )
            deps.append(tname)

        instructions.append(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name=compute_name,
                resource=placement.device,
                depends_on=tuple(deps),
                inputs=tuple(f"activation::{d}" for d in placement.depends_on)
                + ((f"state::{placement.region_id}",) if placement.state_bytes else ()),
                outputs=(f"activation::{placement.region_id}",),
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
                },
            )
        )
        last_compute[placement.region_id] = compute_name

        # Release producer activations after the last scheduled consumer begins
        # (recorded here; runtime frees after consumer completion).
        for dep in placement.depends_on:
            producer = by_id.get(dep)
            if producer is None or producer.output_bytes <= 0:
                continue
            remaining = sum(1 for p in plan.placements if dep in p.depends_on)
            # Emit release when this is the last consumer in plan order.
            later = [
                p
                for p in plan.placements[index + 1 :]
                if dep in p.depends_on
            ]
            if later:
                continue
            instructions.append(
                PlanInstruction(
                    opcode=OpCode.RELEASE,
                    name=f"release::activation::{dep}",
                    resource=producer.device,
                    depends_on=(compute_name,),
                    inputs=(f"activation::{dep}",),
                    outputs=(),
                    nbytes=producer.output_bytes,
                    memory_tier=_tier_for_device(producer.device),
                    predicted_duration_s=0.0,
                    attributes={"kind": "activation", "producer_region": dep, "consumer_count": remaining},
                )
            )

    if residency is not None:
        notes.extend(n for n in residency.notes if n not in notes)

    return ExecutableSchedule(
        graph_name=plan.graph_name,
        fingerprint=plan.fingerprint,
        instructions=instructions,
        notes=notes,
    )


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
            elif dep.opcode == OpCode.TRANSFER:
                after = dep.attributes.get("after_region")
                if after:
                    depends.append(str(after))
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
