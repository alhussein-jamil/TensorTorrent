"""Activation spill planning for executable schedules."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tensortorrent.closed import InstructionKind
from tensortorrent.ir.graph import OpCode
from tensortorrent.runtime.schedule.build import (
    _load_host_for_destination,
    _tier_for_device,
    _transfer_backend,
)
from tensortorrent.runtime.schedule.types import (
    ExecutableSchedule,
    MemoryTier,
    PlanInstruction,
    ScheduleValidationError,
)
from tensortorrent.runtime.schedule.validate import _transfer_resource


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
                            "kind": InstructionKind.ACTIVATION_RELOAD.value,
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
                                "kind": InstructionKind.ACTIVATION_RELOAD_TRANSFER.value,
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
                            "kind": InstructionKind.ACTIVATION_SPILL.value,
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

        if inst.opcode == OpCode.EVICT and inst.attributes.get("kind") != InstructionKind.ACTIVATION_SPILL:
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
