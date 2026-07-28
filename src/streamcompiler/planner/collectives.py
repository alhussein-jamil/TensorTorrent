"""Collective planning over backend-neutral communication contracts."""

from __future__ import annotations

from dataclasses import dataclass

from streamcompiler.communication import select_communication_backend
from streamcompiler.ir.graph import HeterogeneousGraph, OpCode
from streamcompiler.ir.resource_graph import ResourceGraph


@dataclass
class CollectivePlan:
    op: str
    devices: tuple[str, ...]
    backend_id: str
    host_staged: bool


def plan_collectives(
    graph: HeterogeneousGraph,
    machine: ResourceGraph,
    devices: tuple[str, ...],
) -> list[CollectivePlan]:
    """Insert collective ops when the IR requests them; choose a feasible backend."""
    backend = select_communication_backend(devices)
    plans: list[CollectivePlan] = []
    for inst in graph.instructions:
        if inst.opcode not in (OpCode.COLLECTIVE, OpCode.BROADCAST, OpCode.REDUCE):
            continue
        op = inst.opcode.value.lower()
        plans.append(
            CollectivePlan(
                op=op,
                devices=devices,
                backend_id=backend.backend_id,
                host_staged=backend.backend_id == "host_staged",
            )
        )
        inst.backend_id = backend.backend_id
        inst.attributes["collective_backend"] = backend.backend_id
    if not plans and len(devices) > 1:
        # Default allreduce placeholder for multi-device plans that may need sync.
        plans.append(
            CollectivePlan(
                op="allreduce",
                devices=devices,
                backend_id=backend.backend_id,
                host_staged=backend.backend_id == "host_staged",
            )
        )
    _ = machine
    return plans
