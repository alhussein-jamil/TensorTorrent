"""Collective planning over backend-neutral communication contracts."""

from __future__ import annotations

from dataclasses import dataclass

from tensortorrent.backends.communication import select_communication_backend
from tensortorrent.ir.graph import HeterogeneousGraph, OpCode
from tensortorrent.ir.resource_graph import ResourceGraph


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
    # Do not invent collectives the IR never asked for. Multi-device sync cost is
    # modeled by cross-device transfer edges in the simulator instead.
    _ = machine
    return plans
