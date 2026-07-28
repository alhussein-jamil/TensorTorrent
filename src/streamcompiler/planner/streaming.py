"""Weight streaming schedule synthesis over memory tiers."""

from __future__ import annotations

from dataclasses import dataclass

from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode
from streamcompiler.ir.resource_graph import MemoryClass, ResourceGraph
from streamcompiler.planner.buffering import choose_buffering


@dataclass
class StreamStage:
    block_id: str
    action: str  # compute | transfer | prepare | load
    resource: str
    depends_on: tuple[str, ...] = ()


def synthesize_weight_stream(
    graph: HeterogeneousGraph,
    machine: ResourceGraph,
    *,
    compute_device: str,
) -> list[StreamStage]:
    """Build a double/triple-buffered weight streaming schedule for repeated blocks."""
    blocks = [list(b) for b in graph.repeated_blocks] or [[i.name for i in graph.compute_regions()]]
    has_nvme = any(m.memory_class == MemoryClass.NVME for m in machine.memory.values())
    has_cpu = any(n.startswith("cpu_numa_") for n in machine.compute)
    buffering = choose_buffering(
        has_copy_engine=True,
        has_cpu_prepare=has_cpu,
        has_nvme=has_nvme,
    )
    nvme = next((m.id.name for m in machine.memory.values() if m.memory_class == MemoryClass.NVME), None)
    cpu = next((n for n in machine.compute if n.startswith("cpu_numa_")), None)
    pinned = next(
        (m.id.name for m in machine.memory.values() if m.memory_class == MemoryClass.PINNED_HOST),
        None,
    )

    stages: list[StreamStage] = []
    for idx, _block in enumerate(blocks):
        block_id = f"block_{idx}"
        if buffering.storage_slot is not None and nvme is not None and idx + 3 < len(blocks):
            stages.append(
                StreamStage(
                    block_id=f"block_{idx+3}",
                    action="load",
                    resource=nvme,
                )
            )
        if buffering.prepare_slot is not None and cpu is not None and idx + 2 < len(blocks):
            stages.append(
                StreamStage(
                    block_id=f"block_{idx+2}",
                    action="prepare",
                    resource=cpu,
                    depends_on=((f"load:block_{idx+2}",) if has_nvme else ()),
                )
            )
        if buffering.transfer_slot is not None and pinned is not None and idx + 1 < len(blocks):
            stages.append(
                StreamStage(
                    block_id=f"block_{idx+1}",
                    action="transfer",
                    resource=pinned,
                    depends_on=((f"prepare:block_{idx+1}",) if has_cpu else ()),
                )
            )
        stages.append(
            StreamStage(
                block_id=block_id,
                action="compute",
                resource=compute_device,
                depends_on=(f"transfer:{block_id}",) if idx > 0 else (),
            )
        )
        # Also emit explicit IR prefetch/evict hints for the planner/runtime.
        if idx + 1 < len(blocks):
            graph.add_instruction(
                Instruction(
                    opcode=OpCode.PREFETCH,
                    name=f"prefetch_{idx+1}",
                    destination=compute_device,
                    attributes={"block": f"block_{idx+1}", "buffering": buffering.depth},
                )
            )
        if idx > 0:
            graph.add_instruction(
                Instruction(
                    opcode=OpCode.EVICT,
                    name=f"evict_{idx-1}",
                    source=compute_device,
                    attributes={"block": f"block_{idx-1}"},
                )
            )
    return stages
