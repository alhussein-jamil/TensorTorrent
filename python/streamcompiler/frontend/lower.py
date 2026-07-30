"""Lower torch.export graphs into StreamCompiler heterogeneous IR.

The IR mirrors the executable region program one-to-one: every ``Compute``
instruction corresponds to a real fx subgraph, every tensor entry carries the
shape and dtype ``torch.export`` recorded, and instruction inputs/outputs encode
the true dataflow. Nothing here is synthesized or guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streamcompiler.codegen.regions import RegionProgram, build_region_program
from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta


@dataclass
class LoweredModel:
    """A model lowered to both planner IR and executable regions."""

    ir: HeterogeneousGraph
    program: RegionProgram


def lower_exported_program(
    exported: Any,
    *,
    name: str = "model",
    max_region_nodes: int = 16,
    max_region_state_bytes: int | None = None,
    force_single_region: bool = False,
) -> LoweredModel:
    """Convert an ``ExportedProgram`` into hardware-independent heterogeneous IR."""
    program = build_region_program(
        exported,
        name=name,
        max_region_nodes=max_region_nodes,
        max_region_state_bytes=max_region_state_bytes,
        force_single_region=force_single_region,
    )
    return LoweredModel(ir=ir_from_region_program(program), program=program)


def ir_from_region_program(program: RegionProgram) -> HeterogeneousGraph:
    """Build planner IR that exactly describes ``program``."""
    graph = HeterogeneousGraph(
        name=program.graph_name,
        metadata={
            "frontend": "torch.export",
            "user_inputs": list(program.user_inputs),
            "state_bindings": dict(program.state_bindings),
            "region_count": len(program.regions),
            "max_region_nodes": program.metadata.get("max_region_nodes"),
        },
    )

    for spec in program.values.values():
        # Portable logical homes — specialization maps these to concrete resources.
        if spec.kind in ("parameter", "buffer", "constant"):
            home = "parameter_home"
        elif spec.kind == "input":
            home = "host_memory"
        else:
            home = "unassigned"
        storage = program.state_bindings.get(spec.name)
        graph.add_tensor(
            TensorMeta(
                tensor_id=spec.name,
                shape=tuple(spec.shape),
                dtype=spec.dtype,
                size_bytes=spec.nbytes,
                kind=spec.kind,
                home_tier=home,
                storage_id=storage,
                alias_group=f"storage::{storage}" if storage else None,
                mutable=False,
            )
        )

    for name in program.state_bindings:
        graph.add_instruction(
            Instruction(
                opcode=OpCode.LOAD,
                name=f"load::{name}",
                outputs=(name,),
                source=program.state_bindings[name],
                dtype=program.values[name].dtype,
                attributes={"kind": program.values[name].kind},
            )
        )
        tensor = graph.tensors.get(name)
        if tensor is not None:
            tensor.produced_at = 0

    for index, region in enumerate(program.regions):
        graph.add_instruction(
            Instruction(
                opcode=OpCode.COMPUTE,
                name=region.region_id,
                inputs=region.inputs,
                outputs=region.outputs,
                dtype=_region_dtype(program, region.outputs),
                attributes={
                    "submodule": region.submodule,
                    "aten_ops": list(region.aten_ops),
                    "node_count": region.node_count,
                    "depends_on": list(region.depends_on),
                    "state_inputs": list(region.state_inputs),
                    "output_bytes": region.output_bytes,
                    "order": index,
                },
            )
        )
        for out in region.outputs:
            tensor = graph.tensors.get(out)
            if tensor is not None:
                tensor.produced_at = index
        for inp in region.inputs:
            tensor = graph.tensors.get(inp)
            if tensor is not None:
                tensor.last_use_at = index

    graph.parameters = tuple(program.state_bindings)
    graph.outputs = tuple(str(ref) for kind, ref in program.output_refs if kind == "value")
    for out in graph.outputs:
        tensor = graph.tensors.get(out)
        if tensor is not None:
            tensor.last_use_at = len(program.regions)
    return graph


def _region_dtype(program: RegionProgram, outputs: tuple[str, ...]) -> str:
    for name in outputs:
        spec = program.values.get(name)
        if spec is not None and spec.dtype != "unknown":
            return spec.dtype
    return "float32"
