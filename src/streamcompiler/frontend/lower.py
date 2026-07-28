"""Lower torch.export graphs into StreamCompiler heterogeneous IR."""

from __future__ import annotations

from typing import Any

from streamcompiler.ir.graph import HeterogeneousGraph, Instruction, OpCode, TensorMeta


def lower_exported_program(exported: Any, *, name: str = "model") -> HeterogeneousGraph:
    """Convert an ExportedProgram into hardware-independent heterogeneous IR."""
    graph = HeterogeneousGraph(name=name, metadata={"frontend": "torch.export"})
    try:
        gm = exported.module()
    except Exception:
        gm = exported

    # Parameters
    params = []
    try:
        state = dict(exported.state_dict()) if hasattr(exported, "state_dict") else {}
    except Exception:  # noqa: BLE001
        state = {}
    for pname, tensor in state.items():
        tid = f"param::{pname}"
        params.append(tid)
        nbytes = int(tensor.numel() * tensor.element_size()) if hasattr(tensor, "numel") else 0
        shape = tuple(int(x) for x in getattr(tensor, "shape", ()))
        dtype = str(getattr(tensor, "dtype", "float32")).replace("torch.", "")
        graph.add_tensor(
            TensorMeta(
                tensor_id=tid,
                shape=shape,
                dtype=dtype,
                size_bytes=nbytes,
                kind="parameter",
                home_tier="nvme",
                mutable=False,
            )
        )
    graph.parameters = tuple(params)

    # Nodes from FX graph when available.
    fx = getattr(gm, "graph", None)
    compute_names: list[str] = []
    if fx is not None:
        for i, node in enumerate(fx.nodes):
            if node.op == "call_function":
                target = getattr(node.target, "__name__", str(node.target))
                iname = f"op_{i}_{target}"
                compute_names.append(iname)
                graph.add_instruction(
                    Instruction(
                        opcode=OpCode.COMPUTE,
                        name=iname,
                        inputs=tuple(str(a) for a in node.args if hasattr(a, "name")),
                        outputs=(str(node.name),),
                        attributes={"aten": target, "fx_op": node.op},
                    )
                )
                graph.add_tensor(
                    TensorMeta(
                        tensor_id=str(node.name),
                        shape=(),
                        dtype="float32",
                        kind="activation",
                        produced_at=i,
                    )
                )
            elif node.op == "output":
                graph.outputs = (str(node.name),)

    # Repeated-block heuristic: group consecutive similarly named ops.
    if compute_names:
        # Simple chunking into layers of ~equal size for planner templates.
        chunk = max(1, len(compute_names) // 4)
        blocks = []
        for start in range(0, len(compute_names), chunk):
            blocks.append(tuple(compute_names[start : start + chunk]))
        graph.repeated_blocks = tuple(blocks)
    else:
        graph.add_instruction(Instruction(opcode=OpCode.COMPUTE, name="main"))
        graph.repeated_blocks = (("main",),)

    return graph
