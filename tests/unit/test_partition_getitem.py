"""Partitioning must keep ``getitem`` with multi-value producers."""

from __future__ import annotations

import operator

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from streamcompiler.compile.regions import assign_partitions, build_region_program
from streamcompiler.planner.maximal import Placement
from streamcompiler.runtime.schedule import _state_tensors_without_later_use


class _ChunkModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=-1)
        return a + b


def test_assign_partitions_colocates_getitem_with_chunk() -> None:
    gm = symbolic_trace(_ChunkModel())
    partition = assign_partitions(gm.graph, max_region_nodes=1)
    chunk_name = None
    getitem_names: list[str] = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.chunk:
            chunk_name = node.name
        if node.op == "call_function" and node.target is operator.getitem:
            getitem_names.append(node.name)
    assert chunk_name is not None
    assert getitem_names
    chunk_pid = partition[chunk_name]
    for name in getitem_names:
        assert partition[name] == chunk_pid


def test_force_single_region_puts_all_compute_in_one_partition() -> None:
    """Internal fuse path (concurrency off) collapses every compute node."""
    gm = symbolic_trace(_ChunkModel())
    partition = assign_partitions(gm.graph, force_single_region=True)
    assert set(partition.values()) == {0}


def test_state_dict_for_pack_uses_module_targets() -> None:
    model = nn.Linear(4, 2).eval()
    x = torch.randn(1, 4)
    exported = torch.export.export(model, (x,))
    program = build_region_program(exported, name="linear")
    packed = program.state_dict_for_pack()
    assert packed
    for key in packed:
        assert key in program.state_bindings.values()


def test_shared_state_evict_skips_tensors_with_later_consumers() -> None:
    region_io = {
        "region_0": (("x",), ("y0",), ("shared", "w0")),
        "region_1": (("y0",), ("y1",), ("shared", "w1")),
        "region_2": (("y1",), ("y2",), ("w2",)),
    }
    placements = [
        Placement(
            region_id="region_0",
            device="cpu",
            backend_id="cpu",
            dtype="float32",
            kernel_id="k",
            estimated_latency_s=0.0,
            depends_on=(),
            measured=False,
            output_bytes=0,
            state_bytes=8,
        ),
        Placement(
            region_id="region_1",
            device="cpu",
            backend_id="cpu",
            dtype="float32",
            kernel_id="k",
            estimated_latency_s=0.0,
            depends_on=("region_0",),
            measured=False,
            output_bytes=0,
            state_bytes=8,
        ),
        Placement(
            region_id="region_2",
            device="cpu",
            backend_id="cpu",
            dtype="float32",
            kernel_id="k",
            estimated_latency_s=0.0,
            depends_on=("region_1",),
            measured=False,
            output_bytes=0,
            state_bytes=4,
        ),
    ]
    assert _state_tensors_without_later_use(
        ("shared", "w0"),
        placements=placements,
        start_index=0,
        region_io=region_io,
    ) == ("w0",)
    assert set(
        _state_tensors_without_later_use(
            ("shared", "w1"),
            placements=placements,
            start_index=1,
            region_io=region_io,
        )
    ) == {"shared", "w1"}
