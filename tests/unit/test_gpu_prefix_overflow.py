"""Static GPU-prefix + CPU-overflow cut and plan rewrite."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

from tensortorrent.compile.overflow import (
    gpu_prefix_count,
    gpu_prefix_overflow_plan,
    host_cpu_placement_target,
)
from tensortorrent.compile.regions import Region, RegionProgram, ValueSpec
from tensortorrent.config import Objective
from tensortorrent.ir.resource_graph import ComputeClass, ComputeResource, ResourceGraph, ResourceId, ResourceKind
from tensortorrent.planner.maximal import ExecutionPlan, Placement


def _program_with_region_state(sizes: list[int]) -> RegionProgram:
    regions = []
    values: dict[str, ValueSpec] = {
        "x": ValueSpec(name="x", shape=(2, 4), dtype="float32", nbytes=32, kind="input"),
    }
    state_bindings: dict[str, str] = {}
    prev_out = "x"
    for i, nbytes in enumerate(sizes):
        weight = f"w{i}"
        out = f"y{i}"
        values[weight] = ValueSpec(
            name=weight, shape=(nbytes // 4, 1), dtype="float32", nbytes=nbytes, kind="parameter"
        )
        values[out] = ValueSpec(name=out, shape=(2, 4), dtype="float32", nbytes=32, kind="activation")
        state_bindings[weight] = weight
        regions.append(
            Region(
                region_id=f"r{i}",
                submodule="",
                inputs=(prev_out, weight),
                outputs=(out,),
                multi_output=False,
                aten_ops=("aten.linear",),
                node_count=1,
                depends_on=() if i == 0 else (f"r{i - 1}",),
                state_inputs=(weight,),
                output_bytes=32,
            )
        )
        prev_out = out
    x = torch.zeros(2, 4)
    return RegionProgram(
        graph_name="overflow",
        root=torch.nn.Module(),
        regions=tuple(regions),
        values=values,
        user_inputs=("x",),
        state_bindings=state_bindings,
        output_refs=(("value", prev_out),),
        in_spec=pytree.tree_structure(((x,), {})),
        out_spec=pytree.tree_structure(x),
        metadata={},
    )


def _gpu_plan(program: RegionProgram) -> ExecutionPlan:
    placements = [
        Placement(
            region_id=region.region_id,
            device="cuda_gpu_0",
            backend_id="cuda",
            dtype="float32",
            kernel_id="cuda_fx_float32",
            estimated_latency_s=0.01,
            depends_on=region.depends_on,
            state_bytes=100,
        )
        for region in program.regions
    ]
    return ExecutionPlan(
        graph_name="overflow",
        fingerprint="gpu",
        objective=Objective.LATENCY,
        placements=placements,
        decisions=[],
        devices_used=("cuda_gpu_0",),
        communication_backend="none",
        predicted_latency_s=0.1,
        strategy="single_gpu",
        notes=["baseline_compare: stale"],
    )


def test_gpu_prefix_count_interior_and_bounds() -> None:
    program = _program_with_region_state([100, 100, 100, 100])
    assert gpu_prefix_count(program, 0) == 0
    assert gpu_prefix_count(program, 99) == 0
    assert gpu_prefix_count(program, 100) == 1
    assert gpu_prefix_count(program, 250) == 2
    assert gpu_prefix_count(program, 400) == 4


def test_gpu_prefix_count_shared_state_billed_once() -> None:
    program = _program_with_region_state([100, 100])
    shared = program.regions[1]
    regions = (
        program.regions[0],
        Region(
            region_id=shared.region_id,
            submodule=shared.submodule,
            inputs=shared.inputs,
            outputs=shared.outputs,
            multi_output=shared.multi_output,
            aten_ops=shared.aten_ops,
            node_count=shared.node_count,
            depends_on=shared.depends_on,
            state_inputs=("w0",),
            output_bytes=shared.output_bytes,
        ),
    )
    program.regions = regions
    program.state_bindings["w0"] = "shared"
    assert gpu_prefix_count(program, 100) == 2


def test_gpu_prefix_overflow_plan_rewrites_suffix() -> None:
    program = _program_with_region_state([50, 50, 50, 50])
    plan = gpu_prefix_overflow_plan(
        _gpu_plan(program),
        program,
        n_gpu=2,
        cpu_device="cpu_numa_0",
        cpu_backend="cpu",
    )
    by_id = {p.region_id: p for p in plan.placements}
    assert by_id["r0"].device == "cuda_gpu_0"
    assert by_id["r1"].backend_id == "cuda"
    assert by_id["r2"].device == "cpu_numa_0"
    assert by_id["r2"].backend_id == "cpu"
    assert by_id["r2"].kernel_id == "cpu_fx_float32"
    assert by_id["r3"].device == "cpu_numa_0"
    assert "cuda_gpu_0" in plan.devices_used
    assert "cpu_numa_0" in plan.devices_used
    assert plan.strategy == "gpu_prefix_cpu_overflow"
    assert plan.prefetch_distance == 0
    assert not any(str(n).startswith("baseline_compare") for n in plan.notes)


def test_gpu_prefix_overflow_plan_rejects_non_interior() -> None:
    program = _program_with_region_state([10, 10])
    source = _gpu_plan(program)
    with pytest.raises(ValueError, match="interior"):
        gpu_prefix_overflow_plan(source, program, n_gpu=0, cpu_device="cpu_numa_0", cpu_backend="cpu")
    with pytest.raises(ValueError, match="interior"):
        gpu_prefix_overflow_plan(source, program, n_gpu=2, cpu_device="cpu_numa_0", cpu_backend="cpu")


def test_host_cpu_placement_target() -> None:
    empty = ResourceGraph(fingerprint="empty")
    assert host_cpu_placement_target(empty) is None
    graph = ResourceGraph(fingerprint="host")
    graph.add_compute(
        ComputeResource(
            id=ResourceId(ResourceKind.COMPUTE, "cpu_numa_0"),
            compute_class=ComputeClass.CPU_NUMA_POOL,
            backend_id="cpu",
            model="host",
            vendor="cpu",
        )
    )
    assert host_cpu_placement_target(graph) == ("cpu_numa_0", "cpu")
