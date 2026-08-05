"""Static direct-path parameter cache behavior."""

from __future__ import annotations

import torch

from tensortorrent.ir.graph import OpCode
from tensortorrent.runtime.direct_path import DirectParameter, DirectPlan, _single_compute
from tensortorrent.runtime.schedule import ExecutableSchedule, PlanInstruction


def test_direct_parameter_refreshes_after_source_mutation() -> None:
    source = torch.tensor([1.0, 2.0])
    parameter = DirectParameter.place(source, torch.device("cpu"))
    cached = parameter.resolve()
    with torch.no_grad():
        source.add_(3.0)
    refreshed = parameter.resolve()
    assert parameter.source_version == source._version
    assert cached is source
    torch.testing.assert_close(refreshed, source)


def test_direct_parameter_reuses_unchanged_copy() -> None:
    source = torch.tensor([1.0])
    parameter = DirectParameter.place(source, torch.device("cpu"))
    placed = parameter.resolve()
    assert parameter.resolve() is placed


def test_single_compute_direct_plan_reproduces_static_input_transfer() -> None:
    schedule = ExecutableSchedule(
        graph_name="single",
        fingerprint="test",
        instructions=(
            PlanInstruction(OpCode.TRANSFER, "transfer::x", "cuda:0", inputs=("x",), outputs=("x",)),
            PlanInstruction(
                OpCode.RECORD_EVENT,
                "record::x",
                "cuda:0",
                depends_on=("transfer::x",),
            ),
            PlanInstruction(
                OpCode.COMPUTE,
                "compute::region_0",
                "cuda:0",
                depends_on=("record::x",),
                executable_ref="region_0",
            ),
        ),
    )
    assert _single_compute(schedule) is schedule.instructions[-1]

    moved_to: list[str] = []

    class Movable:
        def to(self, device: str) -> Movable:
            moved_to.append(str(device))
            return self

    value = Movable()
    plan = DirectPlan(
        region_id="region_0",
        device="cuda_gpu_0",
        torch_device="cuda:0",
        call=lambda x: x,
        arg_plan=((True, 0),),
        output_names=("y",),
    )
    assert plan.build_args([value]) == [value]
    assert moved_to == ["cuda:0"]
