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


def test_build_direct_plan_ignores_hoisted_parameter_evict() -> None:
    """Canonical GPU schedules keep parameter_evict; DirectPlan must still form."""
    import pytest
    import torch.nn as nn

    import tensortorrent as tt
    from tensortorrent.runtime.direct_path import build_direct_plan

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = Tiny().eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(use_torch_compile=False, measure_regions=False, allow_cpu=False),
    )
    try:
        schedule = compiled.executor.schedule
        assert any(
            inst.opcode == OpCode.EVICT and str(inst.attributes.get("kind") or "") == "parameter_evict"
            for inst in schedule.instructions
        )
        # Canonical schedule still has EVICT → raw _single_compute fails.
        assert _single_compute(schedule) is None
        plan = compiled.executor.direct_plan
        assert isinstance(plan, DirectPlan)
        assert build_direct_plan(compiled.executor) is not None
    finally:
        compiled.close()

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
