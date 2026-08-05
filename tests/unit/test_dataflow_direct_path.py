"""Direct-path parameter hoist + dataflow eligibility."""

from __future__ import annotations

import torch

from tensortorrent.ir.graph import OpCode
from tensortorrent.runtime.direct_path import (
    DirectParameter,
    _compute_region_predecessors,
    _transfer_kinds_ok_for_dataflow,
)
from tensortorrent.runtime.schedule import ExecutableSchedule, MemoryTier, PlanInstruction


def test_direct_parameter_refreshes_after_source_mutation() -> None:
    source = torch.tensor([1.0, 2.0])
    parameter = DirectParameter.place(source, torch.device("cpu"))
    cached = parameter.resolve()
    with torch.no_grad():
        source.add_(3.0)
    refreshed = parameter.resolve()
    assert refreshed is not cached
    torch.testing.assert_close(refreshed, source)


def test_direct_parameter_reuses_unchanged_copy() -> None:
    source = torch.tensor([1.0])
    parameter = DirectParameter.place(source, torch.device("cpu"))
    placed = parameter.resolve()
    assert parameter.resolve() is placed


def test_transfer_kinds_reject_activation_transfers() -> None:
    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=(
            PlanInstruction(
                opcode=OpCode.TRANSFER,
                name="xfer::act",
                resource="cpu",
                destination="cuda:0",
                inputs=("a",),
                outputs=("a",),
                attributes={"kind": "activation"},
                memory_tier=MemoryTier.SYSTEM_RAM,
            ),
        ),
    )
    assert _transfer_kinds_ok_for_dataflow(schedule) is False


def test_transfer_kinds_accept_parameter_hoists() -> None:
    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=(
            PlanInstruction(
                opcode=OpCode.TRANSFER,
                name="xfer::w",
                resource="cpu",
                destination="cuda:0",
                inputs=("w",),
                outputs=("w",),
                attributes={"kind": "parameter_host_to_device"},
                memory_tier=MemoryTier.SYSTEM_RAM,
            ),
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="compute::r0",
                resource="cuda:0",
                executable_ref="r0",
                depends_on=("xfer::w",),
                memory_tier=MemoryTier.DEVICE,
            ),
        ),
    )
    assert _transfer_kinds_ok_for_dataflow(schedule) is True


def test_compute_predecessors_walk_through_events() -> None:
    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="compute::a",
                resource="cpu",
                executable_ref="a",
                memory_tier=MemoryTier.SYSTEM_RAM,
            ),
            PlanInstruction(
                opcode=OpCode.RECORD_EVENT,
                name="rec::a",
                resource="cpu",
                depends_on=("compute::a",),
                memory_tier=MemoryTier.SYSTEM_RAM,
            ),
            PlanInstruction(
                opcode=OpCode.WAIT_EVENT,
                name="wait::a",
                resource="cuda:0",
                depends_on=("rec::a",),
                memory_tier=MemoryTier.DEVICE,
            ),
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="compute::b",
                resource="cuda:0",
                executable_ref="b",
                depends_on=("wait::a",),
                memory_tier=MemoryTier.DEVICE,
            ),
        ),
    )
    preds = _compute_region_predecessors(schedule)
    assert preds["a"] == set()
    assert preds["b"] == {"a"}


def test_direct_path_skips_native_artifact_until_schedule_run() -> None:
    import tensortorrent as tt

    model = torch.nn.Linear(4, 2).eval()
    x = torch.randn(2, 4)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False),
    )
    try:
        assert compiled.executor.direct_plan is not None
        se = compiled.executor._schedule_executor
        assert se is not None
        assert se._native_artifact is None
        torch.testing.assert_close(compiled(x), model(x), atol=1e-4, rtol=1e-4)
        assert se._native_artifact is None
    finally:
        compiled.close()


def test_time_executor_defaults_disable_dataflow() -> None:
    """Fusion timer must not auto-enable dataflow (regression for self-confirm)."""
    import inspect

    from tensortorrent.compile.pipeline import _time_executor

    sig = inspect.signature(_time_executor)
    assert sig.parameters["enable_dataflow_direct_path"].default is False
