"""ExecutableSchedule structural validation must reject unsafe plans."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.ir.graph import OpCode
from streamcompiler.runtime.schedule import (
    ExecutableSchedule,
    PlanInstruction,
    ScheduleValidationError,
    assert_schedule_valid,
    validate_schedule,
    validate_schedule_resources,
)


def _compute(
    name: str, *, depends_on: tuple[str, ...] = (), inputs: tuple[str, ...] = (), outputs: tuple[str, ...] = ()
) -> PlanInstruction:
    return PlanInstruction(
        opcode=OpCode.COMPUTE,
        name=name,
        resource="cpu_numa_0",
        depends_on=depends_on,
        inputs=inputs,
        outputs=outputs,
        executable_ref=name,
    )


def test_real_compiled_plan_is_valid() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    compiled = sc.compile(model, (torch.randn(2, 8),))
    schedule = compiled.specialized.schedule
    assert schedule is not None
    assert validate_schedule(schedule) == []
    compiled.close()


def test_duplicate_instruction_id_is_rejected() -> None:
    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=[_compute("a"), _compute("a")],
    )
    errors = validate_schedule(schedule)
    assert any("duplicate" in e for e in errors)


def test_dangling_dependency_is_rejected() -> None:
    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=[_compute("a", depends_on=("does_not_exist",))],
    )
    errors = validate_schedule(schedule)
    assert any("unknown instruction" in e for e in errors)


def test_dependency_cycle_is_rejected() -> None:
    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=[
            _compute("a", depends_on=("b",)),
            _compute("b", depends_on=("a",)),
        ],
    )
    errors = validate_schedule(schedule)
    assert any("cycle" in e for e in errors)
    with pytest.raises(ScheduleValidationError):
        assert_schedule_valid(schedule)


def test_release_before_use_is_rejected() -> None:
    consumer = _compute("consumer", depends_on=("producer",), inputs=("t0",))
    release = PlanInstruction(
        opcode=OpCode.RELEASE,
        name="release::t0",
        resource="cpu_numa_0",
        depends_on=("producer",),
        inputs=("t0",),
    )
    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=[_compute("producer", outputs=("t0",)), release, consumer],
    )
    errors = validate_schedule(schedule)
    assert any("happens before consumer" in e for e in errors)


def test_release_after_use_is_accepted() -> None:
    producer = _compute("producer", outputs=("t0",))
    consumer = _compute("consumer", depends_on=("producer",), inputs=("t0",))
    release = PlanInstruction(
        opcode=OpCode.RELEASE,
        name="release::t0",
        resource="cpu_numa_0",
        depends_on=("consumer",),
        inputs=("t0",),
    )
    schedule = ExecutableSchedule(graph_name="g", fingerprint="f", instructions=[producer, consumer, release])
    assert validate_schedule(schedule) == []


def test_compute_before_transfer_completion_is_rejected() -> None:
    transfer = PlanInstruction(
        opcode=OpCode.TRANSFER,
        name="transfer::t0",
        resource="copy_engine",
        inputs=("t0",),
        outputs=("t0",),
    )
    # Compute reads t0 but never depends on the transfer that materializes it.
    consumer = _compute("consumer", inputs=("t0",))
    schedule = ExecutableSchedule(graph_name="g", fingerprint="f", instructions=[transfer, consumer])
    errors = validate_schedule(schedule)
    assert any("without depending on transfer completion" in e for e in errors)


def test_compute_resource_must_be_a_real_discovered_device() -> None:
    from streamcompiler.hardware.discovery import discover_resource_graph

    machine = discover_resource_graph()
    real_device = next(iter(machine.compute))
    good = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=[
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="a",
                resource=real_device,
                executable_ref="a",
            )
        ],
    )
    assert validate_schedule_resources(good, machine) == []

    bad = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=[
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="a",
                resource="made_up_device_that_does_not_exist",
                executable_ref="a",
            )
        ],
    )
    errors = validate_schedule_resources(bad, machine)
    assert any("unknown compute resource" in e for e in errors)


def test_compute_after_transfer_completion_is_accepted() -> None:
    transfer = PlanInstruction(
        opcode=OpCode.TRANSFER,
        name="transfer::t0",
        resource="copy_engine",
        inputs=("t0",),
        outputs=("t0",),
    )
    consumer = _compute("consumer", depends_on=("transfer::t0",), inputs=("t0",))
    schedule = ExecutableSchedule(graph_name="g", fingerprint="f", instructions=[transfer, consumer])
    assert validate_schedule(schedule) == []


def test_real_schedule_release_ops_name_real_producer_regions() -> None:
    """Planner RELEASE ops must cite producer regions that exist in the program."""

    class Branch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(16, 16)
            self.b = nn.Linear(16, 16)
            self.c = nn.Linear(16, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.a(x))
            return self.c(torch.relu(self.b(h)) + h)

    compiled = sc.compile(
        Branch().eval(),
        (torch.randn(2, 16),),
        config=sc.CompileConfig(max_concurrent_regions=2),
    )
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        region_ids = set(compiled.regions)
        releases = [i for i in schedule.instructions if i.opcode == OpCode.RELEASE]
        assert releases, "multi-region plan should emit activation Release ops"
        for inst in releases:
            producer = inst.attributes.get("producer_region")
            assert producer in region_ids
            assert any(dep.startswith("compute::") for dep in inst.depends_on)
    finally:
        compiled.close()
