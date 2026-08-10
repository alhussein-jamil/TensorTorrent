"""ExecutableSchedule structural validation must reject unsafe plans."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.ir.graph import OpCode
from tensortorrent.runtime.schedule import (
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
        stream_id="cpu_numa_0::compute",
    )


def test_real_compiled_plan_is_valid() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    compiled = tt.compile(model, (torch.randn(2, 8),))
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
        source="cpu_a",
        destination="cpu_b",
    )
    # Compute on destination without depending on the transfer that materializes t0.
    consumer = PlanInstruction(
        opcode=OpCode.COMPUTE,
        name="consumer",
        resource="cpu_b",
        inputs=("t0",),
        executable_ref="consumer",
    )
    schedule = ExecutableSchedule(graph_name="g", fingerprint="f", instructions=[transfer, consumer])
    errors = validate_schedule(schedule)
    assert any("without depending on transfer completion" in e for e in errors)


def test_compute_resource_must_be_a_real_discovered_device() -> None:
    from tensortorrent.hardware.discovery import discover_resource_graph

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
        source="cpu_a",
        destination="cpu_b",
    )
    consumer = PlanInstruction(
        opcode=OpCode.COMPUTE,
        name="consumer",
        resource="cpu_b",
        depends_on=("transfer::t0",),
        inputs=("t0",),
        executable_ref="consumer",
    )
    schedule = ExecutableSchedule(graph_name="g", fingerprint="f", instructions=[transfer, consumer])
    assert validate_schedule(schedule) == []


def test_wait_for_unrecorded_event_is_rejected() -> None:
    wait = PlanInstruction(
        opcode=OpCode.WAIT_EVENT,
        name="wait::x",
        resource="cpu_numa_0",
        attributes={"waits_for": "record::missing"},
    )
    schedule = ExecutableSchedule(graph_name="g", fingerprint="f", instructions=[wait])
    errors = validate_schedule(schedule)
    assert any("never recorded" in e or "unknown event" in e for e in errors)


def test_transfer_missing_endpoints_is_rejected() -> None:
    transfer = PlanInstruction(
        opcode=OpCode.TRANSFER,
        name="transfer::bad",
        resource="copy_engine",
        inputs=("t0",),
        outputs=("t0",),
    )
    schedule = ExecutableSchedule(graph_name="g", fingerprint="f", instructions=[transfer])
    errors = validate_schedule(schedule)
    assert any("missing source or destination" in e for e in errors)


def test_compute_without_local_activation_copy_is_rejected() -> None:
    producer = PlanInstruction(
        opcode=OpCode.COMPUTE,
        name="compute::a",
        resource="cpu_a",
        outputs=("t0",),
        executable_ref="a",
    )
    consumer = PlanInstruction(
        opcode=OpCode.COMPUTE,
        name="compute::b",
        resource="cpu_b",
        inputs=("t0",),
        executable_ref="b",
    )
    schedule = ExecutableSchedule(graph_name="g", fingerprint="f", instructions=[producer, consumer])
    errors = validate_schedule(schedule)
    assert any("only produces it elsewhere" in e for e in errors)


def test_empty_tensor_id_on_transfer_is_rejected() -> None:
    transfer = PlanInstruction(
        opcode=OpCode.TRANSFER,
        name="transfer::empty",
        resource="copy_engine",
        inputs=("",),
        outputs=("",),
        source="cpu_a",
        destination="cpu_b",
    )
    schedule = ExecutableSchedule(graph_name="g", fingerprint="f", instructions=[transfer])
    errors = validate_schedule(schedule)
    assert any("empty tensor id" in e for e in errors)


def test_release_ops_cite_real_producer_regions() -> None:
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

    compiled = None
    releases = []
    region_ids: set[str] = set()
    # Weight init can collapse the graph to a single fused region; retry seeds
    # until the planner emits a multi-region schedule with activation Releases.
    for seed in range(32):
        if compiled is not None:
            compiled.close()
            compiled = None
        torch.manual_seed(seed)
        compiled = tt.compile(
            Branch().eval(),
            (torch.randn(2, 16),),
            config=tt.CompileConfig(max_concurrent_regions=2),
        )
        schedule = compiled.specialized.schedule
        assert schedule is not None
        region_ids = set(compiled.regions)
        releases = [i for i in schedule.instructions if i.opcode == OpCode.RELEASE]
        if len(region_ids) >= 2 and releases:
            break
    else:
        if compiled is not None:
            compiled.close()
        raise AssertionError("could not obtain multi-region plan with Release ops")

    try:
        for inst in releases:
            producer = inst.attributes.get("producer_region")
            assert producer in region_ids
            assert any(dep.startswith("compute::") for dep in inst.depends_on)
    finally:
        compiled.close()
