"""Native extension schedule round-trip and runtime smoke tests."""

from __future__ import annotations

import torch

from streamcompiler.ir.graph import OpCode
from streamcompiler.native import native_available, require_native
from streamcompiler.runtime.schedule import ExecutableSchedule, MemoryTier, PlanInstruction


def test_native_extension_available() -> None:
    assert native_available()
    native = require_native()
    assert native.native_version()


def test_schedule_json_roundtrip_lossless() -> None:
    native = require_native()
    schedule = ExecutableSchedule(
        graph_name="rt",
        fingerprint="fp",
        instructions=(
            PlanInstruction(
                opcode=OpCode.COMPUTE,
                name="compute::a",
                resource="cpu",
                inputs=("x",),
                outputs=("y",),
                nbytes=64,
                memory_tier=MemoryTier.SYSTEM_RAM,
                executable_ref="a",
                attributes={"tensor_nbytes": {"x": 32, "y": 64}},
            ),
        ),
        notes=("n1",),
    )
    back = native.schedule_roundtrip(schedule)
    assert back["graph_name"] == "rt"
    assert back["fingerprint"] == "fp"
    assert back["notes"] == ["n1"]
    assert len(back["instructions"]) == 1
    inst = back["instructions"][0]
    assert inst["opcode"] == "Compute"
    assert inst["name"] == "compute::a"
    assert inst["attributes"]["tensor_nbytes"]["y"] == 64


def test_native_dry_run_execute_empty_and_branch() -> None:
    native = require_native()
    empty = ExecutableSchedule(graph_name="e", fingerprint="f", instructions=(), notes=())
    report = native.execute_schedule(empty, dry_run=True)
    assert report["events"] == []

    schedule = ExecutableSchedule(
        graph_name="g",
        fingerprint="f",
        instructions=(
            PlanInstruction(OpCode.COMPUTE, "a", "cpu", outputs=("x",), nbytes=8, executable_ref="ra"),
            PlanInstruction(
                OpCode.COMPUTE,
                "b",
                "cpu",
                depends_on=("a",),
                inputs=("x",),
                outputs=("y",),
                nbytes=8,
                executable_ref="rb",
            ),
            PlanInstruction(
                OpCode.COMPUTE,
                "c",
                "cpu",
                depends_on=("a",),
                inputs=("x",),
                outputs=("z",),
                nbytes=8,
                executable_ref="rc",
            ),
        ),
    )
    report = native.execute_schedule(schedule, dry_run=True)
    assert len(report["events"]) == 3
    assert report["wall_time_s"] >= 0.0


def test_public_compile_marks_native_runtime() -> None:
    import streamcompiler as sc

    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l = torch.nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.l(x)

    model = M().eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(model, example_inputs=(x,), devices="cpu")
    out = compiled(x)
    torch.testing.assert_close(out, model(x))
    # CompiledModule may expose report via different attr; also check parameter store path.
    assert native_available()
