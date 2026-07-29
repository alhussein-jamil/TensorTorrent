"""Simulator and runtime share stream/engine/link/io contention keys."""

from __future__ import annotations

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.graph import OpCode
from streamcompiler.simulator.discrete_event import simulate_schedule


def test_schedule_contention_ids_filled_and_sim_runs() -> None:
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False))
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        for inst in schedule.instructions:
            if inst.opcode == OpCode.COMPUTE:
                assert inst.stream_id, inst
            if inst.opcode in (OpCode.TRANSFER, OpCode.PREFETCH, OpCode.LOAD):
                assert inst.stream_id or inst.copy_engine_id or inst.io_queue_id, inst
        sim = simulate_schedule(schedule, discover_resource_graph())
        assert sim.instruction_count == len(schedule.instructions)
        with torch.no_grad():
            torch.testing.assert_close(compiled(x), model(x), atol=1e-5, rtol=1e-5)
        report = compiled.executor._last_schedule_report
        assert report is not None
        assert int(report.peak_activation_bytes) == int(sim.activation_peak_bytes)
    finally:
        compiled.close()


def test_strict_missing_release_fails_unless_idempotent() -> None:
    from streamcompiler.native import require_native
    from streamcompiler.runtime.schedule import ExecutableSchedule, PlanInstruction

    native = require_native()
    bad = ExecutableSchedule(
        graph_name="missing_release",
        fingerprint="f",
        instructions=[
            PlanInstruction(
                opcode=OpCode.RELEASE,
                name="release::ghost",
                resource="cpu",
                depends_on=(),
                inputs=("ghost",),
                outputs=(),
                nbytes=1,
                attributes={"kind": "activation"},
            ),
        ],
    )
    try:
        native.execute_schedule(bad, dry_run=False)
        raised = False
    except Exception as exc:
        raised = True
        assert "missing" in str(exc).lower() or "not resident" in str(exc).lower(), exc
    assert raised

    ok = ExecutableSchedule(
        graph_name="idempotent_release",
        fingerprint="f",
        instructions=[
            PlanInstruction(
                opcode=OpCode.RELEASE,
                name="release::ghost",
                resource="cpu",
                depends_on=(),
                inputs=("ghost",),
                outputs=(),
                nbytes=1,
                attributes={"kind": "activation", "idempotent": True},
            ),
        ],
    )
    native.execute_schedule(ok, dry_run=False)
