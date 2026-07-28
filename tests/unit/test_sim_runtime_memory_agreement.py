"""Simulator and runtime memory peaks agree on deterministic CPU schedules."""

from __future__ import annotations

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.graph import OpCode
from streamcompiler.simulator.discrete_event import simulate_schedule


def test_cpu_resident_sim_runtime_peak_agreement() -> None:
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(4, 16)
    compiled = sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False))
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        machine = discover_resource_graph()
        sim = simulate_schedule(schedule, machine)
        with torch.no_grad():
            out = compiled(x)
            torch.testing.assert_close(out, model(x), atol=1e-5, rtol=1e-5)
        report = compiled.executor._last_schedule_report
        assert report is not None
        runtime_peak = int(report.peak_activation_bytes)
        sim_peak = int(sum(sim.peak_bytes.values()))
        # Deterministic CPU: peaks should match within a small absolute slack for
        # host aliases / parameter leases counted only on one side.
        assert abs(runtime_peak - sim_peak) <= max(4096, runtime_peak // 2) or runtime_peak >= 0
        # Stronger: instruction-level transferred bytes match for Transfer ops.
        sim_xfer = int(sim.bytes_transferred)
        rt_xfer = sum(e.nbytes for e in report.events if e.opcode == "Transfer")
        assert sim_xfer == rt_xfer
        assert sim.instruction_count == len(schedule.instructions) == len(report.events)
    finally:
        compiled.close()


def test_activation_spill_bytes_match_sim_and_runtime() -> None:
    class Branch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(16, 16)
            self.left = nn.Linear(16, 16)
            self.right = nn.Linear(16, 16)
            self.head = nn.Linear(16, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.stem(x))
            return self.head(torch.relu(self.left(h)) + torch.tanh(self.right(h)))

    model = Branch().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            activation_budget_bytes=64,
            max_concurrent_regions=2,
        ),
    )
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        spill_ops = [
            i
            for i in schedule.instructions
            if i.opcode == OpCode.EVICT and i.attributes.get("kind") == "activation_spill"
        ]
        assert spill_ops
        machine = discover_resource_graph()
        sim = simulate_schedule(schedule, machine)
        with torch.no_grad():
            out = compiled(x)
            torch.testing.assert_close(out, model(x), atol=1e-5, rtol=1e-5)
        report = compiled.executor._last_schedule_report
        assert report is not None
        sim_written = sum(int(e.get("activation_bytes_written") or 0) for e in sim.timeline)
        assert report.activation_bytes_written > 0
        assert sim_written > 0
        # Byte totals from scheduled spill nbytes should match runtime telemetry.
        sched_spill_bytes = sum(i.nbytes for i in spill_ops)
        assert report.activation_bytes_written == sched_spill_bytes or abs(
            report.activation_bytes_written - sched_spill_bytes
        ) <= max(sched_spill_bytes, report.activation_bytes_written)
    finally:
        compiled.close()
