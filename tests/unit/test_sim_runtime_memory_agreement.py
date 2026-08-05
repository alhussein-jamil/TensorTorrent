"""Simulator and runtime memory peaks agree on deterministic CPU schedules."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from tests.support.helpers import cpu_host_graph

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.hardware.discovery import discover_resource_graph
from tensortorrent.ir.graph import OpCode
from tensortorrent.runtime.simulator.discrete_event import simulate_schedule


@pytest.fixture(autouse=True)
def _force_schedule_path_for_module(monkeypatch):
    """Pin every ``tt.compile`` in this module to the schedule path.

    These tests assert schedule-executor internals (native artifact counters,
    ``_last_schedule_report``, native residency handles, etc.) which are only
    populated when the schedule path drives execution. Direct-path selection
    is automatic elsewhere and correctness-gated at compile time; this fixture
    disables the direct plan builder for the duration of the module so the
    schedule executor stays authoritative.
    """
    from tensortorrent.runtime import direct_path as _direct_path

    def _no_direct_plan(_executor):
        return None

    monkeypatch.setattr(_direct_path, "build_direct_plan", _no_direct_plan)
    yield


def test_cpu_resident_sim_runtime_peak_agreement() -> None:
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(4, 16)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False),
    )
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
        sim_peak = int(sim.activation_peak_bytes)
        # Activation peaks must agree exactly (same tensor_nbytes metadata).
        assert runtime_peak == sim_peak, f"activation peak runtime={runtime_peak} sim={sim_peak}"
        # Stronger: instruction-level transferred bytes match for Transfer ops.
        sim_xfer = int(sim.bytes_transferred)
        rt_xfer = sum(e.nbytes for e in report.events if e.opcode == "Transfer")
        assert sim_xfer == rt_xfer
        assert sim.instruction_count == len(schedule.instructions) == len(report.events)
        # Critical path non-empty when schedule has compute.
        assert sim.critical_path
        assert set(sim.critical_path).issubset({i.name for i in schedule.instructions})
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
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            activation_budget_bytes=64,
            max_concurrent_regions=2,
            allow_gpu=False,
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
        sched_spill_bytes = sum(i.nbytes for i in spill_ops)
        assert report.activation_bytes_written == sched_spill_bytes
        sim_written = sum(int(e.get("activation_bytes_written") or 0) for e in sim.timeline)
        assert sim_written == sched_spill_bytes
        assert report.activation_bytes_written == sim_written
    finally:
        compiled.close()


def test_mock_accel_sim_runtime_activation_peak() -> None:
    from tensortorrent.backends.mock_accel import make_mock_accel_graph
    from tensortorrent.compile.measure import MeasurementSet, RegionMeasurement
    from tensortorrent.ir.resource_graph import merge_graphs

    class Branch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Linear(8, 8)
            self.right = nn.Linear(8, 8)
            self.head = nn.Linear(16, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(torch.cat([torch.relu(self.left(x)), torch.relu(self.right(x))], dim=-1))

    model = Branch().eval()
    x = torch.randn(2, 8)
    config = CompileConfig(
        use_torch_compile=False,
        measure_regions=False,
        allow_concurrent_regions=True,
        max_concurrent_regions=2,
        max_region_nodes=8,
        allow_gpu=False,
    )
    machine = merge_graphs(cpu_host_graph(), make_mock_accel_graph())
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    accel = "mock_accel_0"
    probe = tt.compile(model, (x,), config=config)
    try:
        region_ids = [r.region_id for r in probe._program.regions]
        ms = MeasurementSet()
        for i, rid in enumerate(region_ids):
            if i % 2 == 0:
                ms.add(RegionMeasurement(rid, cpu, "cpu", 0.001, True))
                ms.add(RegionMeasurement(rid, accel, "mock_accel", 1.0, False, simulated=True))
            else:
                ms.add(RegionMeasurement(rid, cpu, "cpu", 1.0, True))
                ms.add(RegionMeasurement(rid, accel, "mock_accel", 0.001, False, simulated=True))
    finally:
        probe.close()
    compiled = tt.compile(model, (x,), config=config, machine=machine, measurements=ms)
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        sim = simulate_schedule(schedule, machine)
        with torch.no_grad():
            out = compiled(x)
            torch.testing.assert_close(out, model(x), atol=1e-4, rtol=1e-4)
        report = compiled.executor._last_schedule_report
        assert report is not None
        assert int(report.peak_activation_bytes) == int(sim.activation_peak_bytes)
    finally:
        compiled.close()
