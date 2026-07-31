"""ExecutableSchedule stays immutable across execution; runtime state is contextual."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.runtime.simulator.discrete_event import simulate_schedule


def test_schedule_unchanged_before_after_execution() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(
        model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False)
    )
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        before = schedule.as_dict()
        before_json = json.dumps(before, sort_keys=True, default=str)
        with torch.no_grad():
            out = compiled(x)
            torch.testing.assert_close(out, model(x), atol=1e-5, rtol=1e-5)
        after = compiled.specialized.schedule.as_dict()
        assert json.dumps(after, sort_keys=True, default=str) == before_json
        # No futures / tensors in serialized metadata.
        blob = before_json.lower()
        assert "future" not in blob
        assert "tensor(" not in blob
        assert "_async_future" not in blob
    finally:
        compiled.close()


def test_same_schedule_repeated_and_simulated_identically() -> None:
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    x = torch.randn(2, 8)
    compiled = sc.compile(
        model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False)
    )
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        machine = discover_resource_graph()
        sim_before = simulate_schedule(schedule, machine)
        with torch.no_grad():
            for _ in range(3):
                out = compiled(x)
                torch.testing.assert_close(out, model(x), atol=1e-5, rtol=1e-5)
        sim_after = simulate_schedule(compiled.specialized.schedule, machine)
        assert sim_before.makespan_s == sim_after.makespan_s
        assert sim_before.peak_bytes == sim_after.peak_bytes
        assert sim_before.bytes_transferred == sim_after.bytes_transferred
        assert sim_before.critical_path == sim_after.critical_path
    finally:
        compiled.close()


def test_schedule_attributes_are_immutable() -> None:
    model = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    compiled = sc.compile(
        model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False)
    )
    try:
        inst = compiled.specialized.schedule.instructions[0]
        try:
            inst.attributes["mut"] = 1  # type: ignore[index]
            raised = False
        except TypeError:
            raised = True
        assert raised or not hasattr(inst.attributes, "__setitem__") or "mut" not in inst.attributes
        if "mut" in inst.attributes:
            raise AssertionError("instruction attributes must not accept mutation")
    finally:
        compiled.close()


def test_concurrent_executions_share_immutable_schedule() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
    x = torch.randn(2, 8)
    # Separate compiled modules — ScheduleExecutor is not reentrant.
    modules = [
        sc.compile(model, (x,), config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False))
        for _ in range(2)
    ]
    try:
        payloads = [m.specialized.schedule.as_dict() for m in modules]
        assert payloads[0] == payloads[1]

        def _run(m: sc.CompiledModule) -> torch.Tensor:
            with torch.no_grad():
                return m(x)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(_run, modules))
        for out in results:
            torch.testing.assert_close(out, model(x), atol=1e-5, rtol=1e-5)
        assert modules[0].specialized.schedule.as_dict() == payloads[0]
    finally:
        for m in modules:
            m.close()
