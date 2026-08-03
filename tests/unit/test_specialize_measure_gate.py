"""Specialize skips expensive region-input capture unless measuring."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.compile import pipeline as pipe
from tensortorrent.compile.pipeline import specialize_for_machine
from tensortorrent.config import CompileConfig


def test_specialize_skips_capture_when_measure_regions_false() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False),
    )
    try:
        with patch.object(
            pipe,
            "capture_region_inputs",
            side_effect=AssertionError("capture_region_inputs should not run"),
        ) as mocked:
            specialized = specialize_for_machine(
                compiled.portable,
                config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False),
                example_inputs=compiled._example_flat,
            )
        mocked.assert_not_called()
        assert specialized.plan.placements
    finally:
        compiled.close()


def test_specialize_captures_when_measure_regions_true() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(2, 8)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False),
    )
    try:
        with patch.object(
            pipe,
            "capture_region_inputs",
            wraps=pipe.capture_region_inputs,
        ) as mocked:
            specialize_for_machine(
                compiled.portable,
                config=CompileConfig(
                    use_torch_compile=False,
                    measure_regions=True,
                    allow_gpu=False,
                    region_measure_iters=1,
                ),
                example_inputs=compiled._example_flat,
            )
        assert mocked.called
    finally:
        compiled.close()


def test_compile_artifact_dir_persists_exported() -> None:
    model = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "art"
        compiled = tt.compile(
            model,
            (x,),
            config=CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False),
            artifact_dir=out,
        )
        try:
            assert (out / "exported.pt2").is_file()
            assert (out / "compile_config.json").is_file()
            loaded = tt.load_compiled(out)
            try:
                torch.testing.assert_close(loaded(x), compiled(x))
            finally:
                loaded.close()
        finally:
            compiled.close()


def test_compile_exported_two_phase_matches_compile() -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
    x = torch.randn(2, 8)
    cfg = CompileConfig(use_torch_compile=False, measure_regions=False, allow_gpu=False)
    exported = tt.capture_module(model, (x,))
    compiled = tt.compile_exported(exported, config=cfg, name="TwoPhase")
    try:
        torch.testing.assert_close(compiled(x), model(x))
    finally:
        compiled.close()


def test_collect_outputs_prefers_device_over_host() -> None:
    """Device-resident copies win over host when both exist."""
    from types import SimpleNamespace

    from tensortorrent.runtime.schedule_executor import ScheduleExecutor

    class _Copies:
        def __init__(self) -> None:
            self._data = {
                "y": {
                    "host": SimpleNamespace(value=torch.tensor([1.0])),
                    "cuda_gpu_0": SimpleNamespace(value=torch.tensor([2.0])),
                }
            }

        def resources_for(self, name: str) -> list[str]:
            return list(self._data[name])

        def get(self, name: str, resource: str) -> Any:
            return self._data[name][resource]

        def try_get(self, name: str, resource: str) -> Any | None:
            return self._data.get(name, {}).get(resource)

    program = SimpleNamespace(output_refs=[("value", "y")])
    ctx = SimpleNamespace(host_resource="host", copies=_Copies())
    executor = ScheduleExecutor.__new__(ScheduleExecutor)
    executor.program = program
    out = ScheduleExecutor._collect_outputs(executor, ctx)
    assert len(out) == 1
    assert float(out[0].item()) == 2.0


def test_move_tensor_to_resource_host_noop_and_cpu_roundtrip() -> None:
    from tensortorrent.runtime.native_bridge import _move_tensor_to_resource

    host = torch.randn(3, 3)
    assert _move_tensor_to_resource(host, "host", enable_grad=False) is host
    assert _move_tensor_to_resource(host, "cpu", enable_grad=False) is host
