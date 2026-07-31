"""Size sweep against real NVIDIA VRAM: fit, near-limit, and oversize models.

Heavy oversize cases run in isolated subprocesses so host RAM is reclaimed between
sizes (compile temporarily holds ~2× parameter bytes).
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.hardware.discovery import discover_resource_graph

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.hardware,
    pytest.mark.slow,
    pytest.mark.timeout(600),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA GPU required"),
]

_WORKER = Path(__file__).with_name("_vram_size_worker.py")


class DeepMLP(nn.Module):
    def __init__(self, width: int, layers: int, out_features: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.head = nn.Linear(width, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)


def _param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _vram_bytes() -> int:
    return int(torch.cuda.get_device_properties(0).total_memory)


def _dims_for_target_params(target_bytes: int, *, layers: int) -> int:
    per_layer = max(target_bytes // max(layers, 1), 1)
    width = int((per_layer / 4) ** 0.5)
    return max(64, (width // 64) * 64)


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _run_worker(payload: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, str(_WORKER)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        timeout=540,
    )
    if proc.returncode != 0:
        raise AssertionError(f"worker failed rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"worker produced no JSON\nstderr:\n{proc.stderr}"
    return json.loads(lines[-1])


@pytest.fixture(scope="module")
def vram_bytes() -> int:
    return _vram_bytes()


def test_resource_graph_vram_matches_torch(vram_bytes: int) -> None:
    graph = discover_resource_graph()
    gpus = [g for g in graph.gpus() if g.backend_id == "cuda"]
    assert gpus
    mem_names = gpus[0].memory_affinity
    assert mem_names
    discovered = max(graph.memory[n].capacity_bytes for n in mem_names if n in graph.memory)
    assert abs(discovered - vram_bytes) / vram_bytes < 0.15, (discovered, vram_bytes)


@pytest.mark.parametrize(
    "fraction,layers",
    [
        (0.03, 6),
        (0.08, 8),
        (0.15, 10),
        (0.25, 12),
        (0.40, 14),
        (0.55, 16),
        (0.70, 18),
    ],
    ids=["3pct", "8pct", "15pct", "25pct", "40pct", "55pct", "70pct"],
)
def test_models_that_fit_vram_place_on_cuda(vram_bytes: int, fraction: float, layers: int) -> None:
    result = _run_worker(
        {
            "mode": "fit_cuda",
            "vram_bytes": vram_bytes,
            "fraction": fraction,
            "layers": layers,
        }
    )
    assert result["ok"] is True
    assert result["on_cuda"] is True
    assert result["params_bytes"] < int(vram_bytes * 0.80)
    assert result["max_abs_err"] == 0.0
    assert result["cuda_peak_bytes"] < vram_bytes


@pytest.mark.parametrize(
    "fraction,layers",
    [
        (1.05, 12),
        (1.10, 14),
        (1.15, 14),
        (1.20, 16),
        (1.25, 16),
        (1.35, 18),
        (1.50, 18),
    ],
    ids=["1.05x", "1.10x", "1.15x", "1.20x", "1.25x", "1.35x", "1.50x"],
)
def test_models_exceeding_vram_stream_on_cpu_and_match_eager(vram_bytes: int, fraction: float, layers: int) -> None:
    result = _run_worker(
        {
            "mode": "oversize_stream",
            "vram_bytes": vram_bytes,
            "fraction": fraction,
            "layers": layers,
        }
    )
    assert result["ok"] is True
    assert result["on_cuda"] is False
    assert result["on_cpu"] is True
    assert result["streaming"] is True
    assert result["params_bytes"] > vram_bytes
    assert result["max_abs_err"] == 0.0
    assert result["peak_resident_bytes"] <= result["budget_bytes"] + (1 << 20)
    assert result["reads"] > 0
    assert result["cuda_peak_bytes"] < vram_bytes // 4


def test_force_gpu_when_model_exceeds_vram_raises_planning_error(vram_bytes: int) -> None:
    result = _run_worker({"mode": "force_gpu_fail", "vram_bytes": vram_bytes, "fraction": 1.25, "layers": 16})
    assert result["ok"] is True
    assert result["raised"] == "PlanningError"


def test_activation_heavy_fit_model_survives_repeated_forwards(vram_bytes: int, tmp_path: Path) -> None:
    width = _dims_for_target_params(int(vram_bytes * 0.15), layers=10)
    model = DeepMLP(width, 10).eval()
    assert _param_bytes(model) < vram_bytes // 2
    x = torch.randn(16, width)
    with torch.no_grad():
        expected = model(x).clone()
    compiled = sc.compile(
        model,
        (x,),
        config=sc.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            cache_dir=tmp_path / "cache",
        ),
    )
    del model
    _cleanup()
    try:
        assert any(d.startswith("cuda_gpu_") for d in compiled.specialized.plan.devices_used)
        for _ in range(5):
            torch.testing.assert_close(compiled(x), expected, atol=1e-3, rtol=1e-3)
        assert torch.cuda.max_memory_allocated() < vram_bytes
    finally:
        compiled.close()
        del compiled, x, expected
        _cleanup()
