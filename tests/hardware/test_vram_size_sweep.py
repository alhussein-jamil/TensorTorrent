"""Size sweep against real NVIDIA VRAM: fit, near-limit, and oversize models.

Heavy oversize cases run in isolated subprocesses so host RAM is reclaimed between
sizes (compile temporarily holds ~2× parameter bytes).
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.hardware.discovery import discover_resource_graph

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
    payload_out = json.loads(lines[-1])
    if payload_out.get("skip"):
        pytest.skip(str(payload_out.get("error") or "worker requested skip"))
    return payload_out


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
    # CPU and CUDA kernels need not be bit-identical; the worker already checks
    # the declared numerical tolerance with torch.testing.assert_close.
    assert result["max_abs_err"] <= 1e-3
    assert result["cuda_peak_bytes"] < vram_bytes


def _skip_if_insufficient_scratch(needed_bytes: int) -> None:
    """Skip when the cache filesystem cannot hold the pack this case will write.

    The streaming cases materialise a pack roughly the size of the model's
    parameters. Without this check a short-on-disk host spends a long time
    building each oversize case only to fail on ENOSPC or a quota error, and
    the whole sweep looks like a code failure rather than an environment one.
    """
    import shutil

    root = Path(os.environ.get("TT_CACHE_DIR") or Path.home() / ".cache" / "tensortorrent")
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    # Compile transiently holds about twice the parameter bytes on top of the pack.
    required = int(needed_bytes * 1.2)
    if free < required:
        pytest.skip(f"needs ~{required / 1e9:.1f} GB free on {probe} for the pack, only {free / 1e9:.1f} GB available")


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
def test_models_exceeding_vram_stream_on_device_and_match_eager(vram_bytes: int, fraction: float, layers: int) -> None:
    result = _run_worker(
        {
            "mode": "oversize_stream",
            "vram_bytes": vram_bytes,
            "fraction": fraction,
            "layers": layers,
        }
    )
    assert result["ok"] is True
    assert result["on_cuda"] is True
    assert result.get("store_kind") == "resident"
    assert result["streaming"] is False
    assert result["params_bytes"] > vram_bytes
    # Oversize fit uses lower-precision device kernels (bf16/fp16); allow float noise.
    assert result["max_abs_err"] < 1e-3
    assert result["cuda_peak_bytes"] < vram_bytes
    assert result["reads"] == 0


def test_force_gpu_when_model_exceeds_vram_stays_on_cuda_or_errors(vram_bytes: int) -> None:
    """allow_cpu=False + tiny budgets: PlanningError or CUDA streaming — never CPU."""
    result = _run_worker({"mode": "force_gpu_no_cpu", "vram_bytes": vram_bytes, "fraction": 1.25, "layers": 16})
    assert result["ok"] is True
    assert result.get("raised") == "PlanningError" or (
        result.get("raised") is None
        and result.get("store_kind") == "streaming"
        and any(str(d).startswith("cuda_gpu_") for d in (result.get("devices") or []))
    )


def test_activation_heavy_fit_model_survives_repeated_forwards(vram_bytes: int, tmp_path: Path) -> None:
    width = _dims_for_target_params(int(vram_bytes * 0.15), layers=10)
    model = DeepMLP(width, 10).eval()
    assert _param_bytes(model) < vram_bytes // 2
    x = torch.randn(16, width)
    with torch.no_grad():
        expected = model(x).clone()
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
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
            torch.testing.assert_close(compiled(x), expected, atol=1e-3, rtol=1e-3, check_device=False)
        assert torch.cuda.max_memory_allocated() < vram_bytes
    finally:
        compiled.close()
        del compiled, x, expected
        _cleanup()
