"""Real AMD ROCm coverage — skipped when HIP/torch ROCm is unavailable."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.hardware.discovery import discover_resource_graph
from tensortorrent.validation.hardware import CheckStatus, validate_hardware


def _rocm_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return bool(getattr(torch.version, "hip", None))


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.hardware,
    pytest.mark.skipif(not _rocm_available(), reason="AMD ROCm / HIP torch build required"),
]


def test_discovery_reports_rocm_gpu() -> None:
    graph = discover_resource_graph()
    assert "rocm" in graph.backends_present
    gpus = [g for g in graph.gpus() if g.backend_id == "rocm" or g.vendor == "amd"]
    assert gpus, f"expected ROCm GPUs, got {[g.id.name for g in graph.gpus()]}"


def test_compile_on_rocm_matches_eager() -> None:
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 8)).eval()
    x = torch.randn(8, 256)
    with torch.no_grad():
        expected = model(x)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(allow_cpu=True, allow_gpu=True, use_torch_compile=False),
    )
    try:
        devices = set(compiled.specialized.plan.devices_used)
        assert any("rocm" in d or "cuda_gpu_" in d for d in devices), f"expected GPU placement, got {devices}"
        torch.testing.assert_close(compiled(x).detach().cpu(), expected, atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()


def test_validate_hardware_executes_rocm_basic_path() -> None:
    report = validate_hardware(full=False, stress=False)
    assert any(c.name == "backend_available:rocm" and c.status is CheckStatus.BACKEND_AVAILABLE for c in report.checks)
    rocm_exec = [c for c in report.checks if c.name.startswith("basic_execution:rocm_gpu_")]
    assert rocm_exec, "expected basic_execution check for rocm_gpu_*"
    assert all(c.status is CheckStatus.BASIC_EXECUTION_VALIDATED for c in rocm_exec)
