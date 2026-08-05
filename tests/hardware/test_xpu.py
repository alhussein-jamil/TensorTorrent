"""Real Intel XPU coverage — skipped when torch.xpu is unavailable."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.hardware.discovery import discover_resource_graph
from tensortorrent.validation.hardware import CheckStatus, validate_hardware


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    is_available = getattr(xpu, "is_available", None)
    return bool(callable(is_available) and is_available())


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.hardware,
    pytest.mark.skipif(not _xpu_available(), reason="Intel XPU (torch.xpu) required"),
]


def test_discovery_reports_xpu() -> None:
    graph = discover_resource_graph()
    assert "xpu" in graph.backends_present
    gpus = [g for g in graph.gpus() if g.backend_id == "xpu" or "xpu" in g.id.name]
    assert gpus, f"expected XPU devices, got {[g.id.name for g in graph.gpus()]}"


def test_compile_on_xpu_matches_eager() -> None:
    model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 8)).eval()
    x = torch.randn(4, 128)
    with torch.no_grad():
        expected = model(x)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(allow_cpu=True, allow_gpu=True, use_torch_compile=False),
    )
    try:
        torch.testing.assert_close(compiled(x).detach().cpu(), expected, atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()


def test_validate_hardware_executes_xpu_basic_path() -> None:
    report = validate_hardware(full=False, stress=False)
    assert any(c.name == "backend_available:xpu" and c.status is CheckStatus.BACKEND_AVAILABLE for c in report.checks)
    xpu_exec = [
        c
        for c in report.checks
        if c.name.startswith("basic_execution:")
        and "xpu" in c.name
        and c.status is CheckStatus.BASIC_EXECUTION_VALIDATED
    ]
    assert xpu_exec, "expected measured basic_execution for an XPU device"
