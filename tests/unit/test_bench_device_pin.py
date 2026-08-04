"""Regression: CPU-labelled benchmarks must not silently place on CUDA."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig


def test_allow_gpu_false_keeps_output_on_cpu() -> None:
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(4, 32)
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(allow_gpu=False, allow_cpu=True, use_torch_compile=False),
    )
    try:
        out = compiled(x)
        assert out.device.type == "cpu"
        backends = compiled.specialized.validation.get("backends_used", [])
        assert "cuda" not in backends
    finally:
        compiled.close()


def test_compare_baselines_helper_pins_cpu_and_cuda() -> None:
    from bench.compare_baselines import compile_config_for_device

    cpu = compile_config_for_device("cpu")
    assert cpu.allow_gpu is False
    assert cpu.allow_cpu is True
    cuda = compile_config_for_device("cuda")
    assert cuda.allow_gpu is True
    assert cuda.allow_cpu is True
