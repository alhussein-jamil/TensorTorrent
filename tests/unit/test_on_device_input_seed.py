"""On-device inputs skip schedule H2D Transfer work."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.runtime.native_bridge.residency import _tensor_already_on_resource


def test_tensor_already_on_resource_cpu_and_cuda() -> None:
    cpu = torch.zeros(2)
    assert _tensor_already_on_resource(cpu, "cpu")
    assert not _tensor_already_on_resource(cpu, "cuda_gpu_0")
    if not torch.cuda.is_available():
        return
    gpu = torch.zeros(2, device="cuda")
    assert _tensor_already_on_resource(gpu, "cuda_gpu_0")
    assert not _tensor_already_on_resource(gpu, "cpu")


def test_schedule_on_device_input_matches_host_input() -> None:
    if not torch.cuda.is_available():
        return
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16)).eval().cuda()
    x = torch.randn(4, 32, device="cuda")
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=True,
            allow_cpu=False,
            prefer_direct_path=False,
        ),
    )
    try:
        se = compiled.executor._schedule_executor
        assert "input_1" in se._input_transfer_destinations
        assert se._input_transfer_destinations["input_1"].startswith("cuda_gpu_")
        y_gpu = compiled(x)
        y_cpu = compiled(x.cpu())
        assert torch.allclose(y_gpu, y_cpu, rtol=1e-4, atol=1e-4)
    finally:
        compiled.close()
