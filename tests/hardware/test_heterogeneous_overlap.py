"""Opt-in proof that automatic CPU/CUDA overlap beats eager all-CUDA."""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.runtime.direct_path import DataflowDirectPlan

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.hardware,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA GPU required"),
    pytest.mark.skipif(
        os.getenv("TT_RUN_PERF_TESTS") != "1",
        reason="set TT_RUN_PERF_TESTS=1 to run hardware performance assertions",
    ),
]


class _AsymmetricTowers(nn.Module):
    """Large CUDA-friendly tower plus small CPU-friendly independent tower."""

    def __init__(self) -> None:
        super().__init__()
        self.large = nn.Sequential(*sum(([nn.Linear(2048, 2048), nn.ReLU()] for _ in range(16)), []))
        self.small = nn.Sequential(*sum(([nn.Linear(256, 256), nn.ReLU()] for _ in range(16)), []))
        self.large_head = nn.Linear(2048, 32)
        self.small_head = nn.Linear(256, 32)

    def forward(self, large: torch.Tensor, small: torch.Tensor) -> torch.Tensor:
        return self.large_head(self.large(large)) + self.small_head(self.small(small))


def _median_cuda_ms(call, *, warmup: int = 12, iterations: int = 35) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def test_automatic_cpu_cuda_overlap_beats_all_cuda() -> None:
    if torch.cuda.get_device_properties(0).total_memory < 2 * 1024**3:
        pytest.skip("benchmark needs at least 2 GiB CUDA memory")

    torch.manual_seed(7)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    large_cpu = torch.randn(64, 2048)
    small_cpu = torch.randn(64, 256)
    large_cuda = large_cpu.cuda()
    small_cuda = small_cpu.cuda()

    source = _AsymmetricTowers().eval()
    eager_cuda = _AsymmetricTowers().eval().cuda()
    eager_cuda.load_state_dict(source.state_dict())
    compiled = tt.compile(
        source,
        (large_cpu, small_cpu),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=True,
            region_measure_iters=2,
            validate_numerics=False,
        ),
    )
    try:
        devices = set(compiled.specialized.plan.devices_used)
        assert any(device == "cpu" or device.startswith("cpu_") for device in devices)
        assert any(device.startswith("cuda_gpu_") for device in devices)
        assert isinstance(compiled.executor.direct_plan, DataflowDirectPlan)
        schedule_executor = compiled.executor._schedule_executor
        assert schedule_executor is not None
        parameter_transfers = {
            instruction.name
            for instruction in schedule_executor.schedule.instructions
            if instruction.attributes.get("kind") == "parameter_host_to_device"
        }
        assert parameter_transfers
        assert parameter_transfers.isdisjoint(schedule_executor._native_instruction_names)

        with torch.inference_mode():
            expected = eager_cuda(large_cuda, small_cuda)
            actual = compiled(large_cuda, small_cpu)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4, check_device=False)

        direct_plan = compiled.executor._direct_plan
        compiled.executor._direct_plan = None
        try:
            scheduled = compiled(large_cuda, small_cpu)
            torch.testing.assert_close(scheduled, expected, atol=1e-4, rtol=1e-4, check_device=False)
            assert compiled.executor._last_schedule_report is not None
        finally:
            compiled.executor._direct_plan = direct_plan

        eager_ms = _median_cuda_ms(lambda: eager_cuda(large_cuda, small_cuda))
        tensortorrent_ms = _median_cuda_ms(lambda: compiled(large_cuda, small_cpu))
        assert tensortorrent_ms < eager_ms, (
            f"expected automatic CPU/CUDA overlap to beat all-CUDA: "
            f"TensorTorrent={tensortorrent_ms:.3f}ms eager={eager_ms:.3f}ms "
            f"ratio={tensortorrent_ms / eager_ms:.3f}x"
        )
        assert compiled.last_report.parameter_store["execution_path"] == "direct_dataflow"
    finally:
        compiled.close()
        torch.set_num_threads(previous_threads)
