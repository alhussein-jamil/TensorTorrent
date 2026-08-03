"""BackendProfiler: CPU measured, virtual accel simulated."""

from __future__ import annotations

import torch

from tensortorrent.backends.profiler import (
    CpuBackendProfiler,
    VirtualAccelBackendProfiler,
    profiler_for_backend,
)


def test_cpu_profiler_region_and_transfer() -> None:
    prof = CpuBackendProfiler()
    x = torch.randn(32, 32)

    def _mm() -> torch.Tensor:
        return x @ x

    rec = prof.profile_region(
        lambda t: t @ t,
        (x,),
        device_fingerprint="cpu-test",
        region_graph_hash="mm",
        warm_up=1,
        samples=3,
    )
    assert rec.measured is True
    assert rec.simulated is False
    assert rec.sample_count == 3
    assert rec.median_s >= 0.0
    xfer = prof.profile_transfer(4096, source="cpu", destination="cpu", device_fingerprint="cpu-test")
    assert xfer.kind == "transfer"
    assert xfer.measured is True


def test_virtual_accel_profiler_labelled_simulated() -> None:
    prof = VirtualAccelBackendProfiler(compute_delay_s=0.05, transfer_delay_s=0.08)
    x = torch.randn(8, 8)
    rec = prof.profile_region(
        lambda t: t + 1,
        (x,),
        device_fingerprint="mock-test",
        region_graph_hash="add",
        warm_up=1,
        samples=4,
    )
    assert rec.simulated is True
    assert rec.measured is False
    assert abs(rec.median_s - 0.05) < 1e-9
    assert "simulated" in " ".join(rec.notes)
    overlap = prof.profile_overlap(lambda: None, lambda: None, device_fingerprint="mock-test")
    assert abs(overlap.median_s - 0.08) < 1e-9


def test_profiler_for_backend_dispatch() -> None:
    assert isinstance(profiler_for_backend("cpu"), CpuBackendProfiler)
    assert isinstance(profiler_for_backend("mock_accel"), VirtualAccelBackendProfiler)
