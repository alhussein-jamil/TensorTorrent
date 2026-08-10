"""Shared live-VRAM hoist clamp is the single budget authority."""

from __future__ import annotations

from tensortorrent.compile.fit import (
    ASSUMED_IDLE_VRAM_FREE_FRACTION,
    clamp_hoist_budget_to_live_vram,
    cuda_device_index_from_resource,
    live_hoist_budget_bytes,
    optimistic_hoist_budget_without_cuda,
)
from tensortorrent.config import CompileConfig


def test_cuda_device_index_from_resource() -> None:
    assert cuda_device_index_from_resource("cuda_0") == 0
    assert cuda_device_index_from_resource("cuda_gpu_1") == 1
    assert cuda_device_index_from_resource("cpu_numa_0") is None


def test_optimistic_hoist_budget_without_cuda_uses_named_fraction() -> None:
    cfg = CompileConfig(vram_budget_bytes=8 << 30)
    budget = optimistic_hoist_budget_without_cuda(cfg, 8 << 30)
    assumed = int((8 << 30) * ASSUMED_IDLE_VRAM_FREE_FRACTION)
    assert 1 <= budget <= assumed


def test_clamp_hoist_budget_without_cuda_is_identity() -> None:
    # CPU-only hosts: clamp is a no-op.
    assert clamp_hoist_budget_to_live_vram(12345, device_indices={0}, synchronize=False) == 12345


def test_live_hoist_budget_bytes_without_cuda_matches_configured_budget() -> None:
    from tensortorrent.compile.fit import accelerator_hoist_budget_bytes

    cfg = CompileConfig(vram_budget_bytes=4 << 30)
    expected = accelerator_hoist_budget_bytes(cfg, None)
    assert live_hoist_budget_bytes(cfg, None, device_indices={0}) == expected
