"""Streaming handle lifetime: drop opaque values when Rust final-releases."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.native import native_available, require_native

pytestmark = pytest.mark.skipif(not native_available(), reason="native required")


class _Chunky(nn.Module):
    """Enough distinct weights that a tiny budget forces eviction."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(6)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def _rss_bytes() -> int:
    # Linux: VmRSS from /proc/self/status (kB).
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def test_streaming_handle_bytes_stay_within_budget() -> None:
    require_native()
    model = _Chunky().eval()
    x = torch.randn(4, 64)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    budget = max(total // 4, 64 * 64 * 4 * 2)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(
            ram_budget_bytes=budget,
            max_region_nodes=1,
            prefetch_distance=1,
            use_torch_compile=False,
            measure_regions=False,
        ),
    )
    try:
        store = compiled.executor.parameter_store
        assert store.needs_prefetch is True
        assert store.stats()["budget_bytes"] == budget
        peaks: list[int] = []
        for _ in range(3):
            torch.testing.assert_close(compiled(x), model(x))
            stats = compiled.last_report.parameter_store
            peaks.append(int(stats.get("peak_resident_bytes") or 0))
            assert int(stats.get("peak_resident_bytes") or 0) <= budget
            assert int(stats.get("handle_live_bytes") or 0) <= budget + (1 << 20)
        assert max(peaks) <= budget
    finally:
        compiled.close()


def test_handle_release_drops_opaque_value_immediately() -> None:
    require_native()
    from streamcompiler.runtime.handles import NativeResidencyBridge

    bridge = NativeResidencyBridge.create()
    t = torch.randn(8)
    bridge.mirror_put("w", "cpu", t, nbytes=int(t.nbytes))
    assert len(bridge.handles) == 1
    bridge.drop_python_only("w", "cpu")
    assert len(bridge.handles) == 0


def test_streaming_forward_rss_does_not_accumulate_unbounded() -> None:
    """Process RSS must not grow unboundedly across repeated streaming forwards."""
    require_native()
    model = _Chunky().eval()
    x = torch.randn(4, 64)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    budget = max(total // 3, 64 * 64 * 4 * 2)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(
            ram_budget_bytes=budget,
            max_region_nodes=1,
            prefetch_distance=1,
            use_torch_compile=False,
            measure_regions=False,
        ),
    )
    try:
        # Warm allocator / caches.
        for _ in range(4):
            compiled(x)
        baseline = _rss_bytes()
        handle_peaks: list[int] = []
        for _ in range(12):
            compiled(x)
            stats = compiled.last_report.parameter_store
            assert int(stats["peak_resident_bytes"]) <= budget
            live = int(stats.get("handle_live_bytes") or 0)
            handle_peaks.append(live)
            assert live <= budget + (1 << 20)
        after = _rss_bytes()
        assert max(handle_peaks) <= budget + (1 << 20)
        if baseline > 0 and after > 0:
            # Refuse unbounded growth (multi-model leak). Allocator noise OK.
            assert after - baseline < max(budget * 32, 8 << 20), (baseline, after, budget)
    finally:
        compiled.close()
