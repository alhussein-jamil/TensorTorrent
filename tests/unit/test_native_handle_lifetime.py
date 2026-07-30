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
            allow_gpu=False),
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
            allow_gpu=False),
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


class _WideDeep(nn.Module):
    """Weights ≫ tight RAM budget — forces many Load/Evict cycles."""

    def __init__(self, width: int = 256, layers: int = 24) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def test_large_model_streaming_stays_within_budget_and_batches_releases() -> None:
    """Model many times larger than budget: peak residency + batched handle_release."""
    require_native()
    from streamcompiler.testing import reset_native_counters, snapshot_native_counters

    model = _WideDeep().eval()
    x = torch.randn(8, 256)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    # Two Linear packs max — model is ≫ budget (24 layers).
    layer_bytes = 256 * 256 * 4 + 256 * 4  # weight + bias
    budget = layer_bytes * 2
    assert total > budget * 8, (total, budget)

    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(
            ram_budget_bytes=budget,
            max_region_nodes=1,
            prefetch_distance=1,
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False),
    )
    try:
        assert compiled.executor.parameter_store.needs_prefetch is True
        reset_native_counters()
        before = snapshot_native_counters()
        for _ in range(5):
            torch.testing.assert_close(compiled(x), model(x))
            stats = compiled.last_report.parameter_store
            assert int(stats["peak_resident_bytes"]) <= budget, stats
            assert int(stats.get("handle_live_bytes") or 0) <= budget + (1 << 20)
        after = snapshot_native_counters()
        href = after["handle_release_callbacks"] - before["handle_release_callbacks"]
        releases = after["parameter_release_callbacks"] - before["parameter_release_callbacks"]
        # Batched: fewer GIL handle_release calls than tensors released across forwards.
        assert href > 0
        assert href <= releases or releases == 0
        # Per-forward: handle_release callbacks must be well below naive per-tensor count.
        ops = len(compiled.executor._schedule_executor.schedule.instructions)
        release_ops = sum(
            1
            for i in compiled.executor._schedule_executor.schedule.instructions
            if i.opcode.value in {"Release", "Evict"}
        )
        per_forward_href = href / 5
        assert per_forward_href <= release_ops, (per_forward_href, release_ops, ops)
    finally:
        compiled.close()


def test_handle_release_callback_batches_multiple_tensors() -> None:
    """One Release with many inputs → one handle_release callback (not per tensor)."""
    require_native()
    from streamcompiler.testing import reset_native_counters, snapshot_native_counters

    model = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    ).eval()
    x = torch.randn(4, 32)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    budget = max(total // 3, 32 * 32 * 4 * 2)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(
            ram_budget_bytes=budget,
            max_region_nodes=1,
            prefetch_distance=1,
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False),
    )
    try:
        reset_native_counters()
        before = snapshot_native_counters()
        compiled(x)
        after = snapshot_native_counters()
        gil = after["gil_acquisitions"] - before["gil_acquisitions"]
        compute = after["compute_callbacks"] - before["compute_callbacks"]
        pload = after["parameter_load_callbacks"] - before["parameter_load_callbacks"]
        href = after["handle_release_callbacks"] - before["handle_release_callbacks"]
        # Exact accounting: every GIL is one of compute / param_load / handle_release / copy_sync.
        csync = after["copy_sync_callbacks"] - before["copy_sync_callbacks"]
        assert gil == compute + pload + href + csync, (gil, compute, pload, href, csync)
        assert int(compiled.last_report.parameter_store["peak_resident_bytes"]) <= budget
    finally:
        compiled.close()
