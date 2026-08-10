"""Concurrent forwards respect shared capacity accounting."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.errors import TensorTorrentError
from tensortorrent.runtime.capacity import CapacityBudgets, CapacityLease, CapacityLedger
from tensortorrent.serve.model_manager import ModelManager


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_concurrent_module_forwards_under_capacity() -> None:
    model = _MLP().eval()
    x = torch.randn(8, 64)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(allow_gpu=False, use_torch_compile=False),
    )
    try:
        ledger = compiled.capacity_ledger
        assert ledger.max_concurrent() >= 1

        def run() -> torch.Tensor:
            return compiled(x)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(run) for _ in range(16)]
            outs = [f.result() for f in as_completed(futures)]
        assert len(outs) == 16
        assert ledger.inflight == 0
    finally:
        compiled.close()


def test_serve_acquire_fail_closed_on_capacity() -> None:
    model = _MLP().eval()
    x = torch.randn(4, 64)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(allow_gpu=False, use_torch_compile=False),
    )
    try:
        need = CapacityLease(host_bytes=500)
        compiled._capacity_ledger = CapacityLedger(  # noqa: SLF001 - test inject
            CapacityBudgets(host_bytes=500, device_bytes=0, disk_bytes=0),
            per_request=need,
        )
        mgr = ModelManager()
        mgr.load("m", compiled, concurrency_limit=8)
        slot = mgr.acquire("m")
        assert slot.concurrency_limit == 1
        assert compiled.capacity_ledger.inflight == 0
        with pytest.raises(TensorTorrentError, match="concurrency limit"):
            mgr.acquire("m")
        compiled.capacity_ledger.acquire_or_raise()
        try:
            with pytest.raises(TensorTorrentError, match="capacity exhausted"):
                _ = compiled(x)
        finally:
            compiled.capacity_ledger.release()
        mgr.release_slot(slot)
    finally:
        compiled.close()


def test_host_ram_budget_streaming_compiles() -> None:
    """Artificially low ram_budget forces streaming — the >host-RAM capacity path."""
    model = _MLP().eval()
    x = torch.randn(4, 64)
    state = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    budget = max(state // 4, 8 * 1024)
    compiled = tt.compile(
        model,
        example_inputs=(x,),
        config=tt.CompileConfig(
            allow_gpu=False,
            use_torch_compile=False,
            ram_budget_bytes=budget,
            allow_nvme_streaming=True,
            max_region_nodes=2,
            prefer_direct_path=True,
        ),
    )
    try:
        store = compiled._executor.parameter_store  # noqa: SLF001
        assert store.needs_prefetch
        assert compiled._executor.direct_plan is None  # noqa: SLF001
        out = compiled(x)
        assert out.shape == (4, 64)
    finally:
        compiled.close()
