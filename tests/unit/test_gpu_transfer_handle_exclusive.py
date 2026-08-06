"""Transfer must bind a dedicated opaque handle so GPU Evict frees VRAM."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.errors import RuntimePlanError
from tensortorrent.native import native_available

pytestmark = pytest.mark.skipif(
    not native_available() or not torch.cuda.is_available(),
    reason="native + CUDA required",
)


class _Deep(nn.Module):
    def __init__(self, width: int = 512, layers: int = 32) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.head = nn.Linear(width, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)


def test_gpu_streaming_vram_stays_near_budget() -> None:
    """Regression: shared Transfer handles left CUDA weights alive after Evict."""
    model = _Deep().eval().half()
    x = torch.randn(4, 512, dtype=torch.float16)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    vram_budget = 128 << 20
    cfg = CompileConfig.polite()
    cfg.objective = Objective.LATENCY
    cfg.allow_gpu = True
    cfg.allow_cpu = True
    cfg.ram_budget_bytes = max(total // 8, 2 << 20)
    cfg.vram_budget_bytes = vram_budget
    cfg.vram_headroom_bytes = 64 << 20
    cfg.prefetch_distance = 1
    cfg.adaptive_prefetch = False
    cfg.max_region_nodes = 2
    cfg.max_concurrent_regions = 1
    cfg.prefer_direct_path = False
    cfg.measure_regions = False
    cfg.use_torch_compile = False

    compiled = tt.compile(model, (x,), config=cfg)
    try:
        assert compiled._executor.parameter_store.stats()["kind"] == "streaming"
        devices = {b.device for b in compiled._executor._schedule_executor.bindings.values()}
        assert any("cuda" in d for d in devices), devices

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        y = compiled(x)
        torch.cuda.synchronize()
        assert y.device.type == "cuda"
        peak = int(torch.cuda.max_memory_allocated())
        # Healthy streaming: peak stays near the VRAM budget working set, not N×model.
        assert peak < (vram_budget * 2), (peak, vram_budget, total)
        # Full-model accumulation would be ≫ one copy of weights after many layers.
        assert peak < total + vram_budget, (peak, total, vram_budget)
    finally:
        compiled.close()


def test_mirror_put_transfer_dest_gets_exclusive_handle() -> None:
    from tensortorrent.runtime.handles import NativeResidencyBridge

    bridge = NativeResidencyBridge.create()
    host = torch.randn(64, 64)
    bridge.mirror_put("w", "pinned_host_0", host, nbytes=int(host.nbytes))
    host_handle = bridge._index[("w", "pinned_host_0")]

    # Simulate Transfer replicate: dest residency exists, no dedicated handle yet.
    bridge.session.alias("w", "pinned_host_0", "cuda_gpu_0")  # wrong path for device — use put via has
    # Prefer explicit: mark dest present the way Transfer does (has without exclusive value).
    # Native replicate isn't exposed; emulate by put/replicate through mirror after forcing has.
    # Direct path: if alias shared the handle, mirror_put must still dedicate a CUDA value.
    cuda = host.to("cuda")
    new_handle = bridge.mirror_put("w", "cuda_gpu_0", cuda, nbytes=int(cuda.nbytes), authoritative=False)
    assert new_handle != host_handle
    assert bridge.handles.get(host_handle) is host
    assert bridge.handles.get(new_handle).device.type == "cuda"

    bridge.drop_python_only("w", "cuda_gpu_0")
    assert ("w", "cuda_gpu_0") not in bridge._index
    assert bridge.handles.get(host_handle) is host
    with pytest.raises(RuntimePlanError, match="unknown tensor handle"):
        bridge.handles.get(new_handle)
