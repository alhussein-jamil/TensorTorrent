"""Virtual-device buffers free on final native allocation release; memory stays bounded."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.backends.mock_accel import make_mock_accel_graph
from streamcompiler.config import CompileConfig
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.resource_graph import merge_graphs
from streamcompiler.native import native_available, require_native

pytestmark = pytest.mark.skipif(not native_available(), reason="native extension required")


def test_session_release_frees_bound_virtual_buffer() -> None:
    native = require_native()
    ctx = native.NativeExecutionContext()
    ctx.set_virtual_backend_config("mock_accel0", memory_bytes=4096)
    buf = ctx.virtual_buffer_from_bytes("mock_accel0", b"\x00" * 512)
    assert ctx.virtual_backend_used_bytes("mock_accel0") == 512
    session = native.NativeResidencySession.from_execution_context(ctx)
    session.put("t", "mock_accel0", 1, 512, True)
    ctx.bind_virtual_buffer("t", "mock_accel0", int(buf))
    freed = session.release("t", "mock_accel0")
    assert freed == 512
    assert ctx.virtual_backend_used_bytes("mock_accel0") == 0
    assert ctx.virtual_backend_live_buffers("mock_accel0") == 0


def test_virtual_memory_bounded_across_long_mock_forwards() -> None:
    """Long mock forward: peak virtual device memory stays under VRAM and stable."""
    vram = 256 * 1024
    mock = make_mock_accel_graph(capacities_bytes=(vram,), delay_hints_s=(0.0,))
    machine = merge_graphs(discover_resource_graph(), mock)

    class _Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(6)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = torch.relu(layer(x))
            return x

    model = _Stack().eval()
    x = torch.randn(4, 16)
    compiled = sc.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
        machine=machine,
    )
    try:
        peaks: list[int] = []
        for i in range(8):
            out = compiled(x)
            assert isinstance(out, torch.Tensor)
            store = compiled.executor._last_schedule_report.parameter_store
            peak = int(store.get("virtual_peak_bytes") or 0)
            peaks.append(peak)
            assert peak <= vram, peaks
            # Drop retained context between forwards so leftover VB free on Drop.
            compiled.executor._schedule_executor._last_native_ctx = None
            if i > 0:
                # Cross-forward peak must stay flat (no leak accumulation).
                assert peak <= peaks[0] + 1024, peaks

        assert max(peaks) > 0, "expected virtual buffers on mock path"
        torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-4)
    finally:
        compiled.close()
        assert compiled.executor._schedule_executor is None
