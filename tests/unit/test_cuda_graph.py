"""CUDA graph replay for export-free eager GPU DirectPlan."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.runtime.cuda_graph import CudaGraphReplay


def test_cuda_graph_replay_matches_eager_linear() -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = nn.Linear(32, 16).cuda().eval()
    x = torch.randn(4, 32, device="cuda")
    with torch.inference_mode():
        ref = model(x).clone()

    replay = CudaGraphReplay(model)
    out = None
    for _ in range(CudaGraphReplay.warmup_calls + 2):
        out = replay(x)
    assert replay.captured
    assert out is not None
    assert torch.allclose(out, ref, atol=1e-5)
    # Replay must not alias the static output buffer.
    first = out.clone()
    y = torch.randn(4, 32, device="cuda")
    second = replay(y)
    assert not torch.equal(first, second)
    with torch.inference_mode():
        assert torch.allclose(second, model(y), atol=1e-5)


def test_cuda_graph_skips_non_tensor_args() -> None:
    def call(tensor: torch.Tensor, scale: float) -> torch.Tensor:
        return tensor * scale

    replay = CudaGraphReplay(call)
    x = torch.ones(2)
    if torch.cuda.is_available():
        x = x.cuda()
    out = replay(x, 2.0)
    assert replay.skipped_reason == "non_tensor_args"
    assert replay.captured is False
    assert torch.equal(out.cpu(), torch.ones(2) * 2)


def test_export_free_gpu_captures_cuda_graph() -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = nn.Linear(32, 16).eval()
    x = torch.randn(4, 32)
    with torch.inference_mode():
        ref = model(x).clone()
    compiled = tt.compile(model, (x,), config=CompileConfig(use_torch_compile=False))
    try:
        assert compiled.specialized.validation.get("eager_fused_gpu") is True
        out = None
        for _ in range(CudaGraphReplay.warmup_calls + 2):
            out = compiled(x)
        assert compiled.executor.cuda_graph_captured is True
        assert out is not None
        assert torch.allclose(out.cpu(), ref, atol=1e-5)
    finally:
        compiled.close()


def test_torch_compile_skips_eager_cuda_graph() -> None:
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    model = nn.Linear(16, 8).eval()
    x = torch.randn(2, 16)
    compiled = tt.compile(model, (x,), config=CompileConfig(use_torch_compile=True))
    try:
        compiled(x)
        assert compiled.specialized.validation.get("eager_fused_gpu") is True
        assert compiled.executor.cuda_graph_captured is False
        assert getattr(compiled.executor, "_cuda_graph", None) is None
    finally:
        compiled.close()
