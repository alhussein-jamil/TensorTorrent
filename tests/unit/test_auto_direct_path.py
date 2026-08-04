"""Eligible resident single-region graphs use the direct path by default."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

import tensortorrent as tt


def test_prefer_direct_path_default_on() -> None:
    os.environ.pop("TT_DIRECT_PATH", None)
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(4, 32)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False,
            allow_concurrent_regions=False,
            validate_numerics=False,
        ),
    )
    try:
        assert compiled._executor.direct_plan is not None
        torch.testing.assert_close(compiled(x), model(x), atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()


def test_prefer_direct_path_false_uses_schedule() -> None:
    os.environ.pop("TT_DIRECT_PATH", None)
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(4, 32)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False,
            allow_concurrent_regions=False,
            validate_numerics=False,
            prefer_direct_path=False,
        ),
    )
    try:
        assert compiled._executor.direct_plan is None
        torch.testing.assert_close(compiled(x), model(x), atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()
