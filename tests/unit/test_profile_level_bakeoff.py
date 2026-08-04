"""profile_level gates AOT + bake-off cost during region compile."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.backends.torch_device import _COMPILE_CACHE


def _bindings(compiled: tt.CompiledModule):
    for name in ("_executor", "executor", "graph_executor"):
        obj = getattr(compiled, name, None)
        if obj is not None and hasattr(obj, "bindings"):
            return obj.bindings
    raise AssertionError("compiled module has no bindings")


def test_coarse_skips_aot_bakeoff_attributes() -> None:
    _COMPILE_CACHE.clear()
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    x = torch.randn(4, 32)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
            use_torch_compile=True,
            measure_regions=True,
            allow_gpu=False,
            allow_concurrent_regions=False,
            profile_level="coarse",
            validate_numerics=False,
            region_measure_iters=1,
        ),
    )
    try:
        attrs = next(iter(_bindings(compiled).values())).compiled.attributes
        assert attrs.get("profile_level") == "coarse"
        assert attrs.get("selected_candidate") == "torch_compile_inductor"
        assert "aot_compile_time_s" not in attrs
        assert "candidate_latencies_s" not in attrs
        torch.testing.assert_close(compiled(x), model(x), atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()
