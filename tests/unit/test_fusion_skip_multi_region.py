"""Sequential graphs fuse without a discarded multi-region specialize."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt


def test_sequential_fusion_skips_multi_region_specialize() -> None:
    layers: list[nn.Module] = []
    for _ in range(6):
        layers.extend([nn.Linear(32, 32), nn.ReLU()])
    model = nn.Sequential(*layers).eval()
    x = torch.randn(4, 32)
    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=False,
            allow_concurrent_regions=True,
            max_concurrent_regions=0,
            max_region_nodes=2,
            validate_numerics=False,
        ),
    )
    try:
        assert compiled.specialized.validation.get("fusion_skipped_multi_region") is True
        assert compiled.specialized.validation.get("fused_after_sequential_decision") is True
        assert len(compiled._program.regions) == 1
        torch.testing.assert_close(compiled(x), model(x), atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()
