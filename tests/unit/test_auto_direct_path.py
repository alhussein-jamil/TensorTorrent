"""Eligible resident single-region graphs use the direct path automatically.

Direct-path selection is a correctness property (single Compute, resident
parameters, no streaming/training/cancellation), not a user knob. The
compile-time gate in ``build_direct_plan`` returns ``None`` whenever any of
those correctness conditions is violated; otherwise the runtime uses the
zero-overhead call.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt


def test_direct_path_auto_selected_when_eligible() -> None:
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


def test_force_schedule_helper_falls_back_to_schedule() -> None:
    """The private ``_testing.force_schedule_path`` seam disables the direct plan."""
    from tensortorrent._testing import force_schedule_path

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
        force_schedule_path(compiled)
        assert compiled._executor.direct_plan is None
        torch.testing.assert_close(compiled(x), model(x), atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()
