"""Single-region DirectPlan seeds schedule device-param cache for cancel fallback."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.native import require_native
from tensortorrent.runtime.direct_path import DirectParameter, DirectPlan


def test_single_region_direct_plan_seeds_schedule_device_cache_for_cancel_fallback() -> None:
    if not torch.cuda.is_available():
        return

    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32)).eval().cuda()
    x = torch.randn(4, 64, device="cuda")
    compiled = tt.compile(
        model,
        (x,),
        config=CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=True,
            allow_cpu=False,
            prefer_direct_path=True,
        ),
    )
    try:
        executor = compiled.executor
        plan = executor.direct_plan
        assert isinstance(plan, DirectPlan)
        assert "cuda" in str(plan.device).lower() or "gpu" in str(plan.device).lower()

        se = executor._schedule_executor
        assert se is not None
        assert se._resident_parameter_targets, "expected hoisted parameter destinations"

        placed: dict[str, torch.Tensor] = {}
        for is_input, slot in plan.arg_plan:
            if is_input or not isinstance(slot, DirectParameter):
                continue
            matched = False
            for key, entry in se._persistent_device_param_cache.items():
                if entry[1] is slot.value:
                    placed[key[0]] = slot.value
                    matched = True
                    break
            assert matched, "each DirectParameter device copy must be seeded into schedule cache"

        assert placed, "DirectPlan device copies must be present in schedule cache before fallback"
        before_ids = {name: id(tensor) for name, tensor in placed.items()}
        cache_snapshot = {
            key: (entry[0], id(entry[1]))
            for key, entry in se._persistent_device_param_cache.items()
            if key[0] in placed
        }

        # Serving passes a cancel token → GraphExecutor falls back to schedule.
        cancel = require_native().NativeCancelToken()
        y = compiled.forward_with_cancel_token(cancel, x)
        assert y.shape[-1] == 32

        # Exact same accelerator tensors must still be cached (no second residency set).
        for name, tensor_id in before_ids.items():
            destinations = se._resident_parameter_targets[name]
            key = (name, destinations[0])
            cached = se._persistent_device_param_cache[key]
            assert id(cached[1]) == tensor_id
            assert cache_snapshot[key] == (cached[0], id(cached[1]))
    finally:
        compiled.close()
