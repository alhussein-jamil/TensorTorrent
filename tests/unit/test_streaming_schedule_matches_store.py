"""Schedule streaming flag must match ParameterStore choice."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.ir.graph import OpCode


class _Deep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(12):
            layers += [nn.Linear(64, 64), nn.ReLU()]
        layers.append(nn.Linear(64, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_ram_budget_that_fits_model_does_not_emit_streaming_loads() -> None:
    """When params fit ram_budget, schedule must use resident path — not pinned Loads."""
    model = _Deep().eval()
    x = torch.randn(2, 64)
    total = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    cfg = CompileConfig(
        objective=Objective.LATENCY,
        allow_gpu=True,
        allow_cpu=True,
        allow_concurrent_regions=True,
        max_concurrent_regions=1,
        max_region_nodes=2,
        vram_budget_bytes=max(total // 2, 256 << 10),
        ram_budget_bytes=total * 2,  # fits → ResidentParameterStore
        measure_regions=False,
        validate_numerics=False,
        use_torch_compile=False,
        prefer_direct_path=False,
        allow_nvme_streaming=True,
        prefetch_distance=1,
    )
    compiled = tt.compile(model, (x,), config=cfg)
    try:
        schedule = compiled.specialized.schedule
        assert schedule is not None
        materialize = [
            inst
            for inst in schedule.instructions
            if inst.opcode == OpCode.LOAD and str(inst.attributes.get("kind") or "") == "parameter_materialize"
        ]
        assert materialize == [], f"unexpected streaming Loads: {[i.name for i in materialize]}"
        assert compiled._executor.parameter_store.stats()["kind"] == "resident"
        if torch.cuda.is_available() and "cuda_gpu_0" in compiled.specialized.plan.devices_used:
            # Resident + multi-region GPU still must Evict device copies (VRAM bound).
            device_evicts = [
                i
                for i in schedule.instructions
                if i.opcode == OpCode.EVICT
                and i.attributes.get("kind") == "parameter_evict"
                and not i.attributes.get("staging")
                and "cuda" in str(i.resource)
            ]
            assert device_evicts, "resident GPU schedules must Evict device parameters"
            assert not any(i.attributes.get("staging") for i in schedule.instructions if i.opcode == OpCode.EVICT)
        out = compiled(x)
        assert out.shape == (2, 8)
    finally:
        compiled.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for host→device staging Evicts")
def test_gpu_streaming_emits_staging_and_device_evicts() -> None:
    """Pinned/NUMA staging must be Evicted — else DES sees full-model pinned peak."""
    model = _Deep().eval()
    x = torch.randn(2, 64)
    total = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    cfg = CompileConfig(
        objective=Objective.LATENCY,
        allow_gpu=True,
        allow_cpu=True,
        max_concurrent_regions=1,
        max_region_nodes=2,
        vram_budget_bytes=max(total // 4, 256 << 10),
        ram_budget_bytes=max(total // 8, 64 << 10),
        measure_regions=False,
        validate_numerics=False,
        use_torch_compile=False,
        prefer_direct_path=False,
        allow_nvme_streaming=True,
        prefetch_distance=1,
        adaptive_prefetch=False,
    )
    compiled = tt.compile(model, (x,), config=cfg)
    try:
        assert compiled._executor.parameter_store.stats()["kind"] == "streaming"
        schedule = compiled.specialized.schedule
        assert schedule is not None
        staging = [i for i in schedule.instructions if i.opcode == OpCode.EVICT and i.attributes.get("staging")]
        device = [
            i
            for i in schedule.instructions
            if i.opcode == OpCode.EVICT
            and i.attributes.get("kind") == "parameter_evict"
            and not i.attributes.get("staging")
            and "cuda" in str(i.resource)
        ]
        assert staging, "expected host staging Evicts"
        assert device, "expected CUDA parameter Evicts"
        out = compiled(x)
        assert out.shape == (2, 8)
    finally:
        compiled.close()
