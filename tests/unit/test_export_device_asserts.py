"""Regression: export device asserts must not block schedule-managed CUDA placement."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.compile.regions import _drop_device_metadata_asserts


def test_drop_device_metadata_asserts_passthrough() -> None:
    class M(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            torch.ops.aten._assert_tensor_metadata.default(x, None, None, x.dtype, device=torch.device("cpu"))
            return x + 1

    m = M().eval()
    x = torch.zeros(2, 2)
    ep = torch.export.export(m, (x,), strict=False)
    gm = ep.module()
    before = sum(1 for n in gm.graph.nodes if n.op == "call_function")
    removed = _drop_device_metadata_asserts(gm)
    assert removed >= 1
    after = sum(1 for n in gm.graph.nodes if n.op == "call_function" and "assert_tensor_metadata" in str(n.target))
    assert after == 0
    assert before > after


def test_rewrite_hardcoded_cpu_devices_on_arange() -> None:
    from tensortorrent.backends.torch_device import _rewrite_hardcoded_cpu_devices

    class M(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            idx = torch.arange(x.shape[0], device="cpu")
            return x[idx]

    m = M().eval()
    x = torch.randn(4, 3)
    ep = torch.export.export(m, (x,), strict=False)
    gm = ep.module()
    n = _rewrite_hardcoded_cpu_devices(gm, "cuda:0")
    assert n >= 1
    for node in gm.graph.nodes:
        if node.op == "call_function" and "arange" in str(node.target):
            assert node.kwargs.get("device") == torch.device("cuda:0")


def test_embedding_streaming_survives_cpu_export_asserts() -> None:
    """Small embedding model: CPU export + CUDA streaming must not trip device asserts."""
    if not torch.cuda.is_available():
        return

    class EmbMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(128, 64)
            self.fc = nn.Linear(64, 32)

        def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # mask participates like HF attention_mask (export may insert device asserts)
            x = self.emb(ids)
            x = x * mask.to(x.dtype).unsqueeze(-1)
            return self.fc(x.mean(dim=1))

    torch.manual_seed(0)
    model = EmbMLP().eval()
    ids = torch.randint(0, 128, (2, 8))
    mask = torch.ones(2, 8)
    with torch.no_grad():
        expected = model(ids, mask)

    vram = int(torch.cuda.get_device_properties(0).total_memory)
    # Force multi-region streaming relative to a tiny budget.
    budget = max(8 << 20, int(sum(p.numel() * p.element_size() for p in model.parameters()) * 0.4))
    cfg = tt.CompileConfig(
        allow_cpu=False,
        allow_gpu=True,
        measure_regions=False,
        use_torch_compile=False,
        validate_numerics=False,
        vram_budget_bytes=min(budget, vram),
        max_region_nodes=4,
        prefetch_distance=1,
    )
    compiled = tt.compile(model, example_inputs=(ids, mask), config=cfg)
    try:
        with torch.no_grad():
            out = compiled(ids, mask)
        assert torch.allclose(out.cpu(), expected.cpu(), atol=1e-4, rtol=1e-4)
        devices = list(compiled.specialized.plan.devices_used)
        assert any(str(d).startswith("cuda") for d in devices), devices
    finally:
        compiled.close()
