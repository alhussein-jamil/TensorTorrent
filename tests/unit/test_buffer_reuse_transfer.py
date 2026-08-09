"""Buffer reuse must not alias live activations across Transfer/Evict schedules."""

from __future__ import annotations

import torch
import torch.nn as nn

import tensortorrent as tt


def test_beyond_vram_residency_disables_buffer_reuse_for_rotary_fanout() -> None:
    """Embedding fan-out into cos/sin paths must stay correct under residency Transfer."""
    if not torch.cuda.is_available():
        return

    class BigRotary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(4096, 512)
            self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, 512, 2).float() / 512)))

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            x = self.emb(ids)
            t = ids.shape[1]
            pos = torch.arange(t, device=ids.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(pos, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(x.dtype)
            sin = emb.sin().to(x.dtype)
            return x * cos + torch.roll(x, 1, dims=-1) * sin

    torch.manual_seed(0)
    model = BigRotary().eval().cpu()
    ids = torch.randint(0, 4096, (2, 16))
    with torch.no_grad():
        expected = model(ids)

    emb_bytes = int(model.emb.weight.numel() * model.emb.weight.element_size())
    budget = int(emb_bytes * 1.2)
    cfg = tt.CompileConfig(
        allow_cpu=False,
        allow_gpu=True,
        measure_regions=False,
        use_torch_compile=False,
        validate_numerics=False,
        vram_budget_bytes=budget,
        max_region_nodes=1,
        prefetch_distance=0,
        allow_concurrent_regions=False,
        max_concurrent_regions=1,
    )
    compiled = tt.compile(model, example_inputs=(ids,), config=cfg)
    try:
        assert compiled.executor._reuse_assignment == {}
        with torch.no_grad():
            out = compiled(ids)
        torch.testing.assert_close(out.cpu(), expected.cpu(), atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()
