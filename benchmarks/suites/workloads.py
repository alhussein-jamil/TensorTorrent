"""Benchmark workloads: synthetic CI smoke + recognizable public models."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """Deterministic CI/smoke workload (fits anywhere)."""

    def __init__(self, width: int = 128, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPStack(nn.Module):
    def __init__(self, width: int = 512, depth: int = 8) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        return x + self.mlp(self.norm2(x))


class DeepMLP(nn.Module):
    """Parameter-heavy stack used for VRAM pressure / beyond-VRAM cases."""

    def __init__(self, width: int, depth: int, out_features: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = torch.relu(blk(x))
        return self.head(x)


def param_bytes(model: nn.Module) -> int:
    return sum(int(p.numel()) * int(p.element_size()) for p in model.parameters())


def deep_mlp_for_bytes(target_bytes: int, *, width: int = 4096) -> tuple[int, int]:
    per_layer = (width * width + width) * 4
    depth = max(2, int(target_bytes / per_layer) + 1)
    return width, depth


FIT_WORKLOADS: dict[str, tuple[Callable[[], nn.Module], tuple[int, ...]]] = {
    "mlp_512x8": (lambda: MLPStack(512, 8), (32, 512)),
    "transformer_256": (lambda: TransformerBlock(256, 4), (8, 32, 256)),
    "mlp_2048x8": (lambda: MLPStack(2048, 8), (8, 2048)),
}

SMOKE_WORKLOADS: dict[str, tuple[Callable[[], nn.Module], tuple[int, ...]]] = {
    "tiny_mlp": (lambda: TinyMLP(128, 4), (8, 128)),
}
