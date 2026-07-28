"""Activation rematerialization markers for the recompute overflow policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from streamcompiler.errors import RuntimePlanError


@dataclass
class NeedsRecompute:
    """Dropped activation that must be rematerialized by re-running a producer."""

    producer_region_id: str
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    def as_dict(self) -> dict[str, Any]:
        return {
            "producer_region_id": self.producer_region_id,
            "nbytes": self.nbytes,
            "shape": list(self.shape),
            "dtype": str(self.dtype).replace("torch.", ""),
        }


def is_needs_recompute(value: Any) -> bool:
    return isinstance(value, NeedsRecompute)


def mark_for_recompute(value: torch.Tensor, *, producer_region_id: str) -> NeedsRecompute:
    if not isinstance(value, torch.Tensor):
        raise RuntimePlanError(f"recompute marker needs a tensor, got {type(value)!r}")
    return NeedsRecompute(
        producer_region_id=producer_region_id,
        nbytes=int(value.numel() * value.element_size()),
        shape=tuple(value.shape),
        dtype=value.dtype,
    )
