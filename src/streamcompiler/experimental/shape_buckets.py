"""Dynamic-shape specialization buckets.

Static graphs still refuse unexpected shapes inside a bucket. Across buckets the
runtime picks the specialized module whose batch-size range covers the call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from streamcompiler.errors import UnsupportedFeatureError


@dataclass(frozen=True)
class ShapeBucket:
    """Inclusive batch-size range for one specialized compiled module."""

    name: str
    batch_min: int
    batch_max: int
    module: nn.Module

    def covers(self, batch: int) -> bool:
        return self.batch_min <= batch <= self.batch_max


class BucketedModule(nn.Module):
    """Dispatch ``forward`` to the specialized module for the input batch size."""

    def __init__(self, buckets: Sequence[ShapeBucket], *, batch_dim: int = 0) -> None:
        super().__init__()
        if not buckets:
            raise UnsupportedFeatureError("BucketedModule requires at least one ShapeBucket")
        ordered = tuple(sorted(buckets, key=lambda b: (b.batch_min, b.batch_max, b.name)))
        for prev, cur in zip(ordered, ordered[1:], strict=False):
            if cur.batch_min <= prev.batch_max:
                raise UnsupportedFeatureError(
                    f"Shape buckets overlap: {prev.name}[{prev.batch_min},{prev.batch_max}] "
                    f"vs {cur.name}[{cur.batch_min},{cur.batch_max}]"
                )
        self.buckets = ordered
        self.batch_dim = int(batch_dim)
        for bucket in ordered:
            self.add_module(f"bucket_{bucket.name}", bucket.module)

    def _batch_size(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
        if args and isinstance(args[0], torch.Tensor):
            return int(args[0].shape[self.batch_dim])
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[self.batch_dim])
        raise UnsupportedFeatureError("BucketedModule needs a tensor input to read the batch size")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        batch = self._batch_size(args, kwargs)
        for bucket in self.buckets:
            if bucket.covers(batch):
                return getattr(self, f"bucket_{bucket.name}")(*args, **kwargs)
        ranges = ", ".join(f"{b.name}:[{b.batch_min},{b.batch_max}]" for b in self.buckets)
        raise UnsupportedFeatureError(
            f"Batch size {batch} is outside every specialized bucket ({ranges}). "
            "Recompile with an example covering this shape."
        )

    def close(self) -> None:
        for bucket in self.buckets:
            close = getattr(bucket.module, "close", None)
            if callable(close):
                close()
