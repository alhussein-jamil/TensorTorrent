"""Dynamic shape plan families (bucket selection at runtime)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ShapeBucket:
    name: str
    batch_min: int
    batch_max: int
    seq_max: int
    attributes: dict[str, Any] = field(default_factory=dict)


DEFAULT_BUCKETS: tuple[ShapeBucket, ...] = (
    ShapeBucket("decode_b1_s512", 1, 1, 512),
    ShapeBucket("prefill_b1_s4096", 1, 1, 4096),
    ShapeBucket("batch_2_4", 2, 4, 4096),
    ShapeBucket("large_prefill", 1, 1, 32768),
)


def select_bucket(
    batch: int,
    seq: int,
    buckets: tuple[ShapeBucket, ...] = DEFAULT_BUCKETS,
) -> ShapeBucket | None:
    """Select the tightest matching bucket; None means use conservative fallback."""
    matches = [b for b in buckets if b.batch_min <= batch <= b.batch_max and seq <= b.seq_max]
    if not matches:
        return None
    return min(matches, key=lambda b: (b.batch_max - b.batch_min, b.seq_max))


@dataclass
class PlanFamily:
    """Set of specialized plans keyed by shape bucket name."""

    fingerprint: str
    plans: dict[str, Any] = field(default_factory=dict)
    fallback: str | None = None

    def choose(self, batch: int, seq: int) -> Any:
        bucket = select_bucket(batch, seq)
        if bucket is None:
            if self.fallback and self.fallback in self.plans:
                return self.plans[self.fallback]
            raise KeyError(f"No plan bucket for batch={batch} seq={seq} and no fallback")
        if bucket.name not in self.plans:
            if self.fallback and self.fallback in self.plans:
                return self.plans[self.fallback]
            raise KeyError(f"Missing specialized plan for bucket {bucket.name}")
        return self.plans[bucket.name]
