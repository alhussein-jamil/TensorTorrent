"""Contention-aware cost adjustments for concurrent resource use.

Do not assume GPU uploads, CPU compute, and NVMe reads all run at full
bandwidth simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContentionFactors:
    compute: float = 1.0
    transfer: float = 1.0
    storage: float = 1.0


def concurrent_slowdown(
    *,
    active_compute: int,
    active_transfers: int,
    active_storage: int,
) -> ContentionFactors:
    """Return multiplicative slowdowns under concurrent pressure."""
    compute = 1.0 + 0.05 * max(0, active_compute - 1)
    transfer = 1.0 + 0.15 * max(0, active_transfers - 1) + 0.05 * active_compute
    storage = 1.0 + 0.20 * max(0, active_storage - 1) + 0.05 * active_transfers
    return ContentionFactors(compute=compute, transfer=transfer, storage=storage)


def adjust_latency(base_s: float, factor: float) -> float:
    return max(0.0, base_s * factor)
