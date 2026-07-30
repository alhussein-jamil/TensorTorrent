"""Contention-aware cost adjustments for concurrent resource use.

Default coefficients are analytic priors. Call
:func:`set_measured_compute_contention` after a sequential-vs-concurrent bench to
fold a measured multiplier into the transfer/compute model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContentionFactors:
    compute: float = 1.0
    transfer: float = 1.0
    storage: float = 1.0


_MEASURED_COMPUTE: float | None = None


def set_measured_compute_contention(factor: float | None) -> None:
    """Install a process-local measured compute contention multiplier."""
    global _MEASURED_COMPUTE
    _MEASURED_COMPUTE = None if factor is None else max(1.0, float(factor))


def concurrent_slowdown(
    *,
    active_compute: int,
    active_transfers: int,
    active_storage: int,
) -> ContentionFactors:
    """Return multiplicative slowdowns under concurrent pressure."""
    base_compute = 1.0 + 0.05 * max(0, active_compute - 1)
    if _MEASURED_COMPUTE is not None and active_compute > 1:
        base_compute = max(base_compute, _MEASURED_COMPUTE)
    transfer = 1.0 + 0.15 * max(0, active_transfers - 1) + 0.05 * active_compute
    storage = 1.0 + 0.20 * max(0, active_storage - 1) + 0.05 * active_transfers
    return ContentionFactors(compute=base_compute, transfer=transfer, storage=storage)
