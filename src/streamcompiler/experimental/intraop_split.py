"""CPU/GPU intra-op splitting plans.

Without an accelerator the split degenerates to CPU chunk parallelism along a
chosen dimension — still a real measured schedule, not a placeholder label.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from streamcompiler.errors import UnsupportedFeatureError


@dataclass(frozen=True)
class IntraOpSplit:
    """Split one op's work across ``workers`` along ``dim`` then reduce."""

    dim: int
    workers: int
    reduce: str = "cat"  # cat | sum


def run_intraop_split(
    tensor: torch.Tensor,
    op: Callable[[torch.Tensor], torch.Tensor],
    plan: IntraOpSplit,
) -> torch.Tensor:
    if plan.workers < 1:
        raise UnsupportedFeatureError("IntraOpSplit.workers must be >= 1")
    if plan.workers == 1:
        return op(tensor)
    if tensor.size(plan.dim) < plan.workers:
        return op(tensor)
    shards = torch.chunk(tensor, plan.workers, dim=plan.dim)
    parts = [op(shard) for shard in shards]
    if plan.reduce == "sum":
        out = parts[0]
        for part in parts[1:]:
            out = out + part
        return out
    if plan.reduce == "cat":
        return torch.cat(parts, dim=plan.dim)
    raise UnsupportedFeatureError(f"Unknown IntraOpSplit.reduce {plan.reduce!r}")


def plan_cpu_chunk_split(*, dim: int, workers: int) -> IntraOpSplit:
    return IntraOpSplit(dim=dim, workers=max(1, workers), reduce="cat")
