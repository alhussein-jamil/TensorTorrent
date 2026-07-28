"""Tensor parallelism helpers (host-staged shard / reduce).

Unequal GPU TP needs device collectives; this module ships the host-staged path
that always works on CPU tensors and is what the planner falls back to when
vendor NCCL/RCCL/oneCCL are unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from streamcompiler.communication import HostStagedComm
from streamcompiler.errors import UnsupportedFeatureError


def shard_tensor(tensor: torch.Tensor, *, dim: int, world_size: int) -> tuple[torch.Tensor, ...]:
    """Split ``tensor`` along ``dim`` into ``world_size`` contiguous shards."""
    if world_size < 1:
        raise UnsupportedFeatureError("world_size must be >= 1")
    if tensor.size(dim) % world_size != 0:
        raise UnsupportedFeatureError(
            f"Cannot evenly shard size {tensor.size(dim)} on dim {dim} into {world_size} parts"
        )
    return tuple(torch.chunk(tensor, world_size, dim=dim))


def gather_shards(shards: Sequence[torch.Tensor], *, dim: int) -> torch.Tensor:
    return torch.cat(tuple(shards), dim=dim)


def tensor_parallel_linear_host_staged(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    world_size: int,
) -> torch.Tensor:
    """Column-parallel linear on host: shard weight rows, local matmul, allreduce sum."""
    if world_size == 1:
        out = x.matmul(weight.t())
        return out + bias if bias is not None else out
    shards = shard_tensor(weight, dim=0, world_size=world_size)
    bias_shards = shard_tensor(bias, dim=0, world_size=world_size) if bias is not None else (None,) * world_size
    partials: list[torch.Tensor] = []
    for w_shard, b_shard in zip(shards, bias_shards, strict=True):
        # Row-shard of weight means each rank produces a slice of the output features.
        local = x.matmul(w_shard.t())
        if b_shard is not None:
            local = local + b_shard
        partials.append(local)
    # Concatenate feature shards (column-parallel gather), not an allreduce.
    return gather_shards(partials, dim=-1)


def allreduce_sum_host(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sum a list of peer tensors on the host (unequal-GPU fallback)."""
    result = HostStagedComm().allreduce(list(tensors), devices=tuple(f"peer_{i}" for i in range(len(tensors))))
    if not isinstance(result, torch.Tensor):
        raise UnsupportedFeatureError("host-staged allreduce did not return a tensor")
    return result
