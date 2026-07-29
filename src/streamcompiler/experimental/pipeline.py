"""Pipeline microbatching on a sequence of stage callables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from streamcompiler.errors import UnsupportedFeatureError

StageFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class MicrobatchPlan:
    """Split the batch dim into ``microbatch_size`` chunks and run stages in order."""

    microbatch_size: int
    stages: tuple[StageFn, ...]

    def __post_init__(self) -> None:
        if self.microbatch_size < 1:
            raise UnsupportedFeatureError("microbatch_size must be >= 1")
        if not self.stages:
            raise UnsupportedFeatureError("pipeline needs at least one stage")


def run_pipeline_microbatched(plan: MicrobatchPlan, batch: torch.Tensor) -> torch.Tensor:
    """Run stages over microbatches and concatenate outputs along dim 0.

    Stages execute sequentially per microbatch (correct 1F1B-style fill is a
    schedule overlay when multiple devices exist). On one host this still gives
    the numerical result of pipelined stage composition.
    """
    if batch.dim() < 1:
        raise UnsupportedFeatureError("pipeline microbatch needs a batched tensor")
    chunks = torch.split(batch, plan.microbatch_size, dim=0)
    outputs: list[torch.Tensor] = []
    for chunk in chunks:
        value = chunk
        for stage in plan.stages:
            value = stage(value)
        outputs.append(value)
    return torch.cat(outputs, dim=0)


def split_batch_evenly(batch: torch.Tensor, parts: int) -> tuple[torch.Tensor, ...]:
    if parts < 1:
        raise UnsupportedFeatureError("parts must be >= 1")
    return tuple(torch.chunk(batch, parts, dim=0))
