"""Autograd-safe tensor device moves for schedule training Transfers."""

from __future__ import annotations

from typing import Any, cast

import torch


class GradDeviceMove(torch.autograd.Function):
    """``tensor.to(device)`` that routes gradients back to the source device."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        ctx.source_device = tensor.device
        if tensor.device == device:
            return tensor
        return tensor.to(device)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        if grad_output.device == ctx.source_device:
            return grad_output, None
        return grad_output.to(ctx.source_device), None


def move_for_training(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Move ``tensor`` for a train-mode Transfer while preserving autograd."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"move_for_training expects a torch.Tensor, got {type(tensor).__name__}")
    target = torch.device(device) if not isinstance(device, torch.device) else device
    return cast(torch.Tensor, GradDeviceMove.apply(tensor, target))  # type: ignore[no-untyped-call]
