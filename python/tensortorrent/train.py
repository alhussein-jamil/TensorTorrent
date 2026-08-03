"""Thin train-loop helpers over schedule-native ``CompiledModule`` autograd."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

from tensortorrent.errors import RuntimePlanError, UnsupportedFeatureError
from tensortorrent.runtime.module import CompiledModule


def _as_scalar_loss(loss: torch.Tensor) -> torch.Tensor:
    """Require a differentiable scalar (or reduce a singleton) for ``backward``."""
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"loss_fn must return a torch.Tensor, got {type(loss).__name__}")
    if loss.ndim == 0:
        return loss
    if loss.numel() == 1:
        return loss.reshape(())
    raise ValueError(
        f"loss_fn must return a scalar tensor for backward, got shape {tuple(loss.shape)}; "
        "reduce inside loss_fn (for example mean/sum)"
    )


def _prediction_device(value: Any) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, (tuple, list)):
        for item in value:
            if (device := _prediction_device(item)) is not None:
                return device
    if isinstance(value, dict):
        for item in value.values():
            if (device := _prediction_device(item)) is not None:
                return device
    return None


def _move_target(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value if value.device == device else value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_target(item, device) for item in value)
    if isinstance(value, list):
        return [_move_target(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_target(item, device) for key, item in value.items()}
    return value


def fit(
    module: CompiledModule,
    batches: Iterable[Any],
    *,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., torch.Tensor],
    epochs: int = 1,
) -> list[float]:
    """Run a simple train loop on a schedule-training ``CompiledModule``.

    Each batch is either an input tensor, ``(inputs, target)`` (where ``inputs``
    may be a tuple/list), or a tuple/list of positional inputs. ``loss_fn`` receives ``(prediction,)`` or
    ``(prediction, target)``. Returns mean loss per epoch.
    """
    if getattr(module, "_closed", False):
        raise RuntimePlanError("CompiledModule is closed")
    if not module.config.allow_training:
        raise UnsupportedFeatureError(
            "tt.fit requires CompileConfig(allow_training=True) so training uses the ExecutableSchedule with autograd."
        )
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError(f"epochs must be an integer, got {type(epochs).__name__}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs!r}")

    module.train()
    # Reuse batches across epochs; materialize one-shot iterators when needed.
    epoch_batches: Iterable[Any] = batches
    if int(epochs) > 1 and not isinstance(batches, (list, tuple)):
        epoch_batches = list(batches)
    history: list[float] = []
    for epoch in range(int(epochs)):
        total = 0.0
        steps = 0
        for batch in epoch_batches:
            if getattr(module, "_closed", False):
                raise RuntimePlanError("CompiledModule is closed")
            optimizer.zero_grad(set_to_none=True)
            if isinstance(batch, torch.Tensor):
                pred = module(batch)
                loss = loss_fn(pred)
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, target = batch
                pred = module(*inputs) if isinstance(inputs, (tuple, list)) else module(inputs)
                if (device := _prediction_device(pred)) is not None:
                    target = _move_target(target, device)
                loss = loss_fn(pred, target)
            elif isinstance(batch, (tuple, list)):
                pred = module(*batch)
                loss = loss_fn(pred)
            else:
                raise TypeError(
                    f"Unsupported batch type {type(batch).__name__}; expected Tensor, "
                    "(inputs, target), or positional input tuple"
                )
            loss = _as_scalar_loss(loss)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        if steps == 0:
            raise ValueError(f"epoch {epoch} produced zero batches; fit requires a non-empty batch iterable")
        history.append(total / steps)
    return history
