"""Thin train-loop helpers over schedule-native ``CompiledModule`` autograd."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

from streamcompiler.errors import RuntimePlanError, UnsupportedFeatureError
from streamcompiler.runtime.module import CompiledModule


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


def fit(
    module: CompiledModule,
    batches: Iterable[Any],
    *,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., torch.Tensor],
    epochs: int = 1,
) -> list[float]:
    """Run a simple train loop on a schedule-training ``CompiledModule``.

    Each batch is either an input tensor, ``(inputs, target)``, or a tuple/list of
    positional inputs. ``loss_fn`` receives ``(prediction,)`` or
    ``(prediction, target)``. Returns mean loss per epoch.
    """
    if getattr(module, "_closed", False):
        raise RuntimePlanError("CompiledModule is closed")
    if not module.config.allow_training:
        raise UnsupportedFeatureError(
            "sc.fit requires CompileConfig(allow_training=True) so training uses the ExecutableSchedule with autograd."
        )
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
            elif isinstance(batch, (tuple, list)) and len(batch) == 2 and not isinstance(batch[0], (tuple, list)):
                inputs, target = batch
                pred = module(inputs) if isinstance(inputs, torch.Tensor) else module(*inputs)
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
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        if steps == 0:
            raise ValueError(f"epoch {epoch} produced zero batches; fit requires a non-empty batch iterable")
        history.append(total / steps)
    return history
