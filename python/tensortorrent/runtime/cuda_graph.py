"""CUDA graph replay for export-free eager GPU DirectPlan.

Captures the original-module CUDA call after a few warmup forwards. Inductor
already has its own cudagraphs — this wrapper is only for the eager module
path. Non-tensor args skip capture (replay would freeze Python scalars).
"""

from __future__ import annotations

from typing import Any

import torch


def _clone_outputs(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_outputs(v) for v in value)
    if isinstance(value, list):
        return [_clone_outputs(v) for v in value]
    return value


def _all_tensors(args: tuple[Any, ...]) -> bool:
    return bool(args) and all(torch.is_tensor(a) for a in args)


class CudaGraphReplay:
    """Warmup, capture, then replay ``call(*cuda_tensors)``."""

    warmup_calls = 3

    def __init__(self, call: Any) -> None:
        self._call = call
        self._warmup_left = int(self.warmup_calls)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_in: tuple[Any, ...] | None = None
        self._static_out: Any = None
        self.captured = False
        self.skipped_reason: str | None = None

    def __call__(self, *args: Any) -> Any:
        if self.skipped_reason is not None:
            return self._call(*args)
        if not _all_tensors(args):
            self.skipped_reason = "non_tensor_args"
            return self._call(*args)
        if self._graph is None:
            if self._warmup_left > 0:
                self._warmup_left -= 1
                return self._call(*args)
            return self._capture(args)
        return self._replay(args)

    def _capture(self, args: tuple[Any, ...]) -> Any:
        static_in = tuple(a.detach().clone() for a in args)
        try:
            self._call(*static_in)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._call(*static_in)
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - keep eager if capture is illegal
            self.skipped_reason = "capture_failed"
            return self._call(*args)
        self._graph = graph
        self._static_in = static_in
        self._static_out = static_out
        self.captured = True
        return _clone_outputs(static_out)

    def _replay(self, args: tuple[Any, ...]) -> Any:
        static_in = self._static_in
        graph = self._graph
        if static_in is None or graph is None:
            return self._call(*args)
        if len(args) != len(static_in):
            self.skipped_reason = "arity_changed"
            self._graph = None
            return self._call(*args)
        try:
            for dst, src in zip(static_in, args, strict=True):
                if dst.shape != src.shape or dst.dtype != src.dtype:
                    self.skipped_reason = "shape_or_dtype_changed"
                    self._graph = None
                    return self._call(*args)
                dst.copy_(src, non_blocking=True)
            graph.replay()
        except Exception:  # noqa: BLE001
            self.skipped_reason = "replay_failed"
            self._graph = None
            return self._call(*args)
        return _clone_outputs(self._static_out)
