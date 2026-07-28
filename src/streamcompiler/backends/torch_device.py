"""Shared region realization for every backend that PyTorch can target.

CPU, CUDA, ROCm, MPS and XPU all execute the *same* partitioned subgraphs; only
the torch device string and the transfer semantics differ. Keeping this logic in
one place is what prevents per-vendor copies of the compiler.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch

from streamcompiler.backends.base import (
    BenchmarkResult,
    CompiledRegion,
    KernelCandidate,
    RegionSource,
)
from streamcompiler.errors import BackendError


def _has_no_hooks(module: Any) -> bool:
    """True when calling ``forward`` directly is equivalent to calling the module."""
    for attr in (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_forward_hooks_with_kwargs",
    ):
        if getattr(module, attr, None):
            return False
    import torch.nn.modules.module as module_mod

    return not (
        module_mod._global_forward_pre_hooks or module_mod._global_forward_hooks or module_mod._global_backward_hooks
    )


class _RegionCallable:
    """Callable wrapper that places inputs on the target device before running.

    Region subgraphs carry no hooks, so this calls ``forward`` directly and skips
    ``nn.Module.__call__``'s hook dispatch, which is a measurable share of the
    runtime for small regions.
    """

    __slots__ = ("module", "torch_device", "region_id", "_needs_move", "_run")

    def __init__(self, module: Any, torch_device: str, region_id: str) -> None:
        self.module = module
        self.torch_device = torch_device
        self.region_id = region_id
        self._needs_move = torch.device(torch_device).type != "cpu"
        self._run = module.forward if _has_no_hooks(module) else module

    def __call__(self, *inputs: Any) -> Any:
        if self._needs_move:
            inputs = tuple(
                t.to(self.torch_device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in inputs
            )
        return self._run(*inputs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_RegionCallable(region={self.region_id!r}, device={self.torch_device!r})"


def compile_region_for_torch_device(
    region: RegionSource,
    candidate: KernelCandidate,
    *,
    backend_id: str,
    torch_device: str,
) -> CompiledRegion:
    """Realize a region subgraph on ``torch_device`` without rewriting the graph."""
    module = region.module
    if module is None or not callable(module):
        raise BackendError(
            f"Region {region.region_id} has no executable module; region partitioning must supply a torch.nn.Module"
        )
    if torch.device(torch_device).type != "cpu" and hasattr(module, "to"):
        module = module.to(torch_device)
    return CompiledRegion(
        region_id=region.region_id,
        device=candidate.device,
        backend_id=backend_id,
        executable=_RegionCallable(module, torch_device, region.region_id),
        dtype=candidate.dtype,
        torch_device=torch_device,
        attributes={
            "impl": "torch_fx_subgraph",
            "aten_ops": list(region.aten_ops),
            "input_names": list(region.input_names),
            "output_names": list(region.output_names),
        },
    )


def execute_region_on_torch_device(executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]:
    """Execute a compiled region and normalize its outputs to a tuple."""
    result = executable.executable(*inputs)
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def benchmark_region_on_torch_device(
    candidate: KernelCandidate,
    executable: CompiledRegion,
    example_inputs: Sequence[Any],
    *,
    warmup: int = 2,
    iters: int = 5,
    synchronize: Any = None,
) -> BenchmarkResult:
    """Measure real region latency by running it on real tensors."""
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            executable.executable(*example_inputs)
        if synchronize is not None:
            synchronize()
        best = float("inf")
        total = 0.0
        for _ in range(max(1, iters)):
            start = time.perf_counter()
            executable.executable(*example_inputs)
            if synchronize is not None:
                synchronize()
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)
            total += elapsed
    nbytes = sum(int(t.numel() * t.element_size()) for t in example_inputs if isinstance(t, torch.Tensor))
    return BenchmarkResult(
        candidate=candidate,
        latency_s=best,
        memory_bytes=nbytes,
        measured=True,
        notes=(
            f"region {candidate.region_id} on {candidate.device} "
            f"best={best:.6f}s mean={total / max(1, iters):.6f}s iters={iters}"
        ),
    )
