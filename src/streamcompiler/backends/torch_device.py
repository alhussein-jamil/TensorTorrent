"""Shared region realization for every backend that PyTorch can target.

CPU, CUDA, ROCm, MPS and XPU all execute the *same* partitioned subgraphs; only
the torch device string and the transfer semantics differ. Keeping this logic in
one place is what prevents per-vendor copies of the compiler.
"""

from __future__ import annotations

import hashlib
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

# Process-local cache: fingerprint -> compiled callable. Avoids recompiling the
# same region subgraph when specialization re-runs on an unchanged machine.
_COMPILE_CACHE: dict[str, Any] = {}


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

    def __init__(
        self, module: Any, torch_device: str, region_id: str, *, schedule_managed_placement: bool = True
    ) -> None:
        self.module = module
        self.torch_device = torch_device
        self.region_id = region_id
        # Schedule Transfer ops own residency; compute must not hide ``.to``.
        self._needs_move = (not schedule_managed_placement) and torch.device(torch_device).type != "cpu"
        self._run = module.forward if _has_no_hooks(module) else module

    def __call__(self, *inputs: Any) -> Any:
        if self._needs_move:
            inputs = tuple(
                t.to(self.torch_device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in inputs
            )
        return self._run(*inputs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_RegionCallable(region={self.region_id!r}, device={self.torch_device!r})"


class _CompiledRegionCallable:
    """Runs a ``torch.compile`` executable with explicit eager FX fallback.

    Fallback executes the real region module, never metadata placeholders.
    """

    __slots__ = (
        "region_id",
        "torch_device",
        "eager",
        "compiled",
        "impl",
        "compile_time_s",
        "fallback_reason",
        "_needs_move",
        "_use_compiled",
    )

    def __init__(
        self,
        *,
        region_id: str,
        torch_device: str,
        eager: Any,
        compiled: Any | None,
        impl: str,
        compile_time_s: float,
        fallback_reason: str | None,
        schedule_managed_placement: bool = True,
    ) -> None:
        self.region_id = region_id
        self.torch_device = torch_device
        self.eager = eager
        self.compiled = compiled
        self.impl = impl
        self.compile_time_s = compile_time_s
        self.fallback_reason = fallback_reason
        self._needs_move = (not schedule_managed_placement) and torch.device(torch_device).type != "cpu"
        self._use_compiled = compiled is not None and fallback_reason is None

    def _place(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not self._needs_move:
            return inputs
        return tuple(t.to(self.torch_device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in inputs)

    def __call__(self, *inputs: Any) -> Any:
        placed = self._place(inputs)
        if self._use_compiled:
            try:
                compiled = self.compiled
                assert compiled is not None
                return compiled(*placed)
            except Exception:
                self._use_compiled = False
                self.fallback_reason = self.fallback_reason or "runtime_compile_failure"
        return self.eager(*placed)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"_CompiledRegionCallable(region={self.region_id!r}, impl={self.impl!r}, fallback={self.fallback_reason!r})"
        )


def region_compile_fingerprint(
    region: RegionSource,
    *,
    torch_device: str,
    backend: str,
    dtype: str,
    machine_fingerprint: str = "",
) -> str:
    """Hardware + software key for compiled-region caching."""
    payload = "|".join(
        [
            region.region_id,
            torch_device,
            backend,
            dtype,
            machine_fingerprint,
            ",".join(region.aten_ops),
            ",".join(region.input_names),
            ",".join(region.output_names),
            str(torch.__version__),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _eager_runner(module: Any) -> Any:
    return module.forward if _has_no_hooks(module) else module


def _try_torch_compile(
    module: Any,
    *,
    region_id: str,
    torch_device: str,
    compile_backend: str,
    example_inputs: Sequence[Any] | None,
    cache_key: str,
) -> tuple[Any | None, float, str | None]:
    """Attempt Inductor (or configured backend). Return (compiled, time_s, reason)."""
    if cache_key in _COMPILE_CACHE:
        return _COMPILE_CACHE[cache_key], 0.0, None
    if not hasattr(torch, "compile"):
        return None, 0.0, "torch.compile unavailable"
    start = time.perf_counter()
    try:
        compiled_mod = torch.compile(module, backend=compile_backend, fullgraph=False)
        runner = _eager_runner(compiled_mod)
        if example_inputs:
            with torch.inference_mode():
                placed = tuple(
                    t.to(torch_device)
                    if isinstance(t, torch.Tensor) and torch.device(torch_device).type != "cpu"
                    else t
                    for t in example_inputs
                )
                runner(*placed)
        elapsed = time.perf_counter() - start
        _COMPILE_CACHE[cache_key] = runner
        return runner, elapsed, None
    except Exception as exc:  # noqa: BLE001 - compile is best-effort with eager fallback
        elapsed = time.perf_counter() - start
        return None, elapsed, f"torch.compile failed: {type(exc).__name__}: {exc}"


def compile_region_for_torch_device(
    region: RegionSource,
    candidate: KernelCandidate,
    *,
    backend_id: str,
    torch_device: str,
) -> CompiledRegion:
    """Realize a region subgraph on ``torch_device``.

    When ``candidate.attributes['use_torch_compile']`` is true (default), wrap
    with ``torch.compile``. Failure keeps the eager FX callable as explicit
    fallback — still the real graph region, never a metadata stub.
    """
    module = region.module
    if module is None or not callable(module):
        raise BackendError(
            f"Region {region.region_id} has no executable module; region partitioning must supply a torch.nn.Module"
        )
    if torch.device(torch_device).type != "cpu" and hasattr(module, "to"):
        module = module.to(torch_device)

    use_compile = bool(candidate.attributes.get("use_torch_compile", False))
    compile_backend = str(candidate.attributes.get("torch_compile_backend", "inductor"))
    machine_fp = str(candidate.attributes.get("machine_fingerprint", ""))
    eager = _eager_runner(module)
    attrs: dict[str, Any] = {
        "impl": "torch_fx_subgraph",
        "aten_ops": list(region.aten_ops),
        "input_names": list(region.input_names),
        "output_names": list(region.output_names),
        "compile_time_s": 0.0,
        "fallback": False,
        "fallback_reason": None,
        "cache_key": None,
    }

    if use_compile:
        cache_key = region_compile_fingerprint(
            region,
            torch_device=torch_device,
            backend=compile_backend,
            dtype=candidate.dtype,
            machine_fingerprint=machine_fp,
        )
        attrs["cache_key"] = cache_key
        examples = region.example_inputs
        compiled, compile_s, reason = _try_torch_compile(
            module,
            region_id=region.region_id,
            torch_device=torch_device,
            compile_backend=compile_backend,
            example_inputs=examples,
            cache_key=cache_key,
        )
        attrs["compile_time_s"] = compile_s
        schedule_managed = bool(candidate.attributes.get("schedule_managed_placement", True))
        if compiled is not None and reason is None and examples:
            # Keep Inductor only when it is not slower than eager FX on the
            # specialization examples (same honesty pattern as concurrency).
            with torch.inference_mode():
                placed = tuple(
                    t.to(torch_device)
                    if isinstance(t, torch.Tensor) and torch.device(torch_device).type != "cpu"
                    else t
                    for t in examples
                )
                for _ in range(2):
                    eager(*placed)
                    compiled(*placed)
                t0 = time.perf_counter()
                for _ in range(3):
                    eager(*placed)
                eager_s = (time.perf_counter() - t0) / 3.0
                t0 = time.perf_counter()
                for _ in range(3):
                    compiled(*placed)
                compiled_s = (time.perf_counter() - t0) / 3.0
            attrs["eager_latency_s"] = eager_s
            attrs["compiled_latency_s"] = compiled_s
            if compiled_s > eager_s * 1.05:
                reason = (
                    f"torch.compile slower than eager FX on examples "
                    f"({compiled_s * 1e3:.3f} ms > {eager_s * 1e3:.3f} ms)"
                )
                compiled = None
        if compiled is not None and reason is None:
            attrs["impl"] = f"torch_compile_{compile_backend}"
            executable: Any = _CompiledRegionCallable(
                region_id=region.region_id,
                torch_device=torch_device,
                eager=eager,
                compiled=compiled,
                impl=attrs["impl"],
                compile_time_s=compile_s,
                fallback_reason=None,
                schedule_managed_placement=schedule_managed,
            )
        else:
            attrs["impl"] = "torch_fx_subgraph"
            attrs["fallback"] = True
            attrs["fallback_reason"] = reason or "torch.compile declined"
            executable = _CompiledRegionCallable(
                region_id=region.region_id,
                torch_device=torch_device,
                eager=eager,
                compiled=None,
                impl=attrs["impl"],
                compile_time_s=compile_s,
                fallback_reason=attrs["fallback_reason"],
                schedule_managed_placement=schedule_managed,
            )
    else:
        schedule_managed = bool(candidate.attributes.get("schedule_managed_placement", True))
        executable = _RegionCallable(
            module, torch_device, region.region_id, schedule_managed_placement=schedule_managed
        )

    return CompiledRegion(
        region_id=region.region_id,
        device=candidate.device,
        backend_id=backend_id,
        executable=executable,
        dtype=candidate.dtype,
        torch_device=torch_device,
        attributes=attrs,
    )


def unwrap_region_callable(executable: Any) -> Any:
    """Return the bare forward when a CPU ``_RegionCallable`` needs no device move.

    Skipping the wrapper removes a Python call and a boolean check on every region
    invocation. Accelerators keep the wrapper so inputs still migrate.
    ``_CompiledRegionCallable`` is kept intact so Inductor/fallback stays active.
    """
    if isinstance(executable, _CompiledRegionCallable):
        return executable
    if getattr(executable, "_needs_move", None) is False and getattr(executable, "_run", None) is not None:
        return executable._run
    return executable


def execute_region_on_torch_device(executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]:
    """Execute a compiled region and normalize its outputs to a tuple."""
    result = executable.executable(*inputs)
    return coerce_region_result(result)


def coerce_region_result(result: Any) -> tuple[Any, ...]:
    """Normalize a region return value to a tuple of outputs."""
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


def clear_compile_cache() -> None:
    """Drop process-local ``torch.compile`` cache (tests / fingerprint change)."""
    _COMPILE_CACHE.clear()
