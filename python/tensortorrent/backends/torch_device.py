"""Shared region realization for every backend that PyTorch can target.

CPU, CUDA, ROCm, MPS and XPU all execute the *same* partitioned subgraphs; only
the torch device string and the transfer semantics differ. Keeping this logic in
one place is what prevents per-vendor copies of the compiler.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from collections.abc import Sequence
from typing import Any

import torch

from tensortorrent.backends.base import (
    BenchmarkResult,
    CompiledRegion,
    KernelCandidate,
    RegionSource,
)
from tensortorrent.closed import closed_str
from tensortorrent.errors import BackendError

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

    __slots__ = ("module", "torch_device", "region_id", "_target", "_repair_device", "_run")

    def __init__(
        self, module: Any, torch_device: str, region_id: str, *, schedule_managed_placement: bool = True
    ) -> None:
        self.module = module
        self.torch_device = torch_device
        self.region_id = region_id
        self._target = torch.device(torch_device)
        # Accelerator regions: move any arg that is not already on the target
        # device. Covers non-schedule-managed placement and repairs export-time
        # CPU leftovers / missed H2D under schedule-managed placement.
        self._repair_device = self._target.type != "cpu"
        self._run = module.forward if _has_no_hooks(module) else module

    def __call__(self, *inputs: Any) -> Any:
        if self._repair_device:
            # Blocking moves: a non-blocking H2D here would race the compute that
            # immediately consumes the tensor (schedule Transfer path syncs; this
            # repair path does not).
            inputs = tuple(
                t.to(self._target, non_blocking=False)
                if isinstance(t, torch.Tensor) and t.device != self._target
                else t
                for t in inputs
            )
        return self._run(*inputs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_RegionCallable(region={self.region_id!r}, device={self.torch_device!r})"


class _AotRegionCallable:
    """Runs an AOTInductor-compiled region.

    The packaged runner already takes the region's positional arguments, so
    this only normalises the return shape: AOTInductor hands back a list even
    for a single output, while the schedule expects the same shape the eager
    region produced.
    """

    __slots__ = ("region_id", "runner")

    def __init__(self, *, region_id: str, runner: Any) -> None:
        self.region_id = region_id
        self.runner = runner

    def __call__(self, *args: Any) -> Any:
        out = self.runner(*args)
        if isinstance(out, (list, tuple)) and len(out) == 1:
            return out[0]
        return out


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
        "_target",
        "_repair_device",
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
        self._target = torch.device(torch_device)
        self._repair_device = self._target.type != "cpu"
        self._use_compiled = compiled is not None and fallback_reason is None

    def _place(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not self._repair_device:
            return inputs
        return tuple(
            t.to(self._target, non_blocking=False) if isinstance(t, torch.Tensor) and t.device != self._target else t
            for t in inputs
        )

    def __call__(self, *inputs: Any) -> Any:
        placed = self._place(inputs)
        if self._use_compiled:
            # Accepted Inductor executables must not silently fall back on
            # arbitrary runtime errors — only compile/warmup may choose eager.
            compiled = self.compiled
            assert compiled is not None
            return compiled(*placed)
        return self.eager(*placed)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"_CompiledRegionCallable(region={self.region_id!r}, impl={self.impl!r}, fallback={self.fallback_reason!r})"
        )


def _fx_graph_hash(module: Any) -> str:
    """Stable hash of an FX GraphModule's printable code, else module repr."""
    try:
        graph = getattr(module, "graph", None)
        if graph is not None:
            return hashlib.sha256(str(graph).encode("utf-8")).hexdigest()[:24]
        code = getattr(module, "code", None)
        if isinstance(code, str) and code:
            return hashlib.sha256(code.encode("utf-8")).hexdigest()[:24]
    except Exception:  # noqa: BLE001
        pass
    return hashlib.sha256(repr(type(module)).encode("utf-8")).hexdigest()[:24]


def _example_signature(
    example_inputs: Sequence[Any] | None,
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...], tuple[tuple[int, ...], ...], tuple[str, ...]]:
    shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    strides: list[tuple[int, ...]] = []
    layouts: list[str] = []
    for value in example_inputs or ():
        if isinstance(value, torch.Tensor):
            shapes.append(tuple(int(d) for d in value.shape))
            dtypes.append(str(value.dtype).replace("torch.", ""))
            strides.append(tuple(int(s) for s in value.stride()))
            layouts.append("contiguous" if value.is_contiguous() else "strided")
        else:
            shapes.append(())
            dtypes.append(type(value).__name__)
            strides.append(())
            layouts.append("scalar")
    return tuple(shapes), tuple(dtypes), tuple(strides), tuple(layouts)


def region_compile_fingerprint(
    region: RegionSource,
    *,
    torch_device: str,
    backend: str,
    dtype: str,
    machine_fingerprint: str = "",
    input_shapes: Sequence[tuple[int, ...]] | None = None,
    input_dtypes: Sequence[str] | None = None,
    input_strides: Sequence[tuple[int, ...]] | None = None,
    input_layouts: Sequence[str] | None = None,
    compiler_config: str = "",
    inductor_config: str = "",
    fx_graph_hash: str = "",
) -> str:
    """Hardware + software key for compiled-region caching.

    Includes FX/graph hash, shapes/dtypes/strides/layouts, CPU/thread config, and
    PyTorch / Inductor / compiler versions so cache hits stay sound.
    """
    import platform
    import sys

    shapes = ",".join("x".join(str(d) for d in s) for s in (input_shapes or ()))
    dtypes = ",".join(input_dtypes or ())
    strides = ",".join("x".join(str(d) for d in s) for s in (input_strides or ()))
    layouts = ",".join(input_layouts or ())
    cpu_isa = ""
    try:
        import torch.backends.cpu as cpu_backends

        cpu_isa = ",".join(
            name
            for name in ("avx2", "avx512", "vnni", "neon")
            if bool(getattr(cpu_backends, f"has_{name}", lambda: False)())
        )
    except Exception:  # noqa: BLE001
        cpu_isa = platform.machine()
    graph_hash = fx_graph_hash or _fx_graph_hash(region.module)
    payload = "|".join(
        [
            region.region_id,
            torch_device,
            backend,
            dtype,
            machine_fingerprint,
            f"fx={graph_hash}",
            ",".join(region.aten_ops),
            ",".join(region.input_names),
            ",".join(region.output_names),
            shapes,
            dtypes,
            strides,
            layouts,
            compiler_config,
            inductor_config,
            str(torch.__version__),
            platform.machine(),
            platform.processor() or "",
            cpu_isa,
            f"threads={torch.get_num_threads()}",
            f"py={sys.version_info.major}.{sys.version_info.minor}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _eager_runner(module: Any) -> Any:
    return module.forward if _has_no_hooks(module) else module


def _try_aot_inductor(
    module: Any,
    *,
    torch_device: str,
    example_inputs: Sequence[Any] | None,
) -> tuple[Any | None, float, str | None]:
    """Attempt AOTInductor. Return ``(runner, seconds, reason_if_unavailable)``.

    AOTInductor is a third realistic way to run a region, and on measured CPU
    workloads it beat both eager FX and Inductor on two of five. It is offered
    as a candidate rather than assumed better: the selector below decides.

    It needs a system CUDA toolkit (``nvcc``) for GPU regions; the PyPI torch
    wheels ship headers and libraries but no compiler, so failure there is an
    environment fact and is reported as such rather than raised.
    """
    if not example_inputs:
        return None, 0.0, "no example inputs"
    start = time.perf_counter()
    try:
        placed = tuple(
            t.to(torch_device) if isinstance(t, torch.Tensor) and torch.device(torch_device).type != "cpu" else t
            for t in example_inputs
        )
        exported = torch.export.export(module, tuple(placed))
        path = torch._inductor.aoti_compile_and_package(exported)
        runner = torch._inductor.aoti_load_package(path)
        with torch.inference_mode():
            runner(*placed)
        return runner, time.perf_counter() - start, None
    except Exception as exc:  # noqa: BLE001 - a missing candidate is not an error
        detail = f"{type(exc).__name__}: {exc}"
        if "CUDA_HOME" in detail or "nvcc" in detail:
            detail = "needs a system CUDA toolkit (nvcc); the PyPI torch wheels ship no compiler"
        return None, time.perf_counter() - start, detail[:160]


def select_fastest_candidate(
    candidates: Sequence[tuple[str, Any]],
    placed_inputs: Sequence[Any],
    *,
    rounds: int = 9,
    warmup: int = 3,
) -> tuple[str, dict[str, float]]:
    """Measure every candidate on the same inputs and name the fastest.

    Candidates are interleaved so thermal drift and scheduler noise hit each
    equally, and medians are compared rather than means: this decision turns on
    a few percent, and a mean of three samples was demonstrably picking the
    wrong one.

    Returns the winning name and every measurement, so the choice is
    explainable in region attributes rather than opaque.
    """
    names = [n for n, fn in candidates if fn is not None]
    fns = {n: fn for n, fn in candidates if fn is not None}
    if not names:
        return "", {}
    samples: dict[str, list[float]] = {n: [] for n in names}
    with torch.inference_mode():
        for _ in range(warmup):
            for n in names:
                fns[n](*placed_inputs)
        for _ in range(rounds):
            for n in names:
                t0 = time.perf_counter()
                fns[n](*placed_inputs)
                samples[n].append(time.perf_counter() - t0)
    medians = {n: statistics.median(v) for n, v in samples.items()}
    # Ties go to the earliest candidate, which is ordered simplest-first: an
    # equal-speed compiled artefact is not worth its compile time or memory.
    winner = min(names, key=lambda n: (medians[n], names.index(n)))
    return winner, medians


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
        placed = None
        if example_inputs:
            placed = tuple(
                t.to(torch_device) if isinstance(t, torch.Tensor) and torch.device(torch_device).type != "cpu" else t
                for t in example_inputs
            )
        fullgraph = True
        try:
            compiled_mod = torch.compile(module, backend=compile_backend, fullgraph=True)
            runner = _eager_runner(compiled_mod)
            if placed is not None:
                with torch.inference_mode():
                    runner(*placed)
        except Exception:  # noqa: BLE001 — graph break: accept partial compile
            fullgraph = False
            compiled_mod = torch.compile(module, backend=compile_backend, fullgraph=False)
            runner = _eager_runner(compiled_mod)
            if placed is not None:
                with torch.inference_mode():
                    runner(*placed)
        elapsed = time.perf_counter() - start
        runner._tt_fullgraph = fullgraph
        _COMPILE_CACHE[cache_key] = runner
        return runner, elapsed, None
    except Exception as exc:  # noqa: BLE001 - compile is best-effort with eager fallback
        elapsed = time.perf_counter() - start
        return None, elapsed, f"torch.compile failed: {type(exc).__name__}: {exc}"


def assert_region_module_schedule_safe(
    module: Any,
    *,
    region_id: str,
    schedule_managed_placement: bool,
    declared_state: Sequence[str] = (),
) -> None:
    """Reject hidden region state that can bypass schedule-managed residency.

    Under schedule-managed placement the region module must be stateless: every
    parameter/buffer either is absent or was lifted into explicit region inputs.
    """
    if module is None:
        return
    declared = {str(x) for x in declared_state}
    unexpected_parameters = [
        name for name, _p in getattr(module, "named_parameters", lambda: [])() if name not in declared
    ]
    unexpected_buffers = [name for name, _b in getattr(module, "named_buffers", lambda: [])() if name not in declared]
    if unexpected_parameters or unexpected_buffers:
        raise BackendError(
            f"Region {region_id} retains undeclared state"
            f"{' under schedule-managed placement' if schedule_managed_placement else ''}: "
            f"unexpected_parameters={unexpected_parameters} unexpected_buffers={unexpected_buffers}"
        )


def _is_cpu_device(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.device):
        return value.type == "cpu"
    if isinstance(value, str):
        return value.split(":", 1)[0] == "cpu"
    return False


def _rewrite_hardcoded_cpu_devices(module: torch.nn.Module, torch_device: str) -> int:
    """Replace export-time ``device=cpu`` literals so regions can run on ``torch_device``.

    ``torch.export`` on CPU hardcodes ``arange(..., device=cpu)`` and
    ``to(device=cpu)`` / ``to(dtype_layout=..., device=cpu)``. Under
    schedule-managed placement those ops fight CUDA/ROCm/XPU residency and
    raise ``Expected: cpu, Got: cuda:0`` (or silently create CPU tensors that
    then mismatch). Rewrite only explicit CPU device literals; leave dtype /
    layout alone.
    """
    target = torch.device(torch_device)
    if target.type == "cpu":
        return 0
    rewritten = 0
    if isinstance(module, torch.fx.GraphModule):
        for node in module.graph.nodes:
            if node.op != "call_function":
                continue
            new_kwargs = dict(node.kwargs)
            changed = False
            if "device" in new_kwargs and _is_cpu_device(new_kwargs["device"]):
                new_kwargs["device"] = target
                changed = True
            if changed:
                node.kwargs = new_kwargs
                rewritten += 1
            # aten.to.device(cpu, ...) positional device
            if node.args and _is_cpu_device(node.args[0]):
                name = str(getattr(node.target, "__name__", node.target))
                if name.startswith("to.") or name in {"to", "aten.to.device"}:
                    node.args = (target, *node.args[1:])
                    rewritten += 1
        if rewritten:
            module.recompile()
    for child in module.children():
        rewritten += _rewrite_hardcoded_cpu_devices(child, torch_device)
    return rewritten


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
    schedule_managed = bool(candidate.attributes.get("schedule_managed_placement", True))
    declared_state = tuple(str(x) for x in (region.attributes or {}).get("declared_state", ()) or ())
    assert_region_module_schedule_safe(
        module,
        region_id=region.region_id,
        schedule_managed_placement=schedule_managed,
        declared_state=declared_state,
    )
    # Never ``module.to(device)`` when the schedule owns weight/activation movement.
    if (not schedule_managed) and torch.device(torch_device).type != "cpu" and hasattr(module, "to"):
        module = module.to(torch_device)
    elif schedule_managed and torch.device(torch_device).type != "cpu":
        # Copy before rewriting export-time CPU device literals so CPU candidates
        # keep the original graph when both backends are considered.
        import copy

        module = copy.deepcopy(module)
        _rewrite_hardcoded_cpu_devices(module, torch_device)

    use_compile = bool(candidate.attributes.get("use_torch_compile", False))
    compile_backend = str(candidate.attributes.get("torch_compile_backend", "inductor"))
    machine_fp = str(candidate.attributes.get("machine_fingerprint", ""))
    # Competitive/full pay for AOTInductor + interleaved bake-off. Coarse (default)
    # keeps a single torch.compile attempt so specialize wall time stays usable.
    profile_level = closed_str(candidate.attributes.get("profile_level", "coarse") or "coarse")
    competitive_select = profile_level in {"competitive", "full"}
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
        "schedule_managed_placement": schedule_managed,
        "profile_level": profile_level,
    }

    if use_compile:
        examples = region.example_inputs
        shapes, dtypes, strides, layouts = _example_signature(examples)
        fx_hash = _fx_graph_hash(module)
        cache_key = region_compile_fingerprint(
            region,
            torch_device=torch_device,
            backend=compile_backend,
            dtype=candidate.dtype,
            machine_fingerprint=machine_fp,
            input_shapes=shapes,
            input_dtypes=dtypes,
            input_strides=strides,
            input_layouts=layouts,
            compiler_config=str(candidate.attributes.get("compiler_config", "")),
            inductor_config=str(candidate.attributes.get("inductor_config", "")),
            fx_graph_hash=fx_hash,
        )
        attrs["cache_key"] = cache_key
        attrs["fx_graph_hash"] = fx_hash
        compiled, compile_s, reason = _try_torch_compile(
            module,
            region_id=region.region_id,
            torch_device=torch_device,
            compile_backend=compile_backend,
            example_inputs=examples,
            cache_key=cache_key,
        )
        attrs["compile_time_s"] = compile_s
        if compiled is not None and reason is None and examples and competitive_select:
            placed = tuple(
                t.to(torch_device) if isinstance(t, torch.Tensor) and torch.device(torch_device).type != "cpu" else t
                for t in examples
            )
            # Three ways to run this region. Offer all of them and keep whichever
            # is actually fastest here, rather than assuming a compiler wins.
            # Ordered simplest-first so ties fall back to eager FX.
            aot, aot_s, aot_reason = _try_aot_inductor(module, torch_device=torch_device, example_inputs=examples)
            attrs["aot_compile_time_s"] = aot_s
            if aot_reason:
                attrs["aot_unavailable"] = aot_reason

            winner, medians = select_fastest_candidate(
                (("eager_fx", eager), (f"torch_compile_{compile_backend}", compiled), ("aot_inductor", aot)),
                placed,
            )
            attrs["candidate_latencies_s"] = dict(medians)
            attrs["selected_candidate"] = winner
            attrs["eager_latency_s"] = medians.get("eager_fx", 0.0)
            attrs["compiled_latency_s"] = medians.get(f"torch_compile_{compile_backend}", 0.0)

            if winner == "aot_inductor" and aot is not None:
                # AOTInductor wins: it is already a callable taking the same
                # positional arguments, so it replaces the region executable.
                attrs["impl"] = "aot_inductor"
                attrs["compile_time_s"] = attrs.get("compile_time_s", 0.0) + aot_s
                return CompiledRegion(
                    region_id=region.region_id,
                    device=candidate.device,
                    backend_id=backend_id,
                    executable=_AotRegionCallable(region_id=region.region_id, runner=aot),
                    dtype=candidate.dtype,
                    torch_device=torch_device,
                    attributes=attrs,
                )
            if winner == "eager_fx":
                reason = (
                    f"eager FX fastest on examples "
                    f"({medians.get('eager_fx', 0.0) * 1e3:.3f} ms vs "
                    f"{medians.get(f'torch_compile_{compile_backend}', 0.0) * 1e3:.3f} ms compiled)"
                )
                compiled = None
        elif compiled is not None and reason is None:
            attrs["selected_candidate"] = f"torch_compile_{compile_backend}"
        if compiled is not None and reason is None:
            fullgraph = bool(getattr(compiled, "_tt_fullgraph", True))
            attrs["impl"] = f"torch_compile_{compile_backend}"
            attrs["fullgraph"] = fullgraph
            if not fullgraph:
                attrs["compilation_mode"] = "partial"
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
