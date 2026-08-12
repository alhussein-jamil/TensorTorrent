"""Export-free fused-CPU compile path for beyond-VRAM auto selection.

``torch.export`` on multi-GiB models followed by CUDA discovery permanently
slows host BLAS in-process (often 2–3×). When measured host compute is
*confidently* faster than an optimistic partial-resident GPU H2D lower bound,
skip export/CUDA and wrap the original module. Near-ties and GPU-favored
estimates fall through to the normal compile + bakeoff path.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils import _pytree as pytree

from tensortorrent.closed import OutputRefKind, ParameterStoreKind, ValueKind
from tensortorrent.compile.artifacts import PortableArtifact, SpecializedArtifact
from tensortorrent.compile.bakeoff import prefer_cpu_baseline
from tensortorrent.compile.fit import select_persistent_parameter_ids
from tensortorrent.compile.regions import Region, RegionBinding, RegionProgram, ValueSpec
from tensortorrent.config import CompileConfig, Objective
from tensortorrent.ir.graph import HeterogeneousGraph
from tensortorrent.ir.resource_graph import ResourceDecision
from tensortorrent.planner.maximal import ExecutionPlan, Placement

logger = logging.getLogger(__name__)

# Effective host→device bandwidth for cheap streaming estimates (PCIe 3.0 x16-ish).
_DEFAULT_H2D_BYTES_PER_S = 12.0e9
# Non-overlapped setup/kernel tax on sequential partial-resident H2D lower bounds.
PARTIAL_H2D_SERIAL_OVERHEAD = 1.25
# Export-free only when measured CPU beats the optimistic GPU bound by this factor.
# Near-ties fall through so bakeoff can measure both candidates.
_EXPORT_FREE_CPU_CONFIDENCE = 1.15


def module_parameter_bytes(module: Any) -> int:
    """Sum parameter + buffer nbytes on an ``nn.Module``."""
    total = 0
    if not isinstance(module, torch.nn.Module):
        return 0
    for tensor in (*module.parameters(), *module.buffers()):
        total += int(tensor.numel()) * int(tensor.element_size())
    return total


def module_parameter_nbytes_by_name(module: Any) -> dict[str, int]:
    """Named parameter/buffer sizes for partial-residency estimates."""
    sizes: dict[str, int] = {}
    if not isinstance(module, torch.nn.Module):
        return sizes
    for name, param in module.named_parameters():
        sizes[str(name)] = int(param.numel()) * int(param.element_size())
    for name, buf in module.named_buffers():
        sizes[str(name)] = int(buf.numel()) * int(buf.element_size())
    return sizes


def peek_accelerator_vram_bytes() -> int | None:
    """Read CUDA total memory without running discovery dtype probes."""
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:  # noqa: BLE001 - optional fast path only
        return None


def estimate_parameter_stream_latency_s(
    param_bytes: int,
    *,
    host_to_device_bps: float = _DEFAULT_H2D_BYTES_PER_S,
) -> float:
    """Lower-bound PCIe-dominated streamed forward time (H2D of ``param_bytes``)."""
    if param_bytes <= 0 or host_to_device_bps <= 0:
        return float("inf")
    return float(param_bytes) / float(host_to_device_bps)


def estimate_partial_resident_stream_bytes(
    parameter_nbytes: dict[str, int],
    *,
    budget_bytes: int,
    transfer_groups: list[tuple[str, ...]] | None = None,
) -> tuple[int, int, set[str]]:
    """Optimistic residency: largest-first fit with stream headroom.

    Returns ``(resident_bytes, streamed_bytes, selected_ids)``. Streamed bytes
    are the H2D traffic that remains after partial persistent hoist.

    When ``transfer_groups`` is None, each tensor is its own group. Callers that
    know coalesced region packs should pass them so headroom matches runtime.
    """
    if budget_bytes <= 0 or not parameter_nbytes:
        total = sum(max(0, int(v)) for v in parameter_nbytes.values())
        return 0, total, set()
    groups = transfer_groups
    if groups is None:
        groups = [(name,) for name in parameter_nbytes]
    selected = select_persistent_parameter_ids(
        parameter_nbytes,
        budget_bytes=int(budget_bytes),
        transfer_groups=groups,
    )
    resident = sum(max(0, int(parameter_nbytes[n])) for n in selected)
    streamed = sum(max(0, int(v)) for n, v in parameter_nbytes.items() if n not in selected)
    return resident, streamed, selected


@contextmanager
def temporary_eval(module: Any) -> Iterator[None]:
    """Run ``module`` under eval without permanently changing caller modes."""
    if not isinstance(module, torch.nn.Module):
        yield
        return
    if not module.training:
        yield
        return
    training_states = tuple((child, bool(child.training)) for child in module.modules())
    module.eval()
    try:
        yield
    finally:
        for child, was_training in training_states:
            child.training = was_training


def time_eager_call(
    module: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    iters: int = 3,
) -> float:
    """Median wall time for ``module(*args, **kwargs)`` under eval semantics.

    Keep the sample count small: this runs on the export-free critical path and
    multi-GiB repeats thermally throttle the package before steady-state timing.
    """
    import time

    kw = dict(kwargs or {})
    with temporary_eval(module), torch.inference_mode():
        for _ in range(2):
            module(*args, **kw)
        samples: list[float] = []
        for _ in range(max(1, iters)):
            start = time.perf_counter()
            module(*args, **kw)
            samples.append(time.perf_counter() - start)
    samples.sort()
    middle = len(samples) // 2
    if len(samples) % 2:
        return samples[middle]
    return (samples[middle - 1] + samples[middle]) / 2


def should_prefer_eager_cpu_without_export(
    module: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    config: CompileConfig,
    *,
    param_bytes: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """True only when CPU is confidently faster than partial-resident GPU H2D.

    GPU estimate charges H2D for **non-resident** parameter bytes only (optimistic
    partial persistent residency). If the comparison is close or favors GPU,
    return False so normal compile + bakeoff can measure both candidates.
    """
    meta: dict[str, Any] = {"checked": True, "selected": "streamed_or_full_compile"}
    if not config.allow_cpu:
        meta["reason"] = "cpu_disabled"
        return False, meta
    if config.allow_training:
        meta["reason"] = "training"
        return False, meta
    if not (config.allow_gpu or config.allow_integrated_gpu):
        meta["reason"] = "accelerator_disabled"
        return False, meta

    state = int(param_bytes if param_bytes is not None else module_parameter_bytes(module))
    meta["param_bytes"] = state
    vram = config.vram_budget_bytes
    if vram is None:
        vram = peek_accelerator_vram_bytes()
    if vram is None:
        meta["reason"] = "no_vram_capacity"
        return False, meta
    vram = int(vram)
    meta["vram_bytes"] = vram
    # Only the beyond-VRAM sequential case needs this escape hatch.
    if state <= vram:
        meta["reason"] = "fits_vram"
        return False, meta

    # Optimistic hoist budget from configured capacity. Do **not** call
    # ``torch.cuda.mem_get_info`` here: initializing the CUDA context permanently
    # slows multi-GiB host GEMM and defeats the export-free path.
    from tensortorrent.compile.fit import optimistic_hoist_budget_without_cuda

    budget = optimistic_hoist_budget_without_cuda(config, vram)
    meta["hoist_budget_bytes"] = int(budget)

    sizes = module_parameter_nbytes_by_name(module)
    if not sizes and state > 0:
        # Nameless estimate: treat the whole state as one streamed blob when
        # named sizes are unavailable (should not happen for nn.Module).
        sizes = {"__all__": state}
    # Approximate coalesced region H2D groups (max_region_nodes). Per-tensor
    # groups over-pack residents vs real schedules and under-estimate stream.
    names = list(sizes)
    chunk = max(1, int(getattr(config, "max_region_nodes", 1) or 1))
    groups = [tuple(names[i : i + chunk]) for i in range(0, len(names), chunk)]
    resident_b, streamed_b, selected = estimate_partial_resident_stream_bytes(
        sizes,
        budget_bytes=int(budget),
        transfer_groups=groups,
    )
    meta["resident_param_bytes"] = int(resident_b)
    meta["streamed_param_bytes"] = int(streamed_b)
    meta["persistent_parameter_count"] = len(selected)

    full_stream_s = estimate_parameter_stream_latency_s(state)
    partial_stream_s = estimate_parameter_stream_latency_s(streamed_b)
    meta["streamed_predicted_s"] = partial_stream_s
    meta["full_streamed_predicted_s"] = full_stream_s

    cpu_s: float | None = None
    try:
        cpu_s = time_eager_call(module, args, kwargs)
        meta["cpu_fused_s"] = cpu_s
    except Exception as exc:  # noqa: BLE001
        meta["eager_timing_error"] = f"{type(exc).__name__}: {exc}"
        meta["reason"] = "eager_timing_failed"
        return False, meta

    # Optimistic GPU bound: non-resident H2D with a small sequential-stream
    # overhead (kernel/setup that does not fully overlap PCIe). Pure H2D alone
    # under-estimates measured partial-resident GPU and blocks export-free on
    # near-fit CPU wins (beyond@1.05).
    gpu_s = float(partial_stream_s) * PARTIAL_H2D_SERIAL_OVERHEAD
    meta["gpu_partial_h2d_predicted_s"] = gpu_s
    meta["gpu_partial_h2d_raw_s"] = float(partial_stream_s)

    # Confident CPU win only — otherwise measure both via bakeoff.
    confident = (
        prefer_cpu_baseline(cpu_s=cpu_s, streamed_s=gpu_s)
        and math.isfinite(cpu_s)
        and math.isfinite(gpu_s)
        and cpu_s * _EXPORT_FREE_CPU_CONFIDENCE < gpu_s
    )
    if not confident:
        meta["reason"] = "uncertain_or_gpu_favored"
        meta["selected"] = "streamed_or_full_compile"
        return False, meta

    meta["selected"] = "cpu"
    meta["reason"] = "cpu_beats_partial_resident_h2d"
    return True, meta


def build_eager_fused_program(module: Any, example_inputs: Any, *, name: str) -> RegionProgram:
    """Minimal :class:`RegionProgram` for ``module(*args, **kwargs)`` DirectPlan execution."""
    # Local import: ``frontend.export`` may load this module from ``compile()``.
    from tensortorrent.frontend.export import _split_example_inputs

    args, kwargs = _split_example_inputs(example_inputs)
    flat_in, in_spec = pytree.tree_flatten((args, kwargs))
    with temporary_eval(module), torch.inference_mode():
        out = module(*args, **kwargs)
    flat_out, out_spec = pytree.tree_flatten(out)

    values: dict[str, ValueSpec] = {}
    user_inputs: list[str] = []
    for i, value in enumerate(flat_in):
        iname = f"arg_{i}"
        user_inputs.append(iname)
        if torch.is_tensor(value):
            values[iname] = ValueSpec(
                name=iname,
                shape=tuple(int(s) for s in value.shape),
                dtype=str(value.dtype).removeprefix("torch."),
                nbytes=int(value.numel()) * int(value.element_size()),
                kind=ValueKind.INPUT,
            )
        else:
            values[iname] = ValueSpec(name=iname, shape=(), dtype="unknown", nbytes=0, kind=ValueKind.INPUT)

    output_refs: list[tuple[OutputRefKind, Any]] = []
    for i, value in enumerate(flat_out):
        oname = f"out_{i}"
        output_refs.append((OutputRefKind.VALUE, oname))
        if torch.is_tensor(value):
            values[oname] = ValueSpec(
                name=oname,
                shape=tuple(int(s) for s in value.shape),
                dtype=str(value.dtype).removeprefix("torch."),
                nbytes=int(value.numel()) * int(value.element_size()),
                kind=ValueKind.ACTIVATION,
            )
        else:
            values[oname] = ValueSpec(name=oname, shape=(), dtype="unknown", nbytes=0, kind=ValueKind.ACTIVATION)

    region = Region(
        region_id="eager_fused",
        submodule="",
        inputs=tuple(user_inputs),
        outputs=tuple(ref for _, ref in output_refs),
        multi_output=len(output_refs) > 1,
        aten_ops=(),
        node_count=1,
        depends_on=(),
        state_inputs=(),
        output_bytes=sum(values[ref].nbytes for _, ref in output_refs if ref in values),
    )
    return RegionProgram(
        graph_name=name,
        root=module,
        regions=(region,),
        user_inputs=tuple(user_inputs),
        state_bindings={},
        values=values,
        output_refs=tuple(output_refs),
        in_spec=in_spec,
        out_spec=out_spec,
        metadata={"eager_fused_export_free": True},
    )


def build_eager_fused_compiled_module(
    module: Any,
    example_inputs: Any,
    *,
    config: CompileConfig,
    name: str,
    guard: dict[str, Any],
) -> Any:
    """Build a :class:`CompiledModule` that calls ``module`` via DirectPlan.

    Avoids :class:`GraphExecutor` / native schedule construction: those imports
    and CUDA touches permanently slow multi-GiB host GEMM in-process.
    """
    from tensortorrent.frontend.export import _split_example_inputs
    from tensortorrent.runtime.direct_path import make_eager_fused_direct_plan
    from tensortorrent.runtime.module import CompiledModule

    program = build_eager_fused_program(module, example_inputs, name=name)
    region = program.regions[0]
    binding = RegionBinding(region=region, compiled=module, backend_id="cpu", device="cpu_numa_0")
    cpu_s = float(guard.get("cpu_fused_s") or 0.0)
    stream_s = float(guard.get("streamed_predicted_s") or 0.0)
    plan = ExecutionPlan(
        graph_name=name,
        fingerprint="eager_fused_export_free",
        objective=getattr(config, "objective", None) or Objective.LATENCY,
        placements=[
            Placement(
                region_id=region.region_id,
                device="cpu_numa_0",
                backend_id="cpu",
                dtype="float32",
                kernel_id="eager_module",
                estimated_latency_s=cpu_s,
                measured=cpu_s > 0.0,
                state_bytes=int(guard.get("param_bytes") or 0),
            )
        ],
        decisions=[
            ResourceDecision(resource="cpu_numa_0", selected=True, reason="export-free fused CPU baseline"),
        ],
        devices_used=("cpu_numa_0",),
        communication_backend="none",
        predicted_latency_s=cpu_s,
        predicted_transfer_bytes=0,
        predicted_transfer_latency_s=0.0,
        strategy="fused_cpu_baseline_export_free",
        notes=[
            "eager_fused_module: export-free original nn.Module (skipped torch.export + CUDA discovery)",
            (
                f"baseline_compare: cpu_fused={cpu_s * 1e3:.3f}ms "
                f"partial_h2d={stream_s * 1e3:.3f}ms selected=cpu "
                "(partial_h2d=predicted non-resident; skipped export)"
            ),
        ],
    )
    portable = PortableArtifact(
        name=name,
        ir=HeterogeneousGraph(name=name, parameters=(), outputs=(), repeated_blocks=(), metadata={}),
        program=program,
        metadata={"eager_fused_export_free": True},
    )
    specialized = SpecializedArtifact(
        fingerprint=plan.fingerprint,
        plan=plan,
        validation={
            "fused_cpu_baseline": True,
            "eager_fused_module": True,
            "eager_fused_export_free": True,
            "baseline_guard": {
                "measured": bool(cpu_s > 0.0),
                "cpu_fused_s": cpu_s if cpu_s > 0.0 else None,
                "streamed_s": stream_s,
                "streamed_predicted_s": stream_s,
                "skipped_streamed_measure": True,
                "skipped_streamed_specialize": True,
                "skipped_export": True,
                "selected": "cpu",
                "cpu_path": "eager_module_export_free",
                **{k: v for k, v in guard.items() if k not in {"cpu_fused_s", "streamed_predicted_s", "selected"}},
            },
            "baseline_guard_selected": "cpu",
        },
        bindings={"eager_fused": binding},
    )
    param_bytes = int(program.total_state_bytes()) or int(guard.get("param_bytes") or 0)
    direct = make_eager_fused_direct_plan(program, module, param_bytes=param_bytes)
    executor = _EagerDirectExecutor(program, direct, bindings=specialized.bindings)
    args, kwargs = _split_example_inputs(example_inputs)
    # Mirror capture_module: temporary eval during build must not stick.
    training_states: tuple[tuple[Any, bool], ...] = ()
    if isinstance(module, torch.nn.Module):
        training_states = tuple((child, bool(child.training)) for child in module.modules())
    compiled = CompiledModule(
        portable=portable,
        specialized=specialized,
        config=config,
        program=program,
        executor=executor,  # type: ignore[arg-type]
        machine=None,
        example_flat=list(pytree.tree_flatten((args, kwargs))[0]),
    )
    for child, was_training in training_states:
        child.training = was_training
    return compiled


class _EagerDirectExecutor:
    """Minimal executor: DirectPlan only, no native schedule / CUDA discovery."""

    def __init__(self, program: RegionProgram, direct_plan: Any, *, bindings: dict[str, RegionBinding]) -> None:
        self.program = program
        self.bindings = bindings
        self._direct_plan = direct_plan
        self.intraop_threads = 0
        self.max_workers = 1
        self.parameter_store = _EmptyParameterStore(resident_bytes=int(program.total_state_bytes()))
        self.schedule = None
        self._closed = False
        self._last_schedule_report = None

    @property
    def direct_plan(self) -> Any:
        return self._direct_plan

    @property
    def closed(self) -> bool:
        return self._closed

    def run(
        self,
        flat_inputs: list[Any],
        *,
        cancel_token: Any | None = None,
        enable_grad: bool = False,
    ) -> tuple[list[Any], Any]:
        del cancel_token, enable_grad
        if self._closed:
            raise RuntimeError("executor closed")
        import time

        from tensortorrent.runtime.fork_regions import RegionEvent
        from tensortorrent.runtime.graph_executor import ExecutionReport

        plan = self._direct_plan
        start = time.perf_counter()
        outputs = plan.call(*plan.build_args(flat_inputs))
        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)
        wall = time.perf_counter() - start
        # call returns flat leaves matching output_refs order when multi-output.
        if len(outputs) == len(self.program.output_refs):
            flat_outputs = list(outputs)
        else:
            by_name = dict(zip(plan.output_names, outputs, strict=False))
            flat_outputs = [by_name[ref[1]] for ref in self.program.output_refs]
        report = ExecutionReport(
            wall_time_s=wall,
            events=[
                RegionEvent(
                    region_id=plan.region_id,
                    device=plan.device,
                    backend_id="cpu",
                    start_s=start,
                    end_s=start + wall,
                    worker="direct",
                )
            ],
            max_concurrent_regions=1,
            parameter_store={"kind": "eager_fused", "execution_path": "direct"},
            instruction_ids=[f"compute::{plan.region_id}"],
        )
        return flat_outputs, report

    def close(self) -> None:
        self._closed = True

    def request_cancel(self) -> None:
        return None


class _EmptyParameterStore:
    """Placeholder store for export-free DirectPlan (weights live on the module)."""

    kind = ParameterStoreKind.EAGER_FUSED
    needs_prefetch = False

    def __init__(self, *, resident_bytes: int = 0) -> None:
        self._resident_bytes = max(0, int(resident_bytes))

    def acquire(self, name: str) -> Any:
        raise KeyError(name)

    def release(self, names: tuple[str, ...]) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value if isinstance(self.kind, ParameterStoreKind) else self.kind,
            "resident_bytes": self._resident_bytes,
            "tensor_count": 0,
            "needs_prefetch": False,
        }

    def close(self) -> None:
        return None
