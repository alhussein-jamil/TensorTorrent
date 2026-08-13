"""CPU vs accelerator bakeoff helpers for compile-time plan selection.

Keeps measured fused-CPU / streamed-GPU / GPU-prefix-CPU-overflow comparisons
out of the main compile entry so auto mode and forced-GPU paths stay auditable.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from tensortorrent.compile.artifacts import PortableArtifact, SpecializedArtifact
from tensortorrent.compile.regions import RegionBinding, RegionProgram
from tensortorrent.compile.specialize import specialize_for_machine
from tensortorrent.config import CompileConfig
from tensortorrent.ir.resource_graph import ResourceGraph

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tensortorrent.runtime.schedule import ExecutableSchedule

# Prefer CPU on near-ties: avoids PCIe jitter and transfer-dominated noise.
CPU_BASELINE_HYSTERESIS = 1.02


@dataclass(frozen=True)
class BakeoffTiming:
    """Wall time plus whether it came from a real executor measure."""

    seconds: float
    measured: bool

    def __float__(self) -> float:
        return float(self.seconds)


def prefer_cpu_baseline(*, cpu_s: float, streamed_s: float) -> bool:
    """Prefer a valid CPU result unless streaming wins outside the noise margin."""
    if not math.isfinite(cpu_s):
        return False
    if not math.isfinite(streamed_s):
        return True
    return cpu_s <= streamed_s * CPU_BASELINE_HYSTERESIS


def select_beyond_vram_winner(
    *,
    cpu_s: float,
    streamed_s: float,
    overflow_s: float = float("inf"),
) -> str:
    """Pick fused CPU, streamed GPU, or static GPU-prefix + CPU-overflow.

    Accelerator candidates compete on raw time; equal overflow/streamed times
    prefer overflow (static map, no overflow H2D). CPU still wins on hysteresis
    against the best accelerator.
    """
    accel: list[tuple[float, int, str]] = []
    if math.isfinite(streamed_s):
        accel.append((streamed_s, 1, "streamed"))
    if math.isfinite(overflow_s):
        accel.append((overflow_s, 0, "gpu_prefix_overflow"))
    if not accel:
        return "cpu" if math.isfinite(cpu_s) else "streamed"
    _best_s, _tie, best_name = min(accel)
    if prefer_cpu_baseline(cpu_s=cpu_s, streamed_s=_best_s):
        return "cpu"
    return best_name


def _optimistic_partial_h2d_s(
    portable: PortableArtifact,
    config: CompileConfig,
    machine: ResourceGraph | None,
) -> float | None:
    """Lower-bound streamed latency from non-resident parameter bytes only."""
    try:
        from tensortorrent.compile.eager_cpu import (
            PARTIAL_H2D_SERIAL_OVERHEAD,
            estimate_parameter_stream_latency_s,
            estimate_partial_resident_stream_bytes,
        )
        from tensortorrent.compile.fit import live_hoist_budget_bytes

        program = getattr(portable, "program", None)
        if program is None:
            return None
        sizes: dict[str, int] = {}
        for name, spec in (getattr(program, "values", None) or {}).items():
            kind = str(getattr(spec, "kind", "") or "")
            if kind not in {"parameter", "buffer", "state"}:
                continue
            nbytes = int(getattr(spec, "nbytes", 0) or 0)
            if nbytes > 0:
                sizes[str(name)] = nbytes
        if not sizes:
            try:
                total = int(program.total_state_bytes())
            except Exception:  # noqa: BLE001
                total = 0
            if total <= 0:
                return None
            sizes = {"__all__": total}
        # CUDA is already warm during bakeoff; skip sync so timing stays clean.
        # Probe every visible device — bindings are not always available here.
        device_indices: set[int] | None = None
        if torch.cuda.is_available():
            try:
                device_indices = set(range(int(torch.cuda.device_count())))
            except (RuntimeError, ValueError):
                device_indices = None
        budget = live_hoist_budget_bytes(
            config,
            machine,
            device_indices=device_indices,
            synchronize=False,
        )
        if budget is None:
            return None
        _resident, streamed, _selected = estimate_partial_resident_stream_bytes(sizes, budget_bytes=int(budget))
        return float(estimate_parameter_stream_latency_s(streamed)) * PARTIAL_H2D_SERIAL_OVERHEAD
    except Exception as exc:  # noqa: BLE001
        logger.debug("partial H2D estimate unavailable: %s", exc)
        return None


def bindings_use_accelerator(bindings: dict[str, RegionBinding]) -> bool:
    """True when any region binding targets a discrete/accelerator backend."""
    accel = {"cuda", "rocm", "xpu"}
    for binding in bindings.values():
        backend = str(getattr(binding, "backend_id", "") or "").lower()
        device = str(getattr(binding, "device", "") or "").lower()
        if backend in accel or device.startswith("cuda_") or device.startswith("rocm_") or device.startswith("xpu_"):
            return True
    return False


def synchronize_bound_accelerators(bindings: dict[str, RegionBinding]) -> None:
    """Wait for async accelerator kernels before reading wall-clock time."""
    cuda_devices: set[str] = set()
    xpu_devices: set[str] = set()
    for binding in bindings.values():
        backend_id = str(binding.backend_id)
        torch_device = str(getattr(binding.compiled, "torch_device", ""))
        if backend_id in {"cuda", "rocm"} and torch_device.startswith("cuda"):
            cuda_devices.add(torch_device)
        elif backend_id == "xpu" and torch_device.startswith("xpu"):
            xpu_devices.add(torch_device)
    if cuda_devices and torch.cuda.is_available():
        for device in sorted(cuda_devices):
            torch.cuda.synchronize(device)
    xpu = getattr(torch, "xpu", None)
    if xpu_devices and xpu is not None and xpu.is_available():
        for device in sorted(xpu_devices):
            xpu.synchronize(device)


def time_executor(
    program: RegionProgram,
    bindings: dict[str, RegionBinding],
    flat_inputs: list[Any],
    *,
    workers: int,
    intraop_threads: int,
    iters: int = 7,
    enable_dataflow_direct_path: bool = False,
    prefetch_distance: int = 0,
    schedule: ExecutableSchedule | None = None,
    machine: ResourceGraph | None = None,
    config: CompileConfig | None = None,
) -> float:
    """Median synchronized wall time for one executor candidate."""
    from tensortorrent.runtime.graph_executor import GraphExecutor
    from tensortorrent.runtime.tensor_store import ResidentParameterStore

    store = ResidentParameterStore(program.state_tensors())
    executor: GraphExecutor | None = None
    try:
        executor = GraphExecutor(
            program,
            bindings,
            parameter_store=store,
            max_workers=workers,
            prefetch_distance=prefetch_distance,
            intraop_threads=intraop_threads,
            schedule=schedule,
            machine=machine,
            config=config,
            enable_dataflow_direct_path=enable_dataflow_direct_path,
        )
        if enable_dataflow_direct_path:
            from tensortorrent.runtime.direct_path import DataflowDirectPlan

            if not isinstance(executor.direct_plan, DataflowDirectPlan):
                return float("inf")
        run_inputs = list(flat_inputs)
        for _ in range(2):
            executor.run(run_inputs)
        synchronize_bound_accelerators(bindings)
        samples: list[float] = []
        for _ in range(max(1, iters)):
            synchronize_bound_accelerators(bindings)
            start = time.perf_counter()
            executor.run(run_inputs)
            synchronize_bound_accelerators(bindings)
            samples.append(time.perf_counter() - start)
        samples.sort()
        middle = len(samples) // 2
        if len(samples) % 2:
            return samples[middle]
        return (samples[middle - 1] + samples[middle]) / 2
    finally:
        if executor is not None:
            executor.close()
        store.close()


def _schedule_needs_streaming_store(schedule: Any) -> bool:
    """True when the schedule emits parameter Prefetch/Load (needs streaming store)."""
    from tensortorrent.ir.graph import OpCode

    for inst in getattr(schedule, "instructions", ()) or ():
        opcode = getattr(inst, "opcode", None)
        if opcode not in {OpCode.LOAD, OpCode.PREFETCH}:
            continue
        kind = str((getattr(inst, "attributes", None) or {}).get("kind") or "")
        if kind.startswith("parameter"):
            return True
    return False


def safe_time_executor(
    program: RegionProgram,
    specialized: SpecializedArtifact,
    flat_inputs: list[Any],
    *,
    config: CompileConfig,
    machine: ResourceGraph,
    workers: int,
    intraop_threads: int,
) -> BakeoffTiming:
    """Time an executor candidate; failures become ``inf`` (loses bakeoff).

    Streaming Prefetch/Load schedules cannot run under ResidentParameterStore;
    those candidates return planner-predicted latency with ``measured=False``.
    """
    schedule = getattr(specialized, "schedule", None)
    # Bakeoff timing uses ResidentParameterStore; streaming Prefetch/Load schedules
    # cannot run there. Fall back to the plan's predicted latency instead of
    # treating the streamed candidate as infinitely slow (which forced CPU).
    if schedule is not None and _schedule_needs_streaming_store(schedule):
        predicted = float(getattr(specialized.plan, "predicted_latency_s", 0.0) or 0.0)
        if predicted > 0.0:
            return BakeoffTiming(seconds=predicted, measured=False)
        logger.warning("streaming candidate has no predicted latency; bakeoff skips measure")
        return BakeoffTiming(seconds=float("inf"), measured=False)
    try:
        seconds = time_executor(
            program,
            specialized.bindings,
            flat_inputs,
            workers=workers,
            intraop_threads=intraop_threads,
            prefetch_distance=specialized.plan.prefetch_distance,
            schedule=schedule,
            machine=machine,
            config=config,
        )
        return BakeoffTiming(seconds=float(seconds), measured=True)
    except Exception as exc:  # noqa: BLE001 - candidate failure selects the surviving baseline
        logger.warning("baseline candidate timing failed: %s", exc)
        return BakeoffTiming(seconds=float("inf"), measured=False)


def time_eager_module(
    module: Any,
    flat_inputs: list[Any],
    *,
    iters: int = 7,
    in_spec: Any | None = None,
) -> float:
    """Median wall time for the original ``nn.Module`` under eval semantics.

    When ``in_spec`` is set (export / RegionProgram pytree), flat leaves are
    unflattened to ``(*args, **kwargs)`` so kwargs and nested inputs match the
    module signature. Without a spec, leaves are passed positionally (legacy).
    """
    from torch.utils import _pytree as pytree

    from tensortorrent.compile.eager_cpu import time_eager_call

    if in_spec is not None:
        args, kwargs = pytree.tree_unflatten(list(flat_inputs), in_spec)
        if not isinstance(kwargs, dict):
            kwargs = {}
        if not isinstance(args, tuple):
            args = tuple(args)
    else:
        args = tuple(flat_inputs)
        kwargs = {}
    return time_eager_call(module, args, kwargs, iters=iters)


def time_cpu_fused_candidate(
    eager_module: Any | None,
    cpu_program: RegionProgram,
    cpu_specialized: SpecializedArtifact,
    flat_inputs: list[Any],
    *,
    config: CompileConfig,
    machine: ResourceGraph,
) -> float:
    """Prefer timing the original module when available (matches steady-state path)."""
    if eager_module is not None:
        try:
            return time_eager_module(
                eager_module,
                flat_inputs,
                in_spec=getattr(cpu_program, "in_spec", None),
            )
        except Exception as exc:  # noqa: BLE001 - fall back to export GraphModule path
            logger.warning("eager fused CPU timing failed: %s", exc)
    return float(
        safe_time_executor(
            cpu_program,
            cpu_specialized,
            flat_inputs,
            config=config,
            machine=machine,
            workers=1,
            intraop_threads=0,
        )
    )


def choose_fusion_candidate(
    *,
    fused_s: float,
    concurrent_schedule_s: float,
    concurrent_dataflow_s: float,
    hetero_plan: bool,
) -> tuple[bool, bool, float, float]:
    """Pick fused vs concurrent and whether dataflow should stay enabled.

    Returns ``(prefer_fused, dataflow_enabled, concurrent_s, fuse_margin)``.
    """
    dataflow_enabled = concurrent_dataflow_s * 1.02 <= concurrent_schedule_s
    concurrent_s = concurrent_dataflow_s if dataflow_enabled else concurrent_schedule_s
    fuse_margin = 1.10 if hetero_plan else 1.02
    prefer_fused = fused_s * fuse_margin <= concurrent_s
    if (
        prefer_fused
        and hetero_plan
        and dataflow_enabled
        and concurrent_schedule_s >= 1.5 * concurrent_dataflow_s
        and concurrent_s * 1.02 <= fused_s * 1.10
    ):
        prefer_fused = False
    return prefer_fused, dataflow_enabled, concurrent_s, fuse_margin


def select_sequential_beyond_vram_plan(
    exported: Any,
    *,
    program: RegionProgram,
    portable: PortableArtifact,
    config: CompileConfig,
    name: str,
    machine: ResourceGraph,
    measurements: Any | None,
    flat_inputs: list[Any],
    eager_module: Any | None = None,
    lower_to_portable: Any,
) -> tuple[
    RegionProgram,
    PortableArtifact,
    SpecializedArtifact,
    bool,
    str,
    str,
    dict[str, Any],
]:
    """Bake off accelerator streaming, GPU-prefix CPU-overflow, and fused CPU.

    Sequential beyond-VRAM models have no branch overlap to exploit. A streamed
    accelerator plan can lose when parameter traffic dominates compute. Auto mode
    measures a forced-accelerator streamed plan, a static GPU-prefix + CPU-suffix
    plan (overflow weights stay on host), and a fused CPU candidate.

    The streamed candidate must not fall back to multi-region CPU: that path is
    strictly worse than fused CPU and previously fooled the bakeoff when the
    planner declined GPU placement.

    ``lower_to_portable`` is injected from the compile entry to avoid a circular
    import with the lowering pipeline.
    """
    from tensortorrent.runtime.provisioning import intraop_threads, worker_count

    # Measure the original module before any accelerator specialize touches CUDA /
    # host allocators. On multi-GiB models that pollution permanently slows the
    # fused-CPU steady-state path by 2–3×.
    early_cpu_s = float("inf")
    if config.allow_cpu and eager_module is not None:
        try:
            early_cpu_s = time_eager_module(
                eager_module,
                flat_inputs,
                in_spec=getattr(program, "in_spec", None),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("early eager fused CPU timing failed: %s", exc)

    streamed_config = config
    if config.allow_gpu or config.allow_integrated_gpu:
        streamed_config = replace(config, allow_cpu=False)

    cpu_config = replace(
        config,
        allow_cpu=True,
        allow_gpu=False,
        allow_integrated_gpu=False,
        allow_concurrent_regions=False,
        max_concurrent_regions=1,
    )

    def _build_fused_cpu_candidate() -> tuple[RegionProgram, PortableArtifact, SpecializedArtifact]:
        """Force-single lower + specialize — only call when CPU wins the bakeoff."""
        built_program, built_portable = lower_to_portable(
            exported,
            name=name,
            config=cpu_config,
            artifact_dir=None,
            force_single_region=True,
            machine=machine,
        )
        built_specialized = specialize_for_machine(
            built_portable,
            config=cpu_config,
            output_dir=None,
            example_inputs=flat_inputs,
            machine=machine,
            measurements=None,
        )
        return built_program, built_portable, built_specialized

    cpu_program: RegionProgram | None = None
    cpu_portable: PortableArtifact | None = None
    cpu_specialized: SpecializedArtifact | None = None

    # Prefer early eager timing so we do not rematerialize multi-GiB packs just
    # to score the CPU candidate. Fall back to export GraphModule only when no
    # eager module was provided.
    if math.isfinite(early_cpu_s):
        cpu_s = early_cpu_s
    elif eager_module is None:
        cpu_program, cpu_portable, cpu_specialized = _build_fused_cpu_candidate()
        cpu_s = time_cpu_fused_candidate(
            None,
            cpu_program,
            cpu_specialized,
            flat_inputs,
            config=cpu_config,
            machine=machine,
        )
    else:
        cpu_s = float("inf")

    def _ensure_cpu_candidate() -> tuple[RegionProgram, PortableArtifact, SpecializedArtifact]:
        nonlocal cpu_program, cpu_portable, cpu_specialized
        if cpu_specialized is None:
            cpu_program, cpu_portable, cpu_specialized = _build_fused_cpu_candidate()
        assert cpu_program is not None and cpu_portable is not None and cpu_specialized is not None
        return cpu_program, cpu_portable, cpu_specialized

    # Cheap streamed prediction first (no region compile). If fused CPU already
    # wins by a clear margin *and* a partial-resident H2D lower bound agrees,
    # skip full GPU specialize+measure so the eager module stays fast.
    # Beyond-VRAM probes often over-predict (ignore partial residency) — do not
    # skip on prediction alone when the optimistic non-resident H2D bound still
    # looks competitive with CPU.
    streamed_probe = specialize_for_machine(
        portable,
        config=streamed_config,
        output_dir=None,
        example_inputs=flat_inputs,
        machine=machine,
        measurements=measurements,
        compile_regions=False,
    )
    predicted_streamed = float(getattr(streamed_probe.plan, "predicted_latency_s", 0.0) or 0.0)
    probe_devices = [str(d) for d in (getattr(streamed_probe.plan, "devices_used", None) or ())]
    probe_on_accel = bindings_use_accelerator(streamed_probe.bindings) or any(
        d.startswith("cuda_") or d.startswith("rocm_") or d.startswith("xpu_") for d in probe_devices
    )
    partial_h2d_s = _optimistic_partial_h2d_s(portable, streamed_config, machine)
    skip_streamed_specialize = (
        config.allow_cpu
        and math.isfinite(cpu_s)
        and predicted_streamed > 0.0
        and cpu_s * 1.5 < predicted_streamed
        and probe_on_accel
        # Only skip when partial-resident H2D also clearly loses to CPU.
        and (partial_h2d_s is None or cpu_s * 1.5 < partial_h2d_s)
    )
    if skip_streamed_specialize:
        comparison = {
            "measured": False,
            "cpu_fused_s": cpu_s,
            "streamed_s": predicted_streamed,
            "streamed_predicted_s": predicted_streamed,
            "partial_h2d_predicted_s": partial_h2d_s,
            "overflow_meta": {"reason": "skipped_with_streamed_specialize"},
            "skipped_streamed_measure": True,
            "skipped_streamed_specialize": True,
            "cpu_hysteresis": CPU_BASELINE_HYSTERESIS,
            "selected": "cpu",
            "cpu_path": "eager_module" if eager_module is not None else "export_graph",
        }
        cpu_program, cpu_portable, cpu_specialized = _ensure_cpu_candidate()
        cpu_specialized.validation["baseline_guard"] = comparison
        cpu_specialized.plan.notes.append(
            f"baseline_compare: cpu_fused={cpu_s * 1e3:.3f}ms "
            f"streamed={predicted_streamed * 1e3:.3f}ms selected=cpu "
            "(streamed=predicted; skipped GPU specialize)"
        )
        return (
            cpu_program,
            cpu_portable,
            cpu_specialized,
            False,
            "fused CPU baseline: predicted accelerator streaming slower than measured CPU",
            "fused_cpu_baseline: predicted accelerator streaming slower than measured CPU",
            {
                "fused_after_sequential_decision": True,
                "fused_cpu_baseline": True,
                "baseline_guard_selected": "cpu",
                "baseline_guard": comparison,
            },
        )

    streamed = specialize_for_machine(
        portable,
        config=streamed_config,
        output_dir=None,
        example_inputs=flat_inputs,
        machine=machine,
        measurements=measurements,
    )
    if not config.allow_cpu:
        return (
            program,
            portable,
            streamed,
            True,
            "sequential graph kept partitioned for accelerator streaming",
            "kept_multi_region: CPU disabled; retained accelerator streaming",
            {"kept_multi_region_for_accelerator_budget": True},
        )

    if not bindings_use_accelerator(streamed.bindings):
        comparison = {
            "measured": False,
            "cpu_fused_s": cpu_s if math.isfinite(cpu_s) else None,
            "streamed_s": None,
            "cpu_hysteresis": CPU_BASELINE_HYSTERESIS,
            "selected": "cpu",
            "reason": "streamed_candidate_not_on_accelerator",
        }
        cpu_program, cpu_portable, cpu_specialized = _ensure_cpu_candidate()
        cpu_specialized.validation["baseline_guard"] = comparison
        cpu_specialized.plan.notes.append("fused_cpu_baseline: streamed candidate lacked accelerator placement")
        return (
            cpu_program,
            cpu_portable,
            cpu_specialized,
            False,
            "fused CPU baseline: streamed candidate lacked accelerator placement",
            "fused_cpu_baseline: streamed candidate lacked accelerator placement",
            {
                "fused_after_sequential_decision": True,
                "fused_cpu_baseline": True,
                "baseline_guard_selected": "cpu",
                "baseline_guard": comparison,
            },
        )

    streamed_timing = safe_time_executor(
        program,
        streamed,
        flat_inputs,
        config=streamed_config,
        machine=machine,
        workers=worker_count(streamed, streamed_config),
        intraop_threads=intraop_threads(streamed, streamed_config),
    )
    streamed_s = float(streamed_timing.seconds)
    predicted_streamed = float(getattr(streamed.plan, "predicted_latency_s", 0.0) or 0.0)
    overflow_art, overflow_timing, overflow_meta = _maybe_gpu_prefix_overflow(
        portable,
        program,
        streamed,
        config=config,
        machine=machine,
        measurements=measurements,
        flat_inputs=flat_inputs,
    )
    overflow_s = float(overflow_timing.seconds)
    selected = select_beyond_vram_winner(cpu_s=cpu_s, streamed_s=streamed_s, overflow_s=overflow_s)
    if selected == "gpu_prefix_overflow" and overflow_art is None:
        selected = select_beyond_vram_winner(cpu_s=cpu_s, streamed_s=streamed_s, overflow_s=float("inf"))
    # Accelerator specialize permanently slows host BLAS in-process. If early
    # CPU timing beat accelerators but post-specialize eager no longer does, keep
    # the measured accelerator plan (genuinely faster available TT path).
    post_cpu_s: float | None = None
    if selected == "cpu" and eager_module is not None:
        try:
            post_cpu_s = time_eager_module(
                eager_module,
                flat_inputs,
                in_spec=getattr(program, "in_spec", None),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("post-specialize eager CPU timing failed: %s", exc)
            post_cpu_s = float("inf")
        selected = select_beyond_vram_winner(
            cpu_s=post_cpu_s,
            streamed_s=streamed_s,
            overflow_s=overflow_s if overflow_art is not None else float("inf"),
        )
        if selected == "gpu_prefix_overflow" and overflow_art is None:
            selected = select_beyond_vram_winner(cpu_s=post_cpu_s, streamed_s=streamed_s, overflow_s=float("inf"))
    accel_provenance = "measured" if streamed_timing.measured else "predicted"
    overflow_provenance = "measured" if overflow_timing.measured else "predicted"
    comparison = {
        "measured": bool(streamed_timing.measured or overflow_timing.measured),
        "streamed_timing_provenance": accel_provenance,
        "overflow_timing_provenance": overflow_provenance if overflow_art is not None else None,
        "cpu_fused_s": cpu_s,
        "cpu_fused_s_post_specialize": post_cpu_s,
        "streamed_s": streamed_s,
        "streamed_predicted_s": predicted_streamed,
        "overflow_s": overflow_s if math.isfinite(overflow_s) else None,
        "overflow_meta": overflow_meta,
        "skipped_streamed_measure": not streamed_timing.measured,
        "cpu_hysteresis": CPU_BASELINE_HYSTERESIS,
        "selected": selected,
        "cpu_path": "eager_module" if eager_module is not None else "export_graph",
    }
    if (
        post_cpu_s is not None
        and selected != "cpu"
        and math.isfinite(cpu_s)
        and prefer_cpu_baseline(cpu_s=cpu_s, streamed_s=min(streamed_s, overflow_s))
    ):
        comparison["cpu_path_rejected"] = "host_blas_poisoned_after_accelerator_specialize"
    overflow_ms = f"{overflow_s * 1e3:.3f}ms" if math.isfinite(overflow_s) else "n/a"
    compare_note = (
        f"baseline_compare: cpu_fused={cpu_s * 1e3:.3f}ms "
        f"streamed={streamed_s * 1e3:.3f}ms ({accel_provenance}) "
        f"overflow={overflow_ms} ({overflow_provenance}) selected={selected}"
    )

    if selected == "cpu":
        cpu_program, cpu_portable, cpu_specialized = _ensure_cpu_candidate()
        cpu_specialized.validation["baseline_guard"] = comparison
        cpu_specialized.plan.notes.append(compare_note)
        cpu_note = (
            f"{accel_provenance} accelerator streaming vs measured fused CPU; selected=cpu"
            if streamed_timing.measured
            else "predicted accelerator streaming vs measured fused CPU; selected=cpu"
        )
        return (
            cpu_program,
            cpu_portable,
            cpu_specialized,
            False,
            cpu_note,
            f"fused_cpu_baseline: measured CPU path beat {accel_provenance} streaming",
            {
                "fused_after_sequential_decision": True,
                "fused_cpu_baseline": True,
                "baseline_guard_selected": "cpu",
                "baseline_guard": comparison,
            },
        )

    if selected == "gpu_prefix_overflow" and overflow_art is not None:
        overflow_art.validation["baseline_guard"] = comparison
        overflow_art.plan.notes.append(compare_note)
        return (
            program,
            portable,
            overflow_art,
            True,
            f"{overflow_provenance} GPU-prefix + CPU-overflow beat fused CPU and streamed GPU",
            f"kept_multi_region: {overflow_provenance} GPU-prefix CPU-overflow beat fused CPU and streamed GPU",
            {
                "kept_multi_region_for_accelerator_budget": True,
                "gpu_prefix_cpu_overflow": True,
                "baseline_guard_selected": "gpu_prefix_overflow",
                "baseline_guard": comparison,
            },
        )

    streamed.validation["baseline_guard"] = comparison
    streamed.plan.notes.append(compare_note)
    accel_note = (
        f"{accel_provenance} accelerator streaming beat fused CPU baseline"
        if streamed_timing.measured
        else "predicted accelerator streaming beat fused CPU baseline"
    )
    return (
        program,
        portable,
        streamed,
        True,
        accel_note,
        f"kept_multi_region: {accel_provenance} accelerator streaming beat fused CPU baseline",
        {
            "kept_multi_region_for_accelerator_budget": True,
            "baseline_guard_selected": "streamed",
            "baseline_guard": comparison,
        },
    )


def _maybe_gpu_prefix_overflow(
    portable: PortableArtifact,
    program: RegionProgram,
    template: SpecializedArtifact,
    *,
    config: CompileConfig,
    machine: ResourceGraph,
    measurements: Any | None,
    flat_inputs: list[Any],
) -> tuple[SpecializedArtifact | None, BakeoffTiming, dict[str, Any]]:
    """Specialize + time GPU-prefix / CPU-suffix when the cut is interior."""
    from tensortorrent.compile.fit import accelerator_hoist_budget_bytes
    from tensortorrent.compile.overflow import (
        gpu_prefix_count,
        gpu_prefix_overflow_plan,
        host_cpu_placement_target,
    )
    from tensortorrent.runtime.provisioning import intraop_threads, worker_count

    meta: dict[str, Any] = {}
    n_regions = len(program.regions)
    budget = accelerator_hoist_budget_bytes(config, machine)
    if budget is None or n_regions < 2:
        meta["reason"] = "no_budget_or_single_region"
        return None, BakeoffTiming(seconds=float("inf"), measured=False), meta
    n_gpu = gpu_prefix_count(program, int(budget))
    meta["gpu_prefix_regions"] = n_gpu
    meta["region_count"] = n_regions
    if n_gpu < 1 or n_gpu >= n_regions:
        meta["reason"] = "cut_not_interior"
        return None, BakeoffTiming(seconds=float("inf"), measured=False), meta
    cpu_target = host_cpu_placement_target(machine)
    if cpu_target is None:
        meta["reason"] = "no_cpu_device"
        return None, BakeoffTiming(seconds=float("inf"), measured=False), meta
    cpu_device, cpu_backend = cpu_target
    overflow_config = replace(
        config,
        allow_cpu=True,
        allow_gpu=True,
        prefetch_distance=0,
        adaptive_prefetch=False,
    )
    try:
        forced = gpu_prefix_overflow_plan(
            template.plan,
            program,
            n_gpu=n_gpu,
            cpu_device=cpu_device,
            cpu_backend=cpu_backend,
        )
        specialized = specialize_for_machine(
            portable,
            config=overflow_config,
            output_dir=None,
            example_inputs=flat_inputs,
            machine=machine,
            measurements=measurements,
            forced_plan=forced,
        )
    except Exception as exc:  # noqa: BLE001 - candidate failure drops this arm
        logger.warning("GPU-prefix CPU-overflow specialize failed: %s", exc)
        meta["reason"] = f"specialize_failed:{type(exc).__name__}"
        return None, BakeoffTiming(seconds=float("inf"), measured=False), meta
    timing = safe_time_executor(
        program,
        specialized,
        flat_inputs,
        config=overflow_config,
        machine=machine,
        workers=worker_count(specialized, overflow_config),
        intraop_threads=intraop_threads(specialized, overflow_config),
    )
    meta["reason"] = "measured" if timing.measured else "predicted"
    return specialized, timing, meta


def bakeoff_fused_cpu_against_accelerator(
    exported: Any,
    *,
    program: RegionProgram,
    portable: PortableArtifact,
    specialized: SpecializedArtifact,
    config: CompileConfig,
    name: str,
    machine: ResourceGraph,
    flat_inputs: list[Any],
    note: str,
    validation_extra: dict[str, Any],
    eager_module: Any | None = None,
    lower_to_portable: Any,
) -> tuple[RegionProgram, PortableArtifact, SpecializedArtifact, str, dict[str, Any]]:
    """When auto may use CPU, keep fused CPU if it beats the accelerator plan.

    In-VRAM single-region GPU plans can still lose to host eager-style fused CPU
    when PCIe input traffic or dispatch dwarfs compute. Mirror the beyond-VRAM
    baseline guard so auto mode does not force a slower accelerator path.
    """
    if not config.allow_cpu or not config.allow_gpu:
        return program, portable, specialized, note, validation_extra
    if not bindings_use_accelerator(specialized.bindings):
        return program, portable, specialized, note, validation_extra

    from tensortorrent.runtime.provisioning import intraop_threads, worker_count

    cpu_config = replace(
        config,
        allow_cpu=True,
        allow_gpu=False,
        allow_integrated_gpu=False,
        allow_concurrent_regions=False,
        max_concurrent_regions=1,
    )

    # Time CPU first so accelerator measure cannot pollute the host baseline.
    if eager_module is not None:
        cpu_s = time_eager_module(
            eager_module,
            flat_inputs,
            in_spec=getattr(program, "in_spec", None),
        )
        cpu_program: RegionProgram | None = None
        cpu_portable: PortableArtifact | None = None
        cpu_specialized: SpecializedArtifact | None = None
    else:
        cpu_program, cpu_portable = lower_to_portable(
            exported,
            name=name,
            config=cpu_config,
            artifact_dir=None,
            force_single_region=True,
            machine=machine,
        )
        cpu_specialized = specialize_for_machine(
            cpu_portable,
            config=cpu_config,
            output_dir=None,
            example_inputs=flat_inputs,
            machine=machine,
            measurements=None,
        )
        cpu_s = time_cpu_fused_candidate(
            None,
            cpu_program,
            cpu_specialized,
            flat_inputs,
            config=cpu_config,
            machine=machine,
        )

    accel_timing = safe_time_executor(
        program,
        specialized,
        flat_inputs,
        config=config,
        machine=machine,
        workers=worker_count(specialized, config),
        intraop_threads=intraop_threads(specialized, config),
    )
    accel_s = float(accel_timing.seconds)
    use_cpu = prefer_cpu_baseline(cpu_s=cpu_s, streamed_s=accel_s)
    accel_provenance = "measured" if accel_timing.measured else "predicted"
    comparison = {
        "measured": bool(accel_timing.measured),
        "accelerator_timing_provenance": accel_provenance,
        "cpu_fused_s": cpu_s,
        "accelerator_fused_s": accel_s,
        "skipped_accelerator_measure": not accel_timing.measured,
        "cpu_hysteresis": CPU_BASELINE_HYSTERESIS,
        "selected": "cpu" if use_cpu else "accelerator",
        "cpu_path": "eager_module" if eager_module is not None else "export_graph",
    }
    compare_note = (
        f"baseline_compare: cpu_fused={cpu_s * 1e3:.3f}ms "
        f"accelerator_fused={accel_s * 1e3:.3f}ms ({accel_provenance}) selected={comparison['selected']}"
    )
    if use_cpu:
        if cpu_specialized is None:
            cpu_program, cpu_portable = lower_to_portable(
                exported,
                name=name,
                config=cpu_config,
                artifact_dir=None,
                force_single_region=True,
                machine=machine,
            )
            cpu_specialized = specialize_for_machine(
                cpu_portable,
                config=cpu_config,
                output_dir=None,
                example_inputs=flat_inputs,
                machine=machine,
                measurements=None,
            )
        assert cpu_program is not None and cpu_portable is not None and cpu_specialized is not None
        cpu_specialized.validation["baseline_guard"] = comparison
        cpu_specialized.plan.notes.append(compare_note)
        extra = {
            **validation_extra,
            "fused_cpu_baseline": True,
            "baseline_guard_selected": "cpu",
            "baseline_guard": comparison,
        }
        return (
            cpu_program,
            cpu_portable,
            cpu_specialized,
            f"measured fused CPU baseline beat {accel_provenance} fused accelerator plan",
            extra,
        )

    specialized.validation["baseline_guard"] = comparison
    specialized.plan.notes.append(compare_note)
    extra = {
        **validation_extra,
        "baseline_guard_selected": "accelerator",
        "baseline_guard": comparison,
    }
    return program, portable, specialized, note, extra
