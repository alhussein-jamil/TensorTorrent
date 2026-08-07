"""End-to-end compile entry from exported programs."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from tensortorrent.compile.artifacts import PortableArtifact, SpecializedArtifact, portable_compile_from_ir
from tensortorrent.compile.cache import _attach_storage_measurement
from tensortorrent.compile.concurrency import dependency_levels
from tensortorrent.compile.fit import (
    exceeds_accelerator_region_budget,
    exported_parameter_bytes,
    region_state_budget,
    should_force_single_region,
    streaming_region_budget,
)
from tensortorrent.compile.regions import RegionBinding, RegionProgram
from tensortorrent.compile.specialize import (
    specialize_for_machine,
)
from tensortorrent.config import CompileConfig
from tensortorrent.errors import MemoryCapacityError
from tensortorrent.hardware.discovery import discover_resource_graph
from tensortorrent.ir.resource_graph import ResourceGraph

logger = logging.getLogger(__name__)

# Compat aliases — tests and pipeline historically imported these from entry.
_exported_parameter_bytes = exported_parameter_bytes
_region_state_budget = region_state_budget
_streaming_region_budget = streaming_region_budget


def _should_force_single_region(
    config: CompileConfig,
    machine: ResourceGraph,
    exported: Any,
) -> bool:
    return should_force_single_region(
        config,
        machine,
        parameter_bytes=exported_parameter_bytes(exported),
    )


def _exceeds_accelerator_region_budget(
    config: CompileConfig,
    machine: ResourceGraph,
    exported: Any,
) -> bool:
    return exceeds_accelerator_region_budget(
        config,
        machine,
        parameter_bytes=exported_parameter_bytes(exported),
    )


if TYPE_CHECKING:
    from tensortorrent.runtime.module import CompiledModule


def compile_exported_program(
    exported: Any,
    *,
    config: CompileConfig | None = None,
    name: str = "model",
    artifact_dir: Path | None = None,
    pack_lookup_dirs: tuple[Path, ...] = (),
    machine: Any | None = None,
    measurements: Any | None = None,
) -> CompiledModule:
    """Compile an already-captured ``ExportedProgram`` into a runnable module.

    This is the single implementation behind both :func:`tensortorrent.compile`
    and artifact reloading, so both paths exercise identical code.
    """

    from tensortorrent.runtime.graph_executor import GraphExecutor
    from tensortorrent.runtime.module import CompiledModule
    from tensortorrent.runtime.provisioning import (
        build_parameter_store,
        intraop_threads,
        schedule_needs_host_pin,
        worker_count,
    )

    config = config or CompileConfig()
    _machine_for_fit = machine if machine is not None else discover_resource_graph()
    # Fuse to one region when concurrency is off: avoids per-region dispatch
    # when the planner will not schedule branches in parallel anyway.
    # Training keeps multi-region partitions so train and eval share one
    # multi-piece ExecutableSchedule (no fused single-region collapse).
    # Host RAM streaming needs multi-region shards. A VRAM budget alone must
    # not disable fusion for models that still fit on the GPU — but when the
    # full parameter set exceeds the per-region accelerator budget, forcing a
    # single region makes GPU placement infeasible and the planner falls back
    # to CPU-only. Keep partitions in that case so the GPU can stream.
    force_single = _should_force_single_region(config, _machine_for_fit, exported)
    program, portable = _lower_to_portable(
        exported,
        name=name,
        config=config,
        artifact_dir=artifact_dir,
        force_single_region=force_single,
        machine=_machine_for_fit,
    )
    _check_early_fit(program, _machine_for_fit, config)
    example_flat = _example_flat_inputs(exported, program)
    # Prefer a fused single-region fast path when it beats multi-region execution.
    # Auto-concurrency may win against a multi-region sequential schedule yet still
    # lose to one fused region that avoids per-region dispatch.
    # Skip fusion when training: schedule-native autograd needs the partitioned program.
    fusion_eligible = (
        len(program.regions) > 1
        and not config.allow_training
        and config.ram_budget_bytes is None
        and config.allow_concurrent_regions
        and config.max_concurrent_regions == 0
        and not force_single
        and example_flat is not None
    )
    if fusion_eligible:
        from tensortorrent.compile.concurrency import ConcurrencyDecision

        assert example_flat is not None  # guarded by fusion_eligible
        flat_inputs = example_flat

        # Sequential DAGs cannot overlap regions. Fuse immediately unless the
        # full model exceeds the accelerator region budget (needs partitions).
        levels = dependency_levels(program)
        specialized: SpecializedArtifact
        if len(levels.widest()) < 2:
            keep_partitions = _exceeds_accelerator_region_budget(config, _machine_for_fit, exported)
            specialize_config = config
            if keep_partitions:
                concurrency_reason = "sequential graph kept partitioned for accelerator streaming"
                note = (
                    "kept_multi_region: sequential graph exceeds accelerator region budget; "
                    "skipped fusion so GPU/accelerator placement stays feasible"
                )
                validation_extra = {"kept_multi_region_for_accelerator_budget": True}
            else:
                specialize_config = replace(config, allow_concurrent_regions=False, max_concurrent_regions=1)
                program, portable = _lower_to_portable(
                    exported,
                    name=name,
                    config=specialize_config,
                    artifact_dir=artifact_dir,
                    force_single_region=True,
                    machine=_machine_for_fit,
                )
                concurrency_reason = "graph has no independent regions to overlap"
                note = "fused_to_single_region: no independent regions; skipped multi-region specialize"
                validation_extra = {
                    "fused_after_sequential_decision": True,
                    "fusion_skipped_multi_region": True,
                }
            specialized = specialize_for_machine(
                portable,
                config=specialize_config,
                output_dir=(artifact_dir / "specialized") if artifact_dir else None,
                example_inputs=flat_inputs,
                machine=_machine_for_fit,
                measurements=measurements,
            )
            specialized.validation["concurrency"] = ConcurrencyDecision(
                enabled=False,
                workers=1,
                group=levels.widest(),
                reason=concurrency_reason,
            ).as_dict()
            specialized.validation.update(validation_extra)
            specialized.plan.notes.append(note)
            workers = 1 if not keep_partitions else worker_count(specialized, config)
        else:
            # Repartition wide inference graphs into whole dependency branches
            # before comparing them with full fusion. Small fixed node caps split
            # one tower into many callbacks and can hide a real CPU/GPU overlap
            # win behind dispatch overhead.
            graph_nodes = sum(region.node_count for region in program.regions)
            coarse_config = replace(config, max_region_nodes=max(config.max_region_nodes, graph_nodes))
            coarse_program, coarse_portable = _lower_to_portable(
                exported,
                name=name,
                config=coarse_config,
                artifact_dir=artifact_dir,
                force_single_region=False,
                machine=_machine_for_fit,
            )
            coarse_levels = dependency_levels(coarse_program)
            # Coarse regions reuse ordinal ids (region_0, …) but contain different
            # subgraphs — drop caller measurements so placement re-profiles.
            fusion_measurements = measurements
            if len(coarse_levels.widest()) >= 2:
                program, portable = coarse_program, coarse_portable
                fusion_measurements = None

            # Wide graphs: probe concurrency without compiling kernels. workers==1
            # still avoids a discarded multi-region kernel compile.
            probe = specialize_for_machine(
                portable,
                config=config,
                output_dir=None,
                example_inputs=flat_inputs,
                machine=_machine_for_fit,
                measurements=fusion_measurements,
                compile_regions=False,
            )
            decision = probe.validation.get("concurrency", {})
            workers = worker_count(probe, config)
            if workers == 1 and len(probe.plan.devices_used) > 1:
                workers = 2
                decision = {
                    **decision,
                    "enabled": True,
                    "workers": workers,
                    "intraop_threads": 0,
                    "reason": (
                        "heterogeneous placement retained for full executor benchmark; "
                        "CPU-only region microbenchmark is not representative"
                    ),
                }
            prefer_fused = workers == 1
            fused_config = replace(config, allow_concurrent_regions=False, max_concurrent_regions=1)
            if prefer_fused:
                # Sequential decision: only pay for the fused specialize.
                fused_program, fused_portable = _lower_to_portable(
                    exported,
                    name=name,
                    config=fused_config,
                    artifact_dir=artifact_dir,
                    force_single_region=True,
                    machine=_machine_for_fit,
                )
                fused_specialized = specialize_for_machine(
                    fused_portable,
                    config=fused_config,
                    output_dir=(artifact_dir / "specialized") if artifact_dir else None,
                    example_inputs=flat_inputs,
                    machine=_machine_for_fit,
                    measurements=fusion_measurements,
                )
                program, portable, specialized = fused_program, fused_portable, fused_specialized
                specialized.validation["concurrency"] = decision
                specialized.validation["fused_after_sequential_decision"] = True
                specialized.validation["fusion_probe_compile_skipped"] = True
                specialized.plan.notes.append(
                    "fused_to_single_region: single region is faster than multi-region execution"
                )
                workers = 1
            else:
                # Concurrent first (likely winner), then fused for the synchronized
                # wall-clock bake-off. Avoids writing fused artifacts when overlap wins.
                specialized = specialize_for_machine(
                    portable,
                    config=config,
                    output_dir=(artifact_dir / "specialized") if artifact_dir else None,
                    example_inputs=flat_inputs,
                    machine=_machine_for_fit,
                    measurements=fusion_measurements,
                )
                fused_program, fused_portable = _lower_to_portable(
                    exported,
                    name=name,
                    config=fused_config,
                    artifact_dir=artifact_dir,
                    force_single_region=True,
                    machine=_machine_for_fit,
                )
                fused_specialized = specialize_for_machine(
                    fused_portable,
                    config=fused_config,
                    output_dir=None,
                    example_inputs=flat_inputs,
                    machine=_machine_for_fit,
                    measurements=fusion_measurements,
                )
                specialized.validation["concurrency"] = decision
                from tensortorrent.runtime.graph_executor import _direct_path_wanted

                allow_dataflow_probe = _direct_path_wanted(config)
                concurrent_schedule_s = _time_executor(
                    program,
                    specialized.bindings,
                    flat_inputs,
                    workers=workers,
                    intraop_threads=intraop_threads(specialized, config),
                    enable_dataflow_direct_path=False,
                )
                fused_s = _time_executor(
                    fused_program,
                    fused_specialized.bindings,
                    flat_inputs,
                    workers=1,
                    intraop_threads=0,
                    enable_dataflow_direct_path=False,
                )
                concurrent_dataflow_s = (
                    _time_executor(
                        program,
                        specialized.bindings,
                        flat_inputs,
                        workers=workers,
                        intraop_threads=intraop_threads(specialized, config),
                        enable_dataflow_direct_path=True,
                    )
                    if allow_dataflow_probe
                    else float("inf")
                )
                prefer_fused, dataflow_enabled, concurrent_s, fuse_margin = _choose_fusion_candidate(
                    fused_s=fused_s,
                    concurrent_schedule_s=concurrent_schedule_s,
                    concurrent_dataflow_s=concurrent_dataflow_s,
                    hetero_plan=len(specialized.plan.devices_used) > 1,
                )
                compare_note = (
                    f"fusion_compare: concurrent={concurrent_s * 1e3:.3f}ms "
                    f"schedule={concurrent_schedule_s * 1e3:.3f}ms "
                    f"dataflow={concurrent_dataflow_s * 1e3:.3f}ms "
                    f"fused={fused_s * 1e3:.3f}ms prefer_fused={prefer_fused} "
                    f"dataflow_enabled={dataflow_enabled} fuse_margin={fuse_margin}"
                )
                measured_decision = {
                    "enabled": not prefer_fused,
                    "workers": 1 if prefer_fused else workers,
                    "group": decision.get("group", []),
                    "sequential_s": fused_s,
                    "parallel_s": concurrent_s,
                    "speedup": fused_s / concurrent_s if concurrent_s > 0 else 0.0,
                    "measured": True,
                    # Fused single-region runs do not keep a multi-worker thread split.
                    "intraop_threads": 0 if prefer_fused else int(decision.get("intraop_threads", 0)),
                    "reason": "full fused-vs-concurrent executor benchmark",
                    "dataflow_direct_path": False if prefer_fused else dataflow_enabled,
                }
                if prefer_fused:
                    if artifact_dir is not None:
                        fused_specialized.save(artifact_dir / "specialized")
                    program, portable, specialized = fused_program, fused_portable, fused_specialized
                    specialized.validation["concurrency"] = measured_decision
                    specialized.validation["fused_after_sequential_decision"] = True
                    specialized.plan.notes.append(compare_note)
                    specialized.plan.notes.append(
                        "fused_to_single_region: single region is faster than multi-region execution"
                    )
                    workers = 1
                else:
                    specialized.validation["concurrency"] = measured_decision
                    if dataflow_enabled:
                        specialized.validation["dataflow_direct_path"] = True
                    specialized.plan.notes = [
                        note for note in specialized.plan.notes if not note.startswith("concurrency=")
                    ]
                    specialized.plan.notes.extend(
                        ["concurrency=enabled: full executor benchmark selected overlap", compare_note]
                    )
    else:
        specialized = specialize_for_machine(
            portable,
            config=config,
            output_dir=(artifact_dir / "specialized") if artifact_dir else None,
            example_inputs=example_flat,
            machine=_machine_for_fit,
            measurements=measurements,
        )
        workers = worker_count(specialized, config)

    # Candidate probes may have written intermediate fused/coarse metadata into
    # the requested artifact directory. Persist the selected pair last so reload
    # always sees the exact program and plan returned here.
    if artifact_dir is not None:
        portable.save(artifact_dir)
        specialized.save(artifact_dir / "specialized")

    schedule = getattr(specialized, "schedule", None)
    store = build_parameter_store(
        program,
        portable,
        config,
        artifact_dir=artifact_dir,
        pack_lookup_dirs=pack_lookup_dirs,
        pin_memory=schedule_needs_host_pin(schedule),
    )
    _attach_storage_measurement(store, specialized)
    reuse_meta = portable.metadata.get("buffer_reuse") or specialized.profile.get("buffer_reuse") or {}
    reuse_assignment = dict(reuse_meta.get("assignment") or {})
    executor = GraphExecutor(
        program,
        specialized.bindings,
        parameter_store=store,
        max_workers=workers,
        prefetch_distance=specialized.plan.prefetch_distance,
        intraop_threads=intraop_threads(specialized, config),
        activation_budget_bytes=config.activation_budget_bytes,
        schedule=getattr(specialized, "schedule", None),
        buffer_reuse_assignment=reuse_assignment or None,
        process_workers=int(config.process_workers),
        machine=machine,
        config=config,
        enable_dataflow_direct_path=bool(specialized.validation.get("dataflow_direct_path")),
    )
    return CompiledModule(
        portable=portable,
        specialized=specialized,
        config=config,
        program=program,
        executor=executor,
        machine=machine,
        example_flat=example_flat,
    )


def _choose_fusion_candidate(
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
    # Multi-device overlap timers are noisier than single-region fused calls.
    # Require a clearer fused win before discarding a measured heterogeneous plan.
    fuse_margin = 1.10 if hetero_plan else 1.02
    prefer_fused = fused_s * fuse_margin <= concurrent_s
    if (
        prefer_fused
        and hetero_plan
        and dataflow_enabled
        and concurrent_schedule_s >= 1.5 * concurrent_dataflow_s
        and concurrent_s * 1.02 <= fused_s * 1.10
    ):
        # Dataflow removed large schedule overhead on a hetero plan and remains
        # within the fused band — keep the overlap candidate.
        prefer_fused = False
    return prefer_fused, dataflow_enabled, concurrent_s, fuse_margin


def _synchronize_bound_accelerators(bindings: dict[str, RegionBinding]) -> None:
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


def _time_executor(
    program: RegionProgram,
    bindings: dict[str, RegionBinding],
    flat_inputs: list[Any],
    *,
    workers: int,
    intraop_threads: int,
    iters: int = 7,
    enable_dataflow_direct_path: bool = False,
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
            prefetch_distance=0,
            intraop_threads=intraop_threads,
            enable_dataflow_direct_path=enable_dataflow_direct_path,
        )
        if enable_dataflow_direct_path:
            from tensortorrent.runtime.direct_path import DataflowDirectPlan

            if not isinstance(executor.direct_plan, DataflowDirectPlan):
                return float("inf")
        for _ in range(2):
            executor.run(list(flat_inputs))
        _synchronize_bound_accelerators(bindings)
        samples: list[float] = []
        for _ in range(max(1, iters)):
            _synchronize_bound_accelerators(bindings)
            start = time.perf_counter()
            executor.run(list(flat_inputs))
            _synchronize_bound_accelerators(bindings)
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


def _lower_to_portable(
    exported: Any,
    *,
    name: str,
    config: CompileConfig,
    artifact_dir: Path | None,
    force_single_region: bool,
    machine: ResourceGraph | None = None,
) -> tuple[RegionProgram, PortableArtifact]:
    """Lower, analyze, and pack one portable artifact from an exported program."""
    from tensortorrent.frontend.lower import lower_exported_program
    from tensortorrent.ir import (
        default_repeated_blocks,
        run_alias_analysis,
        run_liveness_analysis,
    )

    param_bytes = exported_parameter_bytes(exported)
    lowered = lower_exported_program(
        exported,
        name=name,
        max_region_nodes=config.max_region_nodes,
        max_region_state_bytes=_region_state_budget(config, machine, parameter_bytes=param_bytes),
        enable_linear_sharding=config.enable_linear_sharding and not config.allow_training,
        max_linear_shards=config.max_linear_shards,
        force_single_region=force_single_region,
    )
    ir = lowered.ir
    program = lowered.program
    alias = run_alias_analysis(ir)
    live = run_liveness_analysis(ir)
    ir.repeated_blocks = default_repeated_blocks(ir)
    ir.metadata["alias_groups"] = alias.groups
    ir.metadata["liveness"] = {k: list(v) for k, v in live.intervals.items()}
    portable = portable_compile_from_ir(
        ir,
        state_dict=program.state_dict_for_pack(),
        output_dir=artifact_dir,
        program=program,
        exported=exported,
    )
    return program, portable


def _example_flat_inputs(exported: Any, program: RegionProgram) -> list[Any] | None:
    """Recover the flat example inputs recorded by ``torch.export``."""
    args = getattr(exported, "example_inputs", None)
    if args is None:
        return None
    try:
        if isinstance(args, tuple) and len(args) == 2 and isinstance(args[1], dict):
            return program.flatten_inputs(args[0], args[1])
        return program.flatten_inputs(tuple(args), {})
    except Exception as exc:  # noqa: BLE001 - measurement is optional, correctness is not
        logger.warning(
            "Failed to recover flat example inputs from exported program: %s. "
            "Region measurement will be skipped; planning falls back to priors.",
            exc,
        )
        return None


def _check_early_fit(
    program: RegionProgram,
    machine: ResourceGraph,
    config: CompileConfig,
) -> None:
    """Check that the model can plausibly fit before expensive region capture.

    Raises :class:`~tensortorrent.errors.MemoryCapacityError` when it is
    definitely impossible to fit the model's parameters in the available memory.
    This is an *early* check (before benchmark / capture) and uses conservative
    estimates, so a pass here does not guarantee the plan will succeed.
    """
    from tensortorrent.ir.resource_graph import MemoryClass

    total_param_bytes = program.total_state_bytes()
    if total_param_bytes == 0:
        return  # nothing to check

    # Accumulate host allowed + per-device allowed + disk (when streaming is on)
    host_allowed = 0
    device_allowed_sum = 0
    disk_allowed = 0

    for mem in machine.memory.values():
        if mem.memory_class in (MemoryClass.NUMA_RAM, MemoryClass.PINNED_HOST):
            host_allowed += mem.allocatable_bytes
        elif mem.memory_class == MemoryClass.DEVICE_VRAM:
            cap = mem.allocatable_bytes
            if config.vram_budget_bytes is not None and cap > 0:
                cap = min(cap, config.vram_budget_bytes)
            device_allowed_sum += max(0, cap)
        elif mem.memory_class in (MemoryClass.NVME, MemoryClass.DISK_CACHE):
            disk_allowed += mem.allocatable_bytes

    streaming_permitted = config.allow_nvme_streaming and config.ram_budget_bytes is not None
    if streaming_permitted:
        total_allowed = host_allowed + device_allowed_sum + disk_allowed
        disk_info = f" disk_allowed={disk_allowed}"
    else:
        total_allowed = host_allowed + device_allowed_sum
        disk_info = ""

    if total_allowed > 0 and total_param_bytes > total_allowed:
        # Collect provenance details for an actionable error message
        host_src = "unknown"
        for mem in machine.memory.values():
            if mem.memory_class == MemoryClass.NUMA_RAM:
                host_src = mem.attributes.get("budget_source", "unknown")
                break
        raise MemoryCapacityError(
            f"Model parameters ({total_param_bytes} bytes) definitely cannot fit in "
            f"available memory: host_allowed={host_allowed} (source={host_src}) "
            f"device_allowed_sum={device_allowed_sum}{disk_info} "
            f"total_allowed={total_allowed}. "
            "Reduce the model size, raise memory budgets, or enable NVMe streaming."
        )
