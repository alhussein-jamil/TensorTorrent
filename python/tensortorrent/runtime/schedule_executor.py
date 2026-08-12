"""Instruction-DAG executor: ExecutableSchedule is the exclusive runtime program.

Every Prefetch / Load / Transfer / RecordEvent / WaitEvent / Compute / Evict /
Release op is dispatched when its ``depends_on`` instructions have completed.
Independent instructions may overlap; compute order need not match region order.
"""

from __future__ import annotations

import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from tensortorrent.backends.torch_device import coerce_region_result
from tensortorrent.compile.regions import RegionBinding, RegionProgram
from tensortorrent.errors import RuntimePlanError
from tensortorrent.ir.graph import OpCode
from tensortorrent.runtime.execution_context import ExecutionContext
from tensortorrent.runtime.schedule import ExecutableSchedule, PlanInstruction
from tensortorrent.runtime.schedule_report import InstructionEvent, ScheduleReport
from tensortorrent.runtime.tensor_store import ParameterStore

__all__ = [
    "ScheduleExecutor",
]


class ScheduleExecutor:
    """Runs an :class:`ExecutableSchedule` as an instruction dependency DAG."""

    def __init__(
        self,
        program: RegionProgram,
        bindings: dict[str, RegionBinding],
        schedule: ExecutableSchedule,
        *,
        parameter_store: ParameterStore,
        max_inflight: int = 8,
        max_workers: int = 1,
        process_pool: Any | None = None,
        fork_registry_id: int | None = None,
        callables: dict[str, Any] | None = None,
        allocator: Any | None = None,
        activation_budget_bytes: int | None = None,
        spill_events: list[dict[str, Any]] | None = None,
        reuse_assignment: dict[str, int] | None = None,
        machine: Any | None = None,
        device_workers: Any | None = None,
        hoist_resident_parameters: bool = True,
        config: Any | None = None,
    ) -> None:
        from tensortorrent.runtime.schedule import ensure_explicit_streams

        # Stream fill here; structural validation runs once in
        # NativeCompiledArtifact.from_schedule (Rust tt_ir::validate) — avoid a
        # second Py→Rust schedule_from_py convert on the hot construct path.
        schedule = ensure_explicit_streams(schedule)
        self.program = program
        self.bindings = bindings
        self.schedule = schedule
        self.parameter_store = parameter_store
        self.max_inflight = max(1, int(max_inflight))
        self.max_workers = max(1, int(max_workers))
        self.process_pool = process_pool
        self.fork_registry_id = fork_registry_id
        self.device_workers = device_workers
        self.allocator = allocator
        self.activation_budget_bytes = activation_budget_bytes
        self._spill_events = spill_events if spill_events is not None else []
        self._reuse_assignment = dict(reuse_assignment or {})
        self.machine = machine
        self._config = config
        self._hoist_resident_parameters = bool(hoist_resident_parameters)
        self._partial_hoist_oom = False
        self._persistent_parameter_ids = self._select_persistent_parameter_ids(schedule)
        self._by_name = {i.name: i for i in schedule.instructions}
        if callables is not None:
            self._callables = callables
        else:
            self._callables = {
                rid: getattr(binding.compiled, "executable", binding.compiled) for rid, binding in bindings.items()
            }
        from tensortorrent.runtime.inflight import InFlightGate

        self._run_gate = InFlightGate()
        self._cancel = False
        self._cancel_lock = threading.Lock()
        self._active_cancels: list[Any] = []
        self._closed = False
        # Region-wave pool for concurrent Computes.
        self._region_pool: ThreadPoolExecutor | None = None
        self._region_pool_threads: int | None = None
        self._native_artifact: Any | None = None
        # Resident parameters may be hoisted to their scheduled device once and
        # reused across forwards. Per-run residency metadata stays isolated; only
        # immutable tensor values live in this executor cache.
        self._persistent_param_lock = threading.Lock()
        self._persistent_device_param_cache: dict[tuple[str, str], Any] = {}
        self._input_transfer_destinations: dict[str, str] = {}
        self._install_native_artifact(schedule)
        self._recompute_schedule_caches(schedule)

    def _ensure_region_pool(self, workers: int, *, threads: int | None = None) -> ThreadPoolExecutor:
        """Thread pool for independent Compute waves on the native path.

        ``threads`` is the OpenMP/intra-op budget for worker threads. Callers that
        temporarily pinch ``torch.set_num_threads`` on the main thread (shared
        microbenchmarks) must pass the unpinned CPU budget so overlapped CPU
        regions do not inherit the pinch.
        """
        n = max(1, int(workers))
        thread_count = max(1, int(threads) if threads is not None else torch.get_num_threads())
        if self._region_pool is None:
            self._region_pool = ThreadPoolExecutor(
                max_workers=n,
                thread_name_prefix="tt-region",
                initializer=torch.set_num_threads,
                initargs=(thread_count,),
            )
            self._region_pool_threads = thread_count
            return self._region_pool
        if int(getattr(self._region_pool, "_max_workers", n)) < n:
            self._region_pool.shutdown(wait=False, cancel_futures=True)
            self._region_pool = ThreadPoolExecutor(
                max_workers=n,
                thread_name_prefix="tt-region",
                initializer=torch.set_num_threads,
                initargs=(thread_count,),
            )
            self._region_pool_threads = thread_count
        return self._region_pool

    def _install_native_artifact(self, schedule: ExecutableSchedule) -> None:
        from tensortorrent.native import require_native
        from tensortorrent.runtime.schedule import ScheduleValidationError

        native = require_native()
        runtime_schedule = self._steady_state_schedule(schedule)
        self._native_instruction_names = tuple(inst.name for inst in runtime_schedule.instructions)
        try:
            self._native_artifact = native.NativeCompiledArtifact.from_schedule(runtime_schedule)
        except Exception as exc:
            # from_schedule runs tt_ir::validate — surface as plan error.
            msg = str(exc)
            raise RuntimePlanError(
                f"ExecutableSchedule {schedule.graph_name!r} failed validation: {msg}"
            ) from ScheduleValidationError(msg)

    def _parameter_nbytes_from_schedule(self, schedule: ExecutableSchedule) -> dict[str, int]:
        """Best-effort nbytes per parameter tensor id referenced by H2D transfers.

        Prefer static ``program.values`` metadata — never ``store.acquire`` here.
        Acquiring multi-GiB packs during executor init previously doubled host RAM.
        """
        sizes: dict[str, int] = {}
        values = getattr(self.program, "values", {}) or {}
        for inst in schedule.instructions:
            if inst.opcode != OpCode.TRANSFER:
                continue
            if str(inst.attributes.get("kind") or "") != "parameter_host_to_device":
                continue
            for tensor_id in inst.outputs or inst.inputs or ():
                name = str(tensor_id)
                if name in sizes:
                    continue
                spec = values.get(name)
                nbytes = int(getattr(spec, "nbytes", 0) or 0) if spec is not None else 0
                if nbytes <= 0:
                    tensor_nbytes = (inst.attributes or {}).get("tensor_nbytes") or {}
                    if isinstance(tensor_nbytes, dict) and name in tensor_nbytes:
                        nbytes = int(tensor_nbytes.get(name) or 0)
                if nbytes <= 0:
                    nbytes = int(getattr(inst, "nbytes", 0) or 0) if len(inst.outputs or inst.inputs or ()) == 1 else 0
                if nbytes <= 0 and name in getattr(self.program, "state_bindings", {}):
                    # Live module attribute (no pack materialize) — nbytes only.
                    try:
                        tensor = self.program.state_tensor(name)
                        if torch.is_tensor(tensor) and int(tensor.numel()) > 0:
                            nbytes = int(tensor.numel()) * int(tensor.element_size())
                    except Exception:  # noqa: BLE001
                        nbytes = 0
                sizes[name] = max(0, nbytes)
        return sizes

    def _select_persistent_parameter_ids(self, schedule: ExecutableSchedule) -> set[str] | None:
        """``None`` = full hoist; ``set`` = partial (possibly empty = transfer/evict)."""
        if not self._hoist_resident_parameters or getattr(self.parameter_store, "needs_prefetch", False):
            return set()
        from tensortorrent.compile.fit import (
            cuda_device_index_from_resource,
            live_hoist_budget_bytes,
            select_persistent_parameter_ids,
            should_hoist_resident_parameters,
        )

        try:
            state_bytes = int(self.program.total_state_bytes())
        except Exception:  # noqa: BLE001
            state_bytes = 0
        cfg = self._config
        if self._partial_hoist_oom:
            return set()
        if cfg is not None and should_hoist_resident_parameters(cfg, state_bytes=state_bytes, machine=self.machine):
            return None  # full hoist
        if cfg is None:
            # No config → preserve historical all-or-nothing True behavior.
            return None if self._hoist_resident_parameters else set()
        sizes = self._parameter_nbytes_from_schedule(schedule)
        if not sizes:
            return set()
        target_indices: set[int] = set()
        transfer_groups: list[tuple[str, ...]] = []
        for inst in schedule.instructions:
            if inst.opcode != OpCode.TRANSFER:
                continue
            if str(inst.attributes.get("kind") or "") != "parameter_host_to_device":
                continue
            index = cuda_device_index_from_resource(str(inst.destination or ""))
            if index is not None:
                target_indices.add(index)
            names = tuple(str(t) for t in (inst.outputs or inst.inputs or ()) if str(t) in sizes)
            if names:
                transfer_groups.append(names)
        budget = live_hoist_budget_bytes(
            cfg,
            self.machine,
            device_indices=target_indices or None,
            synchronize=True,
        )
        if budget is None:
            return None
        selected = select_persistent_parameter_ids(
            sizes,
            budget_bytes=int(budget),
            transfer_groups=transfer_groups or None,
        )
        return selected

    def _steady_state_schedule(self, schedule: ExecutableSchedule) -> ExecutableSchedule:
        """Drop one-time resident parameter transfers from repeated inference.

        Canonical ``self.schedule`` retains initialization Transfers for explain,
        simulation, and validation. The installed native artifact runs only the
        steady-state DAG after immutable parameter copies have been hoisted.

        Matching ``parameter_evict`` ops are dropped too: device copies live in
        the executor's persistent cache and are re-seeded each forward, so
        replaying evict bookkeeping every inference is pure overhead.

        Partial residency hoists only ``_persistent_parameter_ids``; the rest
        keep transfer/evict each forward.
        """
        if not self._hoist_resident_parameters or getattr(self.parameter_store, "needs_prefetch", False):
            return schedule
        selected = self._persistent_parameter_ids
        if selected is not None and not selected:
            return schedule
        from tensortorrent.runtime.schedule import hoist_resident_parameter_transfers

        return hoist_resident_parameter_transfers(
            schedule,
            drop_parameter_evicts=True,
            tensor_ids=selected,
        )

    def _recompute_schedule_caches(self, schedule: ExecutableSchedule) -> None:
        """Derive per-forward-invariant schedule facts once instead of per call.

        The schedule is frozen after compile/replan, so which instructions need
        spill callbacks, parameter-load callbacks, or resolve to a given region
        never changes between forwards — compute it once here and let the native
        bridge read the cached result every forward.
        """
        self._compute_by_region = {
            str(inst.executable_ref or ""): inst for inst in schedule.instructions if inst.opcode == OpCode.COMPUTE
        }
        self._needs_spill_callbacks = any(
            (inst.opcode == OpCode.EVICT and str(inst.attributes.get("kind") or "") == "activation_spill")
            or (inst.opcode == OpCode.LOAD and str(inst.attributes.get("kind") or "") == "activation_reload")
            for inst in schedule.instructions
        )
        needs_prefetch = bool(getattr(self.parameter_store, "needs_prefetch", False))
        self._needs_parameter_load = needs_prefetch and any(
            inst.opcode == OpCode.LOAD and str(inst.attributes.get("kind") or "") == "parameter_materialize"
            for inst in schedule.instructions
        )
        self._mock_resources = sorted(
            {str(inst.resource) for inst in schedule.instructions if "mock" in str(inst.resource).lower()}
        )
        host = self._default_host_resource()
        self._alias_target_resources = tuple(
            str(inst.resource)
            for inst in schedule.instructions
            if inst.opcode == OpCode.COMPUTE
            and str(inst.resource) != host
            and "mock" not in str(inst.resource).lower()
            and not _tier_is_device(str(inst.resource))
        )
        resident_targets: dict[str, set[str]] = {}
        persistent = self._persistent_parameter_ids
        for inst in schedule.instructions:
            if inst.opcode != OpCode.TRANSFER or str(inst.attributes.get("kind") or "") != "parameter_host_to_device":
                continue
            destination = str(inst.destination or inst.resource)
            if "mock" in destination.lower():
                continue
            for tensor_id in inst.outputs or inst.inputs:
                name = str(tensor_id)
                # None → full hoist; set → only selected ids stay device-resident.
                if persistent is not None and name not in persistent:
                    continue
                resident_targets.setdefault(name, set()).add(destination)
        self._resident_parameter_targets = {
            tensor_id: tuple(sorted(destinations)) for tensor_id, destinations in resident_targets.items()
        }
        user_inputs = {str(name) for name in self.program.user_inputs}
        input_destinations: dict[str, str] = {}
        for inst in schedule.instructions:
            if inst.opcode != OpCode.TRANSFER:
                continue
            destination = str(inst.destination or inst.resource)
            if not destination or "mock" in destination.lower():
                continue
            for tensor_id in inst.outputs or inst.inputs:
                name = str(tensor_id)
                if name not in user_inputs:
                    continue
                input_destinations.setdefault(name, destination)
        self._input_transfer_destinations = input_destinations
        # Tensor identities may differ under a new schedule — drop the stale cache.
        self._persistent_param_cache = None
        self._persistent_device_param_cache.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.request_cancel()
        self._run_gate.mark_closed_and_wait()
        if self._closed:
            return
        self._closed = True
        self._cancel = True
        self._persistent_param_cache = None
        self._persistent_device_param_cache.clear()
        if self._region_pool is not None:
            self._region_pool.shutdown(wait=True, cancel_futures=True)
            self._region_pool = None
        self._last_native_ctx = None

    def release_device_residency(self, *, demote_hoist: bool = False) -> bool:
        """Drop hoisted device weights and rebuild transfer/evict runtime.

        LATENCY plans hoist resident params once and strip Transfers. Fine when
        the card has headroom; OOMs when peers/activations need the same memory.

        ``demote_hoist=False`` — generation-local (``_partial_hoist_oom``), same
        as the residency OOM path.
        ``demote_hoist=True`` — turn hoist off for this executor's lifetime.

        Idempotent when already demoted: clears the device cache only (no native
        artifact rebuild) so per-forward prep stays cheap.
        """
        if self._closed:
            return False
        self._run_gate.wait_idle()
        already_demoted = demote_hoist and not bool(self._hoist_resident_parameters)
        self._persistent_device_param_cache.clear()
        self._persistent_param_cache = None
        if already_demoted:
            if torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
            return True
        self._resident_parameter_targets = {}
        self._persistent_parameter_ids = set()
        if demote_hoist:
            self._hoist_resident_parameters = False
        self._partial_hoist_oom = True
        try:
            self._install_native_artifact(self.schedule)
            self._recompute_schedule_caches(self.schedule)
        except Exception as exc:
            raise RuntimePlanError("failed to rebuild transfer/evict runtime after releasing device residency") from exc
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        return True

    def replace_schedule(self, schedule: ExecutableSchedule) -> None:
        """Install a new immutable schedule (e.g. attribute annotations for tests)."""
        from tensortorrent.runtime.schedule import ScheduleValidationError, ensure_explicit_streams, validate_schedule

        self._run_gate.wait_idle()
        if self._closed:
            raise RuntimePlanError("ScheduleExecutor is closed")
        schedule = ensure_explicit_streams(schedule)
        violations = validate_schedule(schedule)
        if violations:
            raise RuntimePlanError(
                f"ExecutableSchedule {schedule.graph_name!r} failed validation: {violations}"
            ) from ScheduleValidationError(str(violations))
        self.schedule = schedule
        self._by_name = {i.name: i for i in schedule.instructions}
        self._partial_hoist_oom = False
        self._persistent_parameter_ids = self._select_persistent_parameter_ids(schedule)
        self._install_native_artifact(schedule)
        self._recompute_schedule_caches(schedule)

    def request_cancel(self) -> None:
        with self._cancel_lock:
            self._cancel = True
            tokens = list(self._active_cancels)
        for tok in tokens:
            with contextlib.suppress(Exception):
                tok.cancel()

    def run(
        self,
        flat_inputs: list[Any],
        *,
        cancel_token: Any | None = None,
        enable_grad: bool = False,
    ) -> tuple[list[Any], ScheduleReport]:
        if self._closed:
            raise RuntimePlanError("ScheduleExecutor is closed")
        try:
            self._run_gate.enter()
        except RuntimeError as exc:
            raise RuntimePlanError("ScheduleExecutor is closed") from exc
        try:
            from tensortorrent.runtime.native_bridge import run_schedule_native

            return run_schedule_native(self, flat_inputs, cancel_token=cancel_token, enable_grad=bool(enable_grad))
        finally:
            self._run_gate.leave()

    def _default_host_resource(self) -> str:
        for binding in self.bindings.values():
            if "cpu" in binding.device or "numa" in binding.device:
                return binding.device
        return "cpu"

    def _protected_budget_tensors(self) -> set[str]:
        """Tensors the planner refuses to spill (inputs + graph outputs)."""
        protected: set[str] = set(self.program.user_inputs)
        for kind, ref in getattr(self.program, "output_refs", ()):
            if kind == "value":
                protected.add(str(ref))
        protected.update(getattr(self.program, "user_outputs", ()) or ())
        return protected

    def _pending_spill_tensors(self, completed: set[str]) -> set[str]:
        """Activation tensors still waiting on a scheduled spill Evict."""
        pending: set[str] = set()
        for inst in self.schedule.instructions:
            if inst.name in completed:
                continue
            if inst.opcode == OpCode.EVICT and inst.attributes.get("kind") == "activation_spill":
                pending.update(inst.inputs)
        return pending

    def _assert_activation_budget(self, ctx: ExecutionContext, completed: set[str]) -> None:
        """Fail when durable activation residency exceeds the configured budget.

        Matches planner semantics: disk spills and protected tensors may leave
        live bytes above budget only while a pending spill still covers every
        spillable resident activation. Transient overage between Compute and its
        dependent Evict is allowed; leftover spillable residency is not.
        """
        budget = self.activation_budget_bytes
        live = ctx.copies.activation_live_bytes()
        ctx.note_activation_live(live)
        if budget is None:
            return
        if live <= int(budget):
            return
        protected = self._protected_budget_tensors()
        pending = self._pending_spill_tensors(completed)
        spillable = sorted(
            tid for tid in ctx.copies.activation_tensor_ids() if tid not in protected and tid not in pending
        )
        if not spillable:
            return
        raise RuntimePlanError(f"activation budget {int(budget)} bytes exceeded: live={live} spillable={spillable}")

    # Compute used by native_bridge region callback too.

    def _exec_compute(self, inst: PlanInstruction, ctx: ExecutionContext, submitted: float) -> InstructionEvent:
        region_id = str(inst.executable_ref or "")
        binding = self.bindings[region_id]
        region = binding.region
        resource = binding.device
        enable_grad = bool(getattr(ctx, "enable_grad", False))

        from tensortorrent.runtime.activation_spill import is_spilled
        from tensortorrent.runtime.virtual_tensor import unwrap_for_compute

        args: list[Any] = []
        for name in region.inputs:
            copy = None
            value: Any = None
            if ctx.copies.has(name, resource):
                copy = ctx.copies.require(name, resource)
                value = copy.value
                if ctx.native_residency is not None and not ctx.native_residency.session.has(name, resource):
                    raise RuntimePlanError(
                        f"Compute {region_id} on {resource}: CopyStore has {name!r} but native "
                        f"residency does not (Rust is residency authority)"
                    )
            elif ctx.native_residency is not None and ctx.native_residency.session.has(name, resource):
                # Native Transfer may have registered dest residency before CopyStore.
                value = ctx.native_residency.require_value(name, resource)
                from tensortorrent.runtime.virtual_tensor import VirtualDeviceTensor, wrap_virtual_native

                nctx = getattr(ctx, "native_execution_context", None)
                # Training keeps live host tensors on mock resources (byte wrap detaches).
                if not enable_grad:
                    if "mock" in resource.lower() and not isinstance(value, VirtualDeviceTensor):
                        if nctx is None:
                            raise RuntimePlanError(f"Compute {region_id}: mock wrap requires NativeExecutionContext")
                        value = wrap_virtual_native(value, resource, nctx)
                    elif "mock" not in resource.lower() and isinstance(value, VirtualDeviceTensor):
                        value = value.to_host()
                elif isinstance(value, VirtualDeviceTensor):
                    value = value.payload
                if ctx.copies.has(name, ctx.host_resource):
                    ctx.publish_replica(
                        name,
                        resource,
                        value,
                        ownership="transfer",
                        source_resource=ctx.host_resource,
                    )
                else:
                    ctx.publish_tensor(name, resource, value, ownership="transfer")
                # Native already has residency; publish_* refreshes Python handles only.
                if nctx is not None and isinstance(value, VirtualDeviceTensor) and value.native_buffer_id is not None:
                    nctx.bind_virtual_buffer(name, resource, int(value.native_buffer_id))
                copy = ctx.copies.require(name, resource)
            else:
                raise RuntimePlanError(
                    f"Compute {region_id} on {resource}: required copy of {name!r} missing "
                    f"(schedule must Load/Transfer before Compute; no hidden materialization)"
                )
            if is_spilled(copy.value):
                raise RuntimePlanError(
                    f"Compute {region_id}: {name!r} still spilled on {resource!r}; "
                    f"schedule must emit activation_reload Load before Compute"
                )
            args.append(unwrap_for_compute(copy.value, resource=resource, allow_host_alias=enable_grad))

        call = self._callables[region_id]

        if enable_grad:
            workers = self.device_workers
            if workers is not None and resource in getattr(workers, "device_ids", ()):
                raise RuntimePlanError(
                    f"Compute {region_id}: schedule training cannot use device workers "
                    "(they detach tensors). Compile with in-process execution for training."
                )
            if self.process_pool is not None and self.fork_registry_id is not None and "mock" not in resource:
                raise RuntimePlanError(
                    f"Compute {region_id}: schedule training cannot use process_workers "
                    "(fork detaches tensors). Set process_workers=0 for training."
                )

        workers = self.device_workers
        if not enable_grad and workers is not None and resource in getattr(workers, "device_ids", ()):
            from tensortorrent.runtime.device_workers import run_region_on_device

            region_event, outputs = workers.submit(
                resource,
                run_region_on_device,
                call,
                resource,
                binding.backend_id,
                region_id,
                tuple(_detach_for_worker(a) for a in args),
            ).result()
            for out_name, value in zip(region.outputs, outputs, strict=True):
                ctx.publish_tensor(out_name, resource, value, ownership="activation")
            return InstructionEvent(
                name=inst.name,
                opcode=inst.opcode.value,
                resource=resource,
                submitted_s=submitted,
                start_s=region_event["start_s"],
                end_s=region_event["end_s"],
                notes=f"Compute {region_id} (device-worker)",
                region_id=region_id,
            )

        if (
            not enable_grad
            and self.process_pool is not None
            and self.fork_registry_id is not None
            and "mock" not in resource
        ):
            from tensortorrent.runtime.fork_regions import fork_run_region

            region_event, outputs = self.process_pool.submit(
                fork_run_region,
                self.fork_registry_id,
                region_id,
                resource,
                binding.backend_id,
                tuple(_detach_for_worker(a) for a in args),
            ).result()
            for out_name, value in zip(region.outputs, outputs, strict=True):
                ctx.publish_tensor(out_name, resource, value, ownership="activation")
            return InstructionEvent(
                name=inst.name,
                opcode=inst.opcode.value,
                resource=resource,
                submitted_s=submitted,
                start_s=region_event.start_s,
                end_s=region_event.end_s,
                notes=f"Compute {region_id} (process)",
                region_id=region_id,
            )

        start = time.perf_counter()
        if enable_grad or torch.is_inference_mode_enabled():
            result = call(*args)
        else:
            with torch.inference_mode():
                result = call(*args)
        outputs = coerce_region_result(result)
        if len(outputs) != len(region.outputs):
            raise RuntimePlanError(f"Region {region_id} produced {len(outputs)} values, expected {len(region.outputs)}")
        for out_name, value in zip(region.outputs, outputs, strict=True):
            # Buffer reuse overwrites storage; unsafe while autograd holds saved tensors.
            if not enable_grad and self.allocator is not None and isinstance(value, torch.Tensor):
                slot = self._reuse_assignment.get(out_name)
                if slot is not None:
                    value = self.allocator.acquire(slot, out_name, value)
            from tensortorrent.runtime.virtual_tensor import VirtualDeviceTensor, wrap_virtual_native

            nctx = getattr(ctx, "native_execution_context", None)
            # Inference mock path owns a native virtual buffer; training keeps the
            # live activation tensor so backward can see grad_fn.
            if "mock" in resource and not enable_grad:
                if nctx is None:
                    raise RuntimePlanError(f"Compute {region_id}: mock wrap requires NativeExecutionContext")
                value = wrap_virtual_native(value, resource, nctx)
            ctx.publish_tensor(out_name, resource, value, ownership="activation")
            if nctx is not None and isinstance(value, VirtualDeviceTensor) and value.native_buffer_id is not None:
                nctx.bind_virtual_buffer(out_name, resource, int(value.native_buffer_id))
        end = time.perf_counter()
        return InstructionEvent(
            name=inst.name,
            opcode=inst.opcode.value,
            resource=resource,
            submitted_s=submitted,
            start_s=start,
            end_s=end,
            notes=f"Compute {region_id}",
            region_id=region_id,
        )

    def _collect_outputs(self, ctx: ExecutionContext) -> list[Any]:
        host = ctx.host_resource
        flat: list[Any] = []
        for kind, ref in self.program.output_refs:
            if kind != "value":
                flat.append(ref)
                continue
            name = str(ref)
            resources = ctx.copies.resources_for(name)
            if not resources:
                raise RuntimePlanError(f"Missing output {name}")
            # Prefer accelerator-resident copies so CompiledModule matches
            # ``nn.Module`` device semantics (outputs stay on the compute device).
            # Host is the fallback when no device copy exists.
            chosen = next((r for r in resources if _tier_is_device(r)), None)
            if chosen is None:
                chosen = host if host in resources else resources[0]
            copy = ctx.copies.get(name, chosen)
            value = copy.value
            from tensortorrent.runtime.activation_spill import is_spilled
            from tensortorrent.runtime.virtual_tensor import VirtualDeviceTensor

            if is_spilled(value):
                raise RuntimePlanError(f"Output {name!r} still spilled; schedule must reload before collect")
            if isinstance(value, VirtualDeviceTensor):
                value = value.to_host()
            flat.append(value)
        return flat


def _tier_is_device(resource: str) -> bool:
    from tensortorrent.backends import backend_id_for_resource
    from tensortorrent.runtime.resource_names import is_device_resource

    return is_device_resource(resource) or backend_id_for_resource(resource) != "cpu"


def _detach_for_worker(value: Any) -> Any:
    """Detach tensors before process/device-worker submit (breaks autograd by design)."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (tuple, list)):
        return type(value)(_detach_for_worker(v) for v in value)
    if isinstance(value, dict):
        return {k: _detach_for_worker(v) for k, v in value.items()}
    return value
