"""PyTorch-compatible compiled module."""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from tensortorrent.compile.pipeline import PortableArtifact, SpecializedArtifact
from tensortorrent.compile.regions import RegionProgram, restore_sharded_state_dict
from tensortorrent.config import CompileConfig
from tensortorrent.errors import RuntimePlanError, UnsupportedFeatureError
from tensortorrent.hardware.discovery import discover_resource_graph
from tensortorrent.observability import write_chrome_trace
from tensortorrent.runtime.capacity import CapacityLedger, build_module_capacity_ledger
from tensortorrent.runtime.graph_executor import ExecutionReport, GraphExecutor
from tensortorrent.runtime.simulator import simulate_schedule

logger = logging.getLogger(__name__)


class _ExecutorGenerationManager:
    """Lease executor generations across atomic live replanning.

    A replacement becomes visible immediately to new forwards. The previous
    generation is closed only after its final in-flight forward releases its
    lease, preventing profile-guided replanning from invalidating concurrent
    requests.
    """

    def __init__(self, initial: Any, closer: Callable[[Any], None]) -> None:
        self._lock = threading.RLock()
        self._current = initial
        self._closer = closer
        self._references: dict[int, int] = {}
        self._retired: dict[int, Any] = {}
        self._closed = False

    def acquire(self) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimePlanError("CompiledModule is closed")
            executor = self._current
            key = id(executor)
            self._references[key] = self._references.get(key, 0) + 1
            return executor

    def release(self, executor: Any) -> None:
        close_after: Any | None = None
        with self._lock:
            key = id(executor)
            count = self._references.get(key, 0)
            if count <= 0:
                raise RuntimePlanError("Executor lease released without a matching acquisition")
            if count == 1:
                self._references.pop(key, None)
                close_after = self._retired.pop(key, None)
            else:
                self._references[key] = count - 1
        if close_after is not None:
            self._closer(close_after)

    def swap(self, replacement: Any) -> Any:
        close_after: Any | None = None
        with self._lock:
            if self._closed:
                raise RuntimePlanError("CompiledModule is closed")
            previous = self._current
            self._current = replacement
            key = id(previous)
            if self._references.get(key, 0) > 0:
                self._retired[key] = previous
            else:
                close_after = previous
        if close_after is not None:
            self._closer(close_after)
        return previous

    def cancel_all(self) -> None:
        with self._lock:
            executors = [self._current, *self._retired.values()]
        seen: set[int] = set()
        for executor in executors:
            key = id(executor)
            if key in seen:
                continue
            seen.add(key)
            if hasattr(executor, "request_cancel"):
                executor.request_cancel()

    def close(self) -> None:
        close_now: list[Any] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True
            generations = [self._current, *self._retired.values()]
            self._retired.clear()
            for executor in generations:
                key = id(executor)
                if self._references.get(key, 0) > 0:
                    self._retired[key] = executor
                else:
                    close_now.append(executor)
        seen: set[int] = set()
        for executor in close_now:
            key = id(executor)
            if key in seen:
                continue
            seen.add(key)
            self._closer(executor)


class CompiledModule(torch.nn.Module):
    """A compiled model that behaves like any other ``torch.nn.Module``.

    Default (``allow_training=False``): starts in ``eval()``; ``forward`` runs the
    planned heterogeneous schedule under ``torch.inference_mode``.

    With ``CompileConfig(allow_training=True)``: starts in ``train()``. While
    training, ``forward`` runs the heterogeneous schedule with autograd enabled
    so ``loss.backward()`` and ``optimizer.step()`` work. Call ``.eval()`` to
    switch back to the inference schedule (same updated weights, ``inference_mode``).

    Concurrent ``forward`` calls on the same instance are supported: each
    forward uses an independent execution context sharing the immutable
    native artifact.
    """

    def __init__(
        self,
        *,
        portable: PortableArtifact,
        specialized: SpecializedArtifact,
        config: CompileConfig,
        program: RegionProgram,
        executor: GraphExecutor,
        machine: Any | None = None,
        example_flat: list[Any] | None = None,
    ) -> None:
        super().__init__()
        self.portable = portable
        self.specialized = specialized
        self.config = config
        self._program = program
        self._executor = executor
        self._executor_generations = _ExecutorGenerationManager(executor, self._close_executor_resources)
        self._replan_lock = threading.Lock()
        # Pairs the live executor generation with the capacity ledger that was
        # built for that exact generation. Never publish either independently.
        self._runtime_state_lock = threading.RLock()
        self._machine = machine
        self._example_flat = example_flat
        # Held in a dict because nn.Module.__setattr__ is too expensive to run on
        # every forward just to record the last report.
        self._reports: dict[str, ExecutionReport] = {}
        from tensortorrent.runtime.profile_feedback import ProfileFeedback

        self._profile_feedback = ProfileFeedback()
        # Partitioned FX root for parameters / state_dict / .to(). Call the
        # CompiledModule itself for forward — not this attribute (schedule path).
        # Export-free fused CPU reuses the caller's nn.Module as program.root —
        # do not register it as a submodule or CompiledModule.eval()/train()
        # would permanently mutate the caller's train/eval mode.
        root = program.root
        if getattr(program, "is_export_free", False):
            object.__setattr__(self, "graph_module", root)
        else:
            self.graph_module = root
        self._closed = False
        self._report_lock = threading.Lock()
        self._capacity_ledger = build_module_capacity_ledger(
            program=program,
            plan=getattr(specialized, "plan", None),
            config=config,
            parameter_store=getattr(executor, "parameter_store", None),
            machine=machine,
        )
        # Inference-first default; training opt-in starts ready for a train loop.
        # Set mode through our override so torch.export children that reject
        # train()/eval() do not abort construction.
        if config.allow_training:
            self.train(True)
        else:
            self.training = False
            for child in self.children():
                with contextlib.suppress(NotImplementedError):
                    child.train(False)

    @property
    def capacity_ledger(self) -> CapacityLedger:
        """Snapshot the current generation's shared capacity ledger."""
        with self._runtime_state_lock:
            return self._capacity_ledger

    # ---- nn.Module contract ----------------------------------------
    def train(self, mode: bool = True) -> CompiledModule:
        """Switch train/eval like a normal ``nn.Module``.

        ``.train()`` requires ``CompileConfig(allow_training=True)`` — without it
        the schedule always runs under ``inference_mode`` and gradients never
        appear. With the opt-in, ``.train()`` runs the schedule with autograd and
        ``.eval()`` returns to the max-performance inference schedule.
        """
        if mode and not self.config.allow_training:
            raise UnsupportedFeatureError(
                "CompiledModule.train() requires CompileConfig(allow_training=True). "
                "Default compile stays on the inference schedule for max performance; "
                "with the opt-in, .train() runs the schedule with autograd and .eval() "
                "uses the inference schedule."
            )
        self.training = mode
        for child in self.children():
            with contextlib.suppress(NotImplementedError):
                child.train(mode)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if "enable_grad" in kwargs:
            raise TypeError(
                "CompiledModule.forward() got unexpected keyword argument 'enable_grad' "
                "(train mode is controlled by .train() / .eval(), not forward kwargs)"
            )
        enable_grad = bool(self.config.allow_training and self.training)
        if enable_grad or torch.is_inference_mode_enabled():
            return self.forward_with_cancel_token(None, *args, enable_grad=enable_grad, **kwargs)
        with torch.inference_mode():
            return self.forward_with_cancel_token(None, *args, enable_grad=False, **kwargs)

    def forward_with_cancel_token(
        self,
        cancel_token: Any | None,
        *args: Any,
        enable_grad: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute one forward with an optional request-scoped native cancel token.

        Serving uses this so timing out one request does not cancel unrelated
        concurrent forwards on the same compiled module. When ``enable_grad`` is
        True (train mode), the schedule runs without ``inference_mode`` so
        autograd can record through region Computes.
        """
        # Capacity ownership lives here, not in the serving layer. Snapshot and
        # acquire the ledger together with the executor generation so a live
        # replan cannot make a request release a different generation's ledger.
        with self._runtime_state_lock:
            if self._closed:
                raise RuntimePlanError("CompiledModule is closed")
            ledger = self._capacity_ledger
            ledger.acquire_or_raise()
            try:
                executor = self._executor_generations.acquire()
            except BaseException:
                ledger.release()
                raise
        try:
            flat_inputs = self._program.flatten_inputs(args, kwargs)
            flat_outputs, report = executor.run(flat_inputs, cancel_token=cancel_token, enable_grad=enable_grad)
        finally:
            self._executor_generations.release(executor)
            ledger.release()
        with self._report_lock:
            self._reports["last"] = report
        # Train timings poison infer placement priors — only fold eval/infer runs.
        if self.config.online_profile_feedback and not enable_grad:
            self._profile_feedback.observe_report(report)
        if self._program.single_output and len(flat_outputs) == 1:
            return flat_outputs[0]
        return self._program.unflatten_outputs(flat_outputs)

    def apply_profile_feedback(self) -> dict[str, Any]:
        """Re-specialize and atomically replace the live executor generation.

        In-flight forwards retain a lease on the previous generation. Its worker
        pools and parameter store are closed only after the final such request
        completes.
        """
        from tensortorrent.compile.pipeline import specialize_for_machine
        from tensortorrent.runtime.provisioning import (
            build_parameter_store,
            intraop_threads,
            schedule_needs_host_pin,
            worker_count,
        )

        with self._replan_lock:
            if self._closed:
                raise RuntimePlanError("CompiledModule is closed")
            machine = self._machine if self._machine is not None else discover_resource_graph()
            old_plan = self.specialized.plan
            old_latency = float(getattr(old_plan, "predicted_latency_s", 0.0) or 0.0)
            old_devices = tuple(getattr(old_plan, "devices_used", ()) or ())
            old_placements = {p.region_id: p.device for p in getattr(old_plan, "placements", ()) or ()}
            specialized = specialize_for_machine(
                self.portable,
                config=self.config,
                example_inputs=self._example_flat,
                machine=machine,
                profile_feedback=self._profile_feedback,
            )
            store = build_parameter_store(
                self._program,
                self.portable,
                self.config,
                pin_memory=schedule_needs_host_pin(getattr(specialized, "schedule", None)),
                machine=machine,
            )
            reuse_meta = self.portable.metadata.get("buffer_reuse") or specialized.profile.get("buffer_reuse") or {}
            reuse_assignment = dict(reuse_meta.get("assignment") or {})
            workers = worker_count(specialized, self.config)
            replacement = GraphExecutor(
                self._program,
                specialized.bindings,
                parameter_store=store,
                max_workers=workers,
                prefetch_distance=specialized.plan.prefetch_distance,
                intraop_threads=intraop_threads(specialized, self.config),
                activation_budget_bytes=self.config.activation_budget_bytes,
                schedule=getattr(specialized, "schedule", None),
                buffer_reuse_assignment=reuse_assignment or None,
                process_workers=int(self.config.process_workers),
                machine=machine,
                config=self.config,
                enable_dataflow_direct_path=bool(specialized.validation.get("dataflow_direct_path")),
            )
            replacement_ledger = build_module_capacity_ledger(
                program=self._program,
                plan=getattr(specialized, "plan", None),
                config=self.config,
                parameter_store=getattr(replacement, "parameter_store", None),
                machine=machine,
            )
            installed = False
            try:
                with self._runtime_state_lock:
                    if self._closed:
                        raise RuntimePlanError("CompiledModule is closed")
                    self._executor_generations.swap(replacement)
                    self._executor = replacement
                    self.specialized = specialized
                    self._capacity_ledger = replacement_ledger
                    installed = True
            except BaseException:
                if not installed:
                    self._close_executor_resources(replacement)
                raise

            new_plan = specialized.plan
            new_latency = float(getattr(new_plan, "predicted_latency_s", 0.0) or 0.0)
            new_placements = {p.region_id: p.device for p in getattr(new_plan, "placements", ()) or ()}
            changed = [
                {"region_id": rid, "from": old_placements.get(rid), "to": new_placements.get(rid)}
                for rid in sorted(set(old_placements) | set(new_placements))
                if old_placements.get(rid) != new_placements.get(rid)
            ]
            return {
                "plan": new_plan,
                "deltas": {
                    "predicted_latency_s_before": old_latency,
                    "predicted_latency_s_after": new_latency,
                    "predicted_latency_s_delta": new_latency - old_latency,
                    "devices_before": list(old_devices),
                    "devices_after": list(getattr(new_plan, "devices_used", ()) or ()),
                    "placement_changes": changed,
                },
            }

    def _uses_export_free_eager_root(self) -> bool:
        return bool(getattr(self._program, "is_export_free", False))

    @staticmethod
    def _bind_state_dict_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, str, bool]:
        """Bind ``state_dict`` positional/keyword args like ``nn.Module.state_dict``."""
        if len(args) > 3:
            raise TypeError(f"state_dict() takes at most 3 positional arguments but {len(args)} were given")
        names = ("destination", "prefix", "keep_vars")
        options: dict[str, Any] = {"destination": None, "prefix": "", "keep_vars": False}
        for index, value in enumerate(args):
            name = names[index]
            if name in kwargs:
                raise TypeError(f"state_dict() got multiple values for argument {name!r}")
            options[name] = value
        unknown = set(kwargs) - set(names)
        if unknown:
            name = sorted(unknown)[0]
            raise TypeError(f"state_dict() got an unexpected keyword argument {name!r}")
        options.update(kwargs)
        return options["destination"], str(options["prefix"]), bool(options["keep_vars"])

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Return real parameter tensors even when the runtime streams from disk.

        Streaming replaces module attributes with empty placeholders so the RAM
        budget stays within limits during ``forward``. Callers of ``state_dict``
        still need the true weights, so this rematerializes them from the pack
        one block at a time (a tight budget cannot pin the whole model at once).

        When oversized linears were rewritten into output-feature shards, this
        also reconstructs the original module attribute names (``layers.0.weight``)
        by concatenating shard rows, so the public state_dict matches eager.

        Export-free fused CPU keeps the caller's module unregistered (so
        ``.train()`` / ``.eval()`` cannot mutate it) and still exposes the public
        ``graph_module.<key>`` namespace.
        """
        if self._uses_export_free_eager_root():
            destination, prefix, keep_vars = self._bind_state_dict_args(args, kwargs)
            return self.graph_module.state_dict(
                destination=destination,
                prefix=f"{prefix}graph_module.",
                keep_vars=keep_vars,
            )

        executor = self._executor_generations.acquire()
        try:
            payload = torch.nn.Module.state_dict(self, *args, **kwargs)
            store = executor.parameter_store
            prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
            if getattr(store, "kind", None) == "streaming":
                for env_name, target in self._program.state_bindings.items():
                    key = f"{prefix}graph_module.{target}"
                    if key not in payload:
                        continue
                    tensor = store.acquire(env_name)
                    try:
                        payload[key] = tensor.detach().clone()
                    finally:
                        store.release((env_name,))
            return self._restore_sharded_state_dict(payload, prefix=prefix)
        finally:
            self._executor_generations.release(executor)

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        """Load weights while preserving the export-free caller-module contract.

        Public keys must use the ``graph_module.`` prefix (same as a normal
        compiled module). Bare eager keys are unexpected, not silently accepted.
        """
        if not self._uses_export_free_eager_root():
            return torch.nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)

        from collections import OrderedDict

        prefix = "graph_module."
        payload = OrderedDict()
        unexpected_public: list[str] = []
        for key, value in state_dict.items():
            name = str(key)
            if name.startswith(prefix):
                payload[name[len(prefix) :]] = value
            else:
                unexpected_public.append(name)

        if strict and unexpected_public:
            raise RuntimeError(f"Unexpected key(s) in state_dict: {', '.join(unexpected_public)}")

        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            payload._metadata = {}  # type: ignore[attr-defined]
            for key, value in metadata.items():
                name = str(key)
                if name == "graph_module":
                    name = ""
                elif name.startswith(prefix):
                    name = name[len(prefix) :]
                else:
                    continue
                payload._metadata[name] = value  # type: ignore[attr-defined]

        result = self.graph_module.load_state_dict(payload, strict=strict, assign=assign)
        if unexpected_public:
            return type(result)(list(result.missing_keys), list(result.unexpected_keys) + unexpected_public)
        return result

    def _restore_sharded_state_dict(self, payload: dict[str, Any], *, prefix: str) -> dict[str, Any]:
        """Replace linear-shard keys with reconstructed original parameter names."""
        return restore_sharded_state_dict(
            payload,
            self._program.metadata.get("linear_shards", []),
            prefix=prefix,
        )

    @staticmethod
    def _close_executor_resources(executor: Any) -> None:
        """Close one executor generation without destabilizing an installed replacement."""
        try:
            if hasattr(executor, "close"):
                executor.close()
        except Exception:  # noqa: BLE001 - retirement must not roll back a live swap
            logger.exception("failed to close retired executor generation")
        try:
            executor.parameter_store.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup after executor failure
            logger.exception("failed to close retired executor parameter store")

    def close(self) -> None:
        """Stop new work and retire every executor generation safely."""
        with self._runtime_state_lock:
            if self._closed:
                return
            self._closed = True
        self._executor_generations.close()

    def request_cancel(self) -> None:
        """Request cancellation on current and draining executor generations."""
        self._executor_generations.cancel_all()

    def __enter__(self) -> CompiledModule:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- introspection ---------------------------------------------
    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(r.region_id for r in self._program.regions)

    @property
    def program(self) -> RegionProgram:
        return self._program

    @property
    def executor(self) -> GraphExecutor:
        return self._executor

    @property
    def last_report(self) -> ExecutionReport | None:
        """Per-region timings from the most recent call, or ``None`` before the first."""
        return self._reports.get("last")

    def last_execution_report(self) -> dict[str, Any]:
        report = self._reports.get("last")
        if report is None:
            raise RuntimePlanError("No execution has run yet; call the module first")
        return report.as_dict()

    def explain(self) -> str:
        if self.config.allow_training:
            if self.training:
                exec_note = "execution: train() via schedule with autograd (call .eval() for the inference schedule)"
            else:
                exec_note = (
                    "execution: eval() via inference schedule (call .train() for schedule autograd / optimizer steps)"
                )
        else:
            exec_note = "execution: inference schedule (allow_training=False)"
        lines = [self.specialized.plan.explain(), exec_note, "regions:"]
        for region in self._program.regions:
            lines.append(
                f"  {region.region_id}: {region.node_count} ops "
                f"inputs={list(region.inputs)} outputs={list(region.outputs)} "
                f"depends_on={list(region.depends_on)}"
            )
        schedule = getattr(self.specialized, "schedule", None)
        if schedule is not None:
            spill_n = sum(
                1
                for i in schedule.instructions
                if i.opcode.value == "Evict" and i.attributes.get("kind") == "activation_spill"
            )
            reload_n = sum(
                1
                for i in schedule.instructions
                if i.opcode.value == "Load" and i.attributes.get("kind") == "activation_reload"
            )
            lines.append(
                f"executable_schedule: {len(schedule.instructions)} ops "
                f"(compute={len(schedule.compute_ops())}, "
                f"transferish={len(schedule.transfer_ops())}, "
                f"activation_spills={spill_n}, activation_reloads={reload_n})"
            )
            sim = self.specialized.profile.get("simulator") or {}
            if sim:
                lines.append(
                    f"simulator: makespan_s={sim.get('makespan_s')} "
                    f"peak_bytes={sim.get('peak_bytes')} "
                    f"critical_path_len={len(sim.get('critical_path') or [])}"
                )
        for region_meta in self.specialized.compiled_regions:
            impl = region_meta.get("impl")
            if impl:
                fb = region_meta.get("fallback_reason")
                extra = f" fallback={fb}" if region_meta.get("fallback") else ""
                lines.append(
                    f"  compiled {region_meta.get('region_id')}: impl={impl} "
                    f"compile_s={region_meta.get('compile_time_s')}{extra}"
                )
        lines.append(f"parameter_store: {self._executor.parameter_store.stats()}")
        if getattr(self._executor.parameter_store, "kind", None) == "streaming":
            lines.append(
                "note: module attributes are empty placeholders under streaming; use state_dict() for real weights"
            )
        return "\n".join(lines)

    def profile(self) -> dict[str, Any]:
        return dict(self.specialized.profile)

    def validate(self) -> dict[str, Any]:
        """Validate schedule structure and specialized-machine resource refs."""
        from tensortorrent.runtime.schedule import validate_schedule, validate_schedule_resources

        schedule = getattr(self.specialized, "schedule", None)
        result: dict[str, Any] = {
            "ok": True,
            "schedule_errors": [],
            "resource_errors": [],
            "notes": [],
        }
        if schedule is None:
            direct = getattr(self._executor, "direct_plan", None)
            if direct is not None:
                result["notes"].append("direct execution path: no executable schedule required")
                result["direct_path"] = type(direct).__name__
                return result
            result["ok"] = False
            result["schedule_errors"] = ["missing executable schedule"]
            return result
        structural = validate_schedule(schedule)
        result["schedule_errors"] = list(structural)
        machine = self._machine if self._machine is not None else discover_resource_graph()
        result["notes"].append(
            "resource check uses specialized machine"
            if self._machine is not None
            else "resource check falls back to discover_resource_graph() (no specialized machine attached)"
        )
        result["machine_fingerprint"] = getattr(machine, "fingerprint", None)
        resource_errors = validate_schedule_resources(schedule, machine)
        result["resource_errors"] = list(resource_errors)
        # Consumers of a spilled tensor must wait on some activation_reload Load.
        reload_by_tensor: dict[str, set[str]] = {}
        for inst in schedule.instructions:
            if inst.opcode.value == "Load" and inst.attributes.get("kind") == "activation_reload":
                for tensor in inst.inputs:
                    reload_by_tensor.setdefault(tensor, set()).add(inst.name)
        for tensor, reload_names in reload_by_tensor.items():
            consumers = [i for i in schedule.instructions if i.opcode.value == "Compute" and tensor in i.inputs]
            if len(consumers) < 2:
                continue
            missing = [c.name for c in consumers if not (reload_names & set(c.depends_on))]
            if missing:
                result["schedule_errors"].append(f"activation_reload for {tensor!r} missing on consumers {missing}")
        spill_ops = sum(
            1
            for i in schedule.instructions
            if i.opcode.value == "Evict" and i.attributes.get("kind") == "activation_spill"
        )
        result["activation_spill_ops"] = spill_ops
        result["instruction_count"] = len(schedule.instructions)
        result["immutable_schedule"] = type(schedule).__name__ == "ExecutableSchedule"
        if structural or resource_errors:
            result["ok"] = False
        return result

    def visualize(self, path: str, *, measured: bool = False) -> str:
        """Write a plan timeline. Default is analytic simulation.

        Pass ``measured=True`` after at least one forward to export runtime
        telemetry instead (Chrome JSON or HTML). Simulated and measured traces
        are never mixed silently.
        """
        from tensortorrent.observability import (
            write_execution_timeline_html,
            write_execution_trace,
        )

        out = Path(path)
        plan = self.specialized.plan
        if measured:
            report = self._reports.get("last")
            if report is None:
                raise RuntimePlanError("No execution has run yet; call the module before measured=True visualize")
            io_intervals: list[dict[str, Any]] | None = None
            store = self._executor.parameter_store
            if hasattr(store, "io_intervals"):
                io_intervals = [
                    {
                        "name": getattr(iv, "name", "read"),
                        "start_s": float(getattr(iv, "start_s", 0.0)),
                        "end_s": float(getattr(iv, "end_s", 0.0)),
                        "nbytes": int(getattr(iv, "nbytes", 0) or 0),
                        "cache_hit": bool(getattr(iv, "cache_hit", False)),
                        "prefetch_hit": bool(getattr(iv, "prefetch_hit", False)),
                    }
                    for iv in store.io_intervals
                ]
            residency_events: list[dict[str, Any]] = []
            transfer_events = list(getattr(self._executor, "_transfer_events", []) or [])
            schedule_report = getattr(self._executor, "_last_schedule_report", None)
            if schedule_report is not None:
                snap = getattr(schedule_report, "copy_snapshot", {}) or {}
                residency_events = [
                    {
                        "event": "copy_snapshot",
                        "name": key,
                        "nbytes": int(meta.get("nbytes", 0) or 0),
                        "tier": meta.get("tier"),
                        "ownership": meta.get("ownership"),
                        "allocation_id": meta.get("allocation_id"),
                        "resource": key.rsplit("@", 1)[-1] if "@" in key else "",
                    }
                    for key, meta in snap.items()
                    if isinstance(meta, dict)
                ]
            if out.suffix == ".json":
                write_execution_trace(
                    report,
                    out,
                    plan=plan,
                    residency_events=residency_events,
                    transfer_events=transfer_events,
                    io_intervals=io_intervals,
                )
            else:
                write_execution_timeline_html(report, out, plan=plan)
                write_execution_trace(
                    report,
                    Path(str(out).rsplit(".", 1)[0] + ".trace.json"),
                    plan=plan,
                    residency_events=residency_events,
                    transfer_events=transfer_events,
                    io_intervals=io_intervals,
                )
            return str(out)

        machine = discover_resource_graph()
        schedule = getattr(self.specialized, "schedule", None)
        if schedule is None:
            raise RuntimePlanError("No ExecutableSchedule on specialized artifact; cannot simulate without a schedule")
        sim = simulate_schedule(schedule, machine)
        if out.suffix == ".json":
            write_chrome_trace(plan, sim, out)
            return str(out)
        rows = [
            "<tr>"
            f"<td>{item.get('instruction', item.get('region', ''))}</td>"
            f"<td>{item.get('opcode', item.get('event', ''))}</td>"
            f"<td>{item.get('resource', item.get('device', ''))}</td>"
            f"<td>{item.get('start_s', 0):.6f}</td>"
            f"<td>{(item.get('end_s', 0) - item.get('start_s', 0)):.6f}</td></tr>"
            for item in sim.timeline
            if "start_s" in item and "end_s" in item
        ]
        util_rows = "".join(f"<li>{name}: {frac:.1%}</li>" for name, frac in sorted(sim.resource_utilization.items()))
        decisions = "".join(
            f"<li><b>{'SELECTED' if d.selected else 'EXCLUDED'}</b> {d.resource}: {d.reason}</li>"
            for d in plan.decisions
        )
        html = (
            "<html><body><h1>TensorTorrent plan</h1>"
            "<p><b>Timeline is analytic simulation</b> "
            f"(simulated={sim.simulated}; makespan={sim.makespan_s:.6f}s; "
            f"instructions={sim.instruction_count}; "
            f"exposed_transfer_stall_s={sim.exposed_transfer_latency_s:.6f}; "
            f"bytes_read={sim.bytes_read}; bytes_transferred={sim.bytes_transferred}). "
            "Not measured hardware validation. Accelerator paths on GPU-less VMs are simulated.</p>"
            f"<pre>{plan.explain()}</pre>"
            f"<h2>Critical path</h2><ol>"
            + "".join(f"<li>{n}</li>" for n in sim.critical_path)
            + f"</ol><h2>Resource utilization</h2><ul>{util_rows}</ul>"
            f"<h2>Resource decisions</h2><ul>{decisions}</ul>"
            "<table border=1><tr><th>instruction</th><th>opcode</th><th>resource</th>"
            "<th>start</th><th>dur</th></tr>" + "".join(rows) + "</table></body></html>"
        )
        out.write_text(html, encoding="utf-8")
        write_chrome_trace(plan, sim, Path(str(out).rsplit(".", 1)[0] + ".trace.json"))
        return str(out)

    # ---- serialization ---------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist a self-contained, atomically published compiled artifact.

        The bundle is built in a sibling staging directory, checksummed, and only
        then atomically replaces ``directory``. A failed save therefore cannot
        leave a partially updated production artifact. ``exported.pt2`` remains
        trusted executable content; integrity verification detects corruption or
        accidental modification, not malicious code.
        """
        import shutil

        from tensortorrent.artifact_io import (
            atomic_replace_directory,
            atomic_write_json,
            atomic_write_text,
            write_integrity_manifest,
        )

        out = Path(directory)
        exported = self.portable.exported
        if exported is None:
            raise RuntimePlanError("This CompiledModule was built without an ExportedProgram and cannot be saved")
        store = self._executor.parameter_store

        def _write(stage: Path) -> None:
            torch.export.save(exported, stage / "exported.pt2")
            if getattr(store, "kind", None) == "streaming":
                pack_src = Path(store.stats()["pack_path"])
                pack_dst = stage / "model.pack"
                shutil.copy2(pack_src, pack_dst)
                self.portable.packed_model_path = "model.pack"
            self.portable.save(stage)
            self.specialized.save(stage / "specialized")
            atomic_write_text(stage / "fingerprint", self.specialized.fingerprint + "\n")
            atomic_write_json(stage / "compile_config.json", self.config.to_json_dict())
            files = [path for path in stage.rglob("*") if path.is_file()]
            write_integrity_manifest(stage, files)

        return atomic_replace_directory(out, _write)

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with contextlib.suppress(Exception):
            self.close()


def load_compiled(
    directory: str | Path,
    config: CompileConfig | None = None,
    *,
    refresh_artifacts: bool = False,
    verify_integrity: bool = True,
) -> CompiledModule:
    """Reload a saved artifact and re-specialize it for the current machine.

    Treat ``directory`` as trusted code: ``exported.pt2`` is deserialized with
    ``torch.export.load``. Integrity verification is fail-closed by default;
    unsigned legacy bundles require the explicit ``verify_integrity=False`` opt-out.
    With ``refresh_artifacts`` the freshly measured plan is written back into
    ``directory``, which is what ``tensortorrent autotune`` does.
    """
    from tensortorrent.compile.pipeline import compile_exported_program

    if not isinstance(verify_integrity, bool):
        raise TypeError("verify_integrity must be a bool")
    if not isinstance(refresh_artifacts, bool):
        raise TypeError("refresh_artifacts must be a bool")
    if config is not None and not isinstance(config, CompileConfig):
        raise TypeError("config must be a CompileConfig or None")
    out = Path(directory)
    if verify_integrity:
        from tensortorrent.artifact_io import verify_integrity_manifest

        verify_integrity_manifest(out, required=True)
    exported_path = out / "exported.pt2"
    if not exported_path.exists():
        raise RuntimePlanError(f"No exported program found at {exported_path}")
    saved_config = config
    if saved_config is None:
        cfg_path = out / "compile_config.json"
        if cfg_path.exists():
            try:
                config_payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                saved_config = CompileConfig.from_json_dict(config_payload)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
                raise RuntimePlanError(f"Invalid compile config {cfg_path}: {exc}") from exc
        else:
            saved_config = CompileConfig()
    exported = torch.export.load(exported_path)
    return compile_exported_program(
        exported,
        config=saved_config,
        name=_artifact_name(out),
        artifact_dir=out if refresh_artifacts else None,
        pack_lookup_dirs=(out,),
    )


def _artifact_name(directory: Path) -> str:
    portable = directory / "portable.json"
    if portable.exists():
        return str(json.loads(portable.read_text(encoding="utf-8")).get("name", "model"))
    return "model"
