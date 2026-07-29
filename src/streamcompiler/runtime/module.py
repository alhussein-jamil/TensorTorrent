"""PyTorch-compatible compiled module."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import torch

from streamcompiler.codegen.regions import RegionProgram
from streamcompiler.compile.pipeline import PortableArtifact, SpecializedArtifact
from streamcompiler.config import CompileConfig
from streamcompiler.errors import RuntimePlanError
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.observability import write_chrome_trace
from streamcompiler.runtime.fingerprint import specialized_fingerprint_mismatch
from streamcompiler.runtime.graph_executor import ExecutionReport, GraphExecutor
from streamcompiler.simulator import simulate_schedule


class CompiledModule(torch.nn.Module):
    """A compiled model that behaves like any other ``torch.nn.Module``.

    Calling the module runs the planned regions on real tensors and returns the
    same structure eager PyTorch returns.

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
        self._machine = machine
        self._example_flat = example_flat
        # Held in a dict because nn.Module.__setattr__ is too expensive to run on
        # every forward just to record the last report.
        self._reports: dict[str, ExecutionReport] = {}
        from streamcompiler.runtime.profile_feedback import ProfileFeedback

        self._profile_feedback = ProfileFeedback()
        # Registering the partitioned graph keeps parameters, buffers, `state_dict`,
        # `.to()` and `.eval()` working exactly as callers expect.
        self.graph_module = program.root
        self._closed = False
        self._report_lock = __import__("threading").Lock()

    # ---- nn.Module contract ----------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.config.allow_training:
            return self._forward_training(*args, **kwargs)
        if torch.is_inference_mode_enabled():
            return self._forward_impl(*args, **kwargs)
        with torch.inference_mode():
            return self._forward_impl(*args, **kwargs)

    def _forward_training(self, *args: Any, **kwargs: Any) -> Any:
        """Autograd path through the partitioned ``graph_module``.

        ``torch.export`` region executables used by ``GraphExecutor`` run under
        inference-oriented wrappers; training therefore executes the live
        ``nn.Module`` tree (same partitions) so ``backward()`` can populate
        input and parameter gradients. The specialized schedule remains
        available for introspection.
        """
        # Validate shapes/dtypes the same way the executor path does.
        self._program.flatten_inputs(args, kwargs)
        result = self.graph_module(*args, **kwargs)
        if self._program.single_output:
            if isinstance(result, (tuple, list)):
                if len(result) != 1:
                    raise RuntimeError(f"Training path expected one output, got {len(result)} values from graph_module")
                return result[0]
            return result
        if isinstance(result, (tuple, list)):
            return self._program.unflatten_outputs(list(result))
        return result

    def _forward_impl(self, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimePlanError("CompiledModule is closed")
        flat_inputs = self._program.flatten_inputs(args, kwargs)
        flat_outputs, report = self._executor.run(flat_inputs)
        with self._report_lock:
            self._reports["last"] = report
        if self.config.online_profile_feedback:
            self._profile_feedback.observe_report(report)
        if self._program.single_output and len(flat_outputs) == 1:
            return flat_outputs[0]
        return self._program.unflatten_outputs(flat_outputs)

    def replan_with_profile_feedback(self) -> dict[str, Any]:
        """Re-specialize using online profile priors and swap the live executor.

        Returns ``{\"plan\": ExecutionPlan, \"deltas\": {...}}`` with latency and
        placement change summaries.
        """
        from streamcompiler.compile.pipeline import specialize_for_machine
        from streamcompiler.runtime.provisioning import (
            build_parameter_store,
            intraop_threads,
            worker_count,
        )

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
        store = build_parameter_store(self._program, self.portable, self.config)
        reuse_meta = self.portable.metadata.get("buffer_reuse") or specialized.profile.get("buffer_reuse") or {}
        reuse_assignment = dict(reuse_meta.get("assignment") or {})
        workers = worker_count(specialized, self.config)
        old = self._executor
        self._executor = GraphExecutor(
            self._program,
            specialized.bindings,
            parameter_store=store,
            max_workers=workers,
            prefetch_distance=self.config.prefetch_distance,
            intraop_threads=intraop_threads(specialized, self.config),
            activation_budget_bytes=self.config.activation_budget_bytes,
            schedule=getattr(specialized, "schedule", None),
            buffer_reuse_assignment=reuse_assignment or None,
            process_workers=int(self.config.process_workers),
        )
        if hasattr(old, "close"):
            old.close()
        self.specialized = specialized
        new_plan = specialized.plan
        new_latency = float(getattr(new_plan, "predicted_latency_s", 0.0) or 0.0)
        new_placements = {p.region_id: p.device for p in getattr(new_plan, "placements", ()) or ()}
        changed = [
            {"region_id": rid, "from": old_placements[rid], "to": new_placements[rid]}
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

    def apply_profile_feedback(self) -> Any:
        """Alias for :meth:`replan_with_profile_feedback`."""
        return self.replan_with_profile_feedback()

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Return real parameter tensors even when the runtime streams from disk.

        Streaming replaces module attributes with empty placeholders so the RAM
        budget stays honest during ``forward``. Callers of ``state_dict`` still
        need the true weights, so this rematerializes them from the pack one
        block at a time (a tight budget cannot pin the whole model at once).
        """
        payload = torch.nn.Module.state_dict(self, *args, **kwargs)
        store = self._executor.parameter_store
        if getattr(store, "kind", None) != "streaming":
            return payload
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        for env_name, target in self._program.state_bindings.items():
            key = f"{prefix}graph_module.{target}"
            if key not in payload:
                continue
            tensor = store.acquire(env_name)
            try:
                payload[key] = tensor.detach().clone()
            finally:
                store.release((env_name,))
        return payload

    def close(self) -> None:
        """Release streaming FDs / prefetch threads. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if hasattr(self._executor, "close"):
            self._executor.close()
        self._executor.parameter_store.close()

    def request_cancel(self) -> None:
        """Stop dispatching new schedule instructions; drain in-flight work, then abort."""
        self._executor.request_cancel()

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
        lines = [self.specialized.plan.explain(), "regions:"]
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
        from streamcompiler.runtime.schedule import validate_schedule, validate_schedule_resources

        schedule = getattr(self.specialized, "schedule", None)
        result: dict[str, Any] = {
            "ok": True,
            "schedule_errors": [],
            "resource_errors": [],
            "notes": [],
        }
        if schedule is None:
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
        from streamcompiler.observability import (
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
                        "version": meta.get("version"),
                        "stale": meta.get("stale"),
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
            "<html><body><h1>StreamCompiler plan</h1>"
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

    def matches_current_machine(self) -> bool:
        """False when this artifact was specialized for a different machine."""
        return not specialized_fingerprint_mismatch(self.specialized, discover_resource_graph())

    # ---- serialization ---------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist a reproducible compiled artifact.

        The directory is a trusted code bundle: ``exported.pt2`` is loaded with
        ``torch.export.load`` and can execute arbitrary captured graph code. Only
        load artifacts you produced or obtained from a trusted source.

        When the runtime streams weights from a model pack, that pack is copied
        into ``directory/model.pack`` so the bundle stays self-contained.
        """
        import shutil

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        exported = self.portable.exported
        if exported is None:
            raise RuntimePlanError("This CompiledModule was built without an ExportedProgram and cannot be saved")
        torch.export.save(exported, out / "exported.pt2")
        store = self._executor.parameter_store
        if getattr(store, "kind", None) == "streaming":
            pack_src = Path(store.stats()["pack_path"])
            pack_dst = out / "model.pack"
            if pack_src.resolve() != pack_dst.resolve():
                shutil.copy2(pack_src, pack_dst)
            self.portable.packed_model_path = "model.pack"
        self.portable.save(out)
        self.specialized.save(out / "specialized")
        (out / "fingerprint").write_text(self.specialized.fingerprint + "\n", encoding="utf-8")
        (out / "compile_config.json").write_text(
            json.dumps(self.config.to_json_dict(), indent=2),
            encoding="utf-8",
        )
        return out

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with contextlib.suppress(Exception):
            self.close()


def load_compiled(
    directory: str | Path,
    config: CompileConfig | None = None,
    *,
    refresh_artifacts: bool = False,
) -> CompiledModule:
    """Reload a saved artifact and re-specialize it for the current machine.

    Treat ``directory`` as trusted code: ``exported.pt2`` is deserialized with
    ``torch.export.load``. With ``refresh_artifacts`` the freshly measured plan is
    written back into ``directory``, which is what ``streamcompiler autotune`` does.
    """
    from streamcompiler.compile.pipeline import compile_exported_program

    out = Path(directory)
    exported_path = out / "exported.pt2"
    if not exported_path.exists():
        raise RuntimePlanError(f"No exported program found at {exported_path}")
    saved_config = config
    if saved_config is None:
        cfg_path = out / "compile_config.json"
        if cfg_path.exists():
            saved_config = CompileConfig.from_json_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
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
