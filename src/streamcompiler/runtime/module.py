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
from streamcompiler.simulator import simulate_plan


class CompiledModule(torch.nn.Module):
    """A compiled model that behaves like any other ``torch.nn.Module``.

    Calling the module runs the planned regions on real tensors and returns the
    same structure eager PyTorch returns.

    Concurrent ``forward`` calls on the same instance are rejected: the executor
    mutates per-call scratch for the fast paths. Serialize callers or compile
    separate modules per thread.
    """

    def __init__(
        self,
        *,
        portable: PortableArtifact,
        specialized: SpecializedArtifact,
        config: CompileConfig,
        program: RegionProgram,
        executor: GraphExecutor,
    ) -> None:
        super().__init__()
        self.portable = portable
        self.specialized = specialized
        self.config = config
        self._program = program
        self._executor = executor
        # Held in a dict because nn.Module.__setattr__ is too expensive to run on
        # every forward just to record the last report.
        self._reports: dict[str, ExecutionReport] = {}
        # Registering the partitioned graph keeps parameters, buffers, `state_dict`,
        # `.to()` and `.eval()` working exactly as callers expect.
        self.graph_module = program.root

    # ---- nn.Module contract ----------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # One inference-mode guard for the call so the fast path does not pay
        # enter/exit on every tiny region invoke.
        if torch.is_inference_mode_enabled():
            return self._forward_impl(*args, **kwargs)
        with torch.inference_mode():
            return self._forward_impl(*args, **kwargs)

    def _forward_impl(self, *args: Any, **kwargs: Any) -> Any:
        flat_inputs = self._program.flatten_inputs(args, kwargs)
        flat_outputs, report = self._executor.run(flat_inputs)
        self._reports["last"] = report
        if self._program.single_output and len(flat_outputs) == 1:
            return flat_outputs[0]
        return self._program.unflatten_outputs(flat_outputs)

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
        self._executor.parameter_store.close()

    def request_cancel(self) -> None:
        """Abort an in-flight ``forward`` / ``run`` at the next region boundary."""
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
            lines.append(
                f"executable_schedule: {len(schedule.instructions)} ops "
                f"(compute={len(schedule.compute_ops())}, "
                f"transferish={len(schedule.transfer_ops())})"
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
                        "nbytes": int(getattr(iv, "nbytes", 0)),
                        "cache_hit": False,
                        "prefetch_hit": False,
                    }
                    for iv in store.io_intervals
                ]
            residency_events = self._executor.tensor_directory.drain_events()
            transfer_events = list(getattr(self._executor, "_transfer_events", []) or [])
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
        sim = simulate_plan(plan, machine)
        if out.suffix == ".json":
            write_chrome_trace(plan, sim, out)
            return str(out)
        rows = [
            "<tr>"
            f"<td>{item['region']}</td><td>{item['device']}</td><td>{item.get('backend', '')}</td>"
            f"<td>{item.get('dtype', '')}</td><td>{item['start_s']:.6f}</td>"
            f"<td>{item['end_s'] - item['start_s']:.6f}</td></tr>"
            for item in sim.timeline
            if item.get("event", "compute") == "compute" and "start_s" in item and "end_s" in item
        ]
        decisions = "".join(
            f"<li><b>{'SELECTED' if d.selected else 'EXCLUDED'}</b> {d.resource}: {d.reason}</li>"
            for d in plan.decisions
        )
        html = (
            "<html><body><h1>StreamCompiler plan</h1>"
            "<p><b>Timeline is analytic simulation</b> "
            f"(simulated={sim.simulated}; makespan={sim.makespan_s:.6f}s). "
            "Not measured hardware validation.</p>"
            f"<pre>{plan.explain()}</pre>"
            f"<h2>Resource decisions</h2><ul>{decisions}</ul>"
            "<table border=1><tr><th>region</th><th>device</th><th>backend</th>"
            "<th>dtype</th><th>start</th><th>dur</th></tr>" + "".join(rows) + "</table></body></html>"
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
