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
from streamcompiler.runtime.executor import Coordinator
from streamcompiler.runtime.graph_executor import ExecutionReport, GraphExecutor
from streamcompiler.simulator import simulate_plan


class CompiledModule(torch.nn.Module):
    """A compiled model that behaves like any other ``torch.nn.Module``.

    Calling the module runs the planned regions on real tensors and returns the
    same structure eager PyTorch returns.
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
        self._last_report: ExecutionReport | None = None
        # Registering the partitioned graph keeps parameters, buffers, `state_dict`,
        # `.to()` and `.eval()` working exactly as callers expect.
        self.graph_module = program.root

    # ---- nn.Module contract ----------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        flat_inputs = self._program.flatten_inputs(args, kwargs)
        flat_outputs, report = self._executor.run(flat_inputs)
        self._last_report = report
        return self._program.unflatten_outputs(flat_outputs)

    # ---- introspection ---------------------------------------------
    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(r.region_id for r in self._program.regions)

    @property
    def program(self) -> RegionProgram:
        return self._program

    def last_execution_report(self) -> dict[str, Any]:
        if self._last_report is None:
            raise RuntimePlanError("No execution has run yet; call the module first")
        return self._last_report.as_dict()

    def explain(self) -> str:
        lines = [self.specialized.plan.explain(), "regions:"]
        for region in self._program.regions:
            lines.append(
                f"  {region.region_id}: {region.node_count} ops "
                f"inputs={list(region.inputs)} outputs={list(region.outputs)} "
                f"depends_on={list(region.depends_on)}"
            )
        lines.append(f"parameter_store: {self._executor.parameter_store.stats()}")
        return "\n".join(lines)

    def profile(self) -> dict[str, Any]:
        return dict(self.specialized.profile)

    def visualize(self, path: str) -> str:
        machine = discover_resource_graph()
        sim = simulate_plan(self.specialized.plan, machine)
        if path.endswith(".json"):
            write_chrome_trace(self.specialized.plan, sim, Path(path))
            return path
        plan = self.specialized.plan
        rows = [
            "<tr>"
            f"<td>{item['region']}</td><td>{item['device']}</td><td>{item['backend']}</td>"
            f"<td>{item['dtype']}</td><td>{item['start_s']:.6f}</td>"
            f"<td>{item['end_s'] - item['start_s']:.6f}</td></tr>"
            for item in sim.timeline
        ]
        decisions = "".join(
            f"<li><b>{'SELECTED' if d.selected else 'EXCLUDED'}</b> {d.resource}: {d.reason}</li>"
            for d in plan.decisions
        )
        html = (
            "<html><body><h1>StreamCompiler plan</h1>"
            f"<pre>{plan.explain()}</pre>"
            f"<h2>Resource decisions</h2><ul>{decisions}</ul>"
            "<table border=1><tr><th>region</th><th>device</th><th>backend</th>"
            "<th>dtype</th><th>start</th><th>dur</th></tr>" + "".join(rows) + "</table></body></html>"
        )
        Path(path).write_text(html, encoding="utf-8")
        write_chrome_trace(plan, sim, Path(path.rsplit(".", 1)[0] + ".trace.json"))
        return path

    def coordinator(self) -> Coordinator:
        """Whole-machine coordinator used by validation and telemetry tooling."""
        return Coordinator(self.specialized, discover_resource_graph())

    # ---- serialization ---------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist a reproducible compiled artifact."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        exported = self.portable.exported
        if exported is None:
            raise RuntimePlanError("This CompiledModule was built without an ExportedProgram and cannot be saved")
        torch.export.save(exported, out / "exported.pt2")
        self.portable.save(out)
        self.specialized.save(out / "specialized")
        (out / "compile_config.json").write_text(
            json.dumps(
                {
                    "objective": self.config.objective.value,
                    "max_region_nodes": self.config.max_region_nodes,
                    "profile_level": self.config.profile_level,
                    "ram_budget_bytes": self.config.ram_budget_bytes,
                    "prefetch_distance": self.config.prefetch_distance,
                    "allow_concurrent_regions": self.config.allow_concurrent_regions,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        executor = getattr(self, "_executor", None)
        if executor is not None:
            with contextlib.suppress(Exception):
                executor.parameter_store.close()


def load_compiled(directory: str | Path, config: CompileConfig | None = None) -> CompiledModule:
    """Reload a saved artifact and re-specialize it for the current machine."""
    from streamcompiler.compile.pipeline import compile_exported_program

    out = Path(directory)
    exported_path = out / "exported.pt2"
    if not exported_path.exists():
        raise RuntimePlanError(f"No exported program found at {exported_path}")
    saved_config = config
    if saved_config is None:
        cfg_path = out / "compile_config.json"
        saved_config = CompileConfig()
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            saved_config.max_region_nodes = int(data.get("max_region_nodes", 16))
            saved_config.prefetch_distance = int(data.get("prefetch_distance", 1))
            saved_config.ram_budget_bytes = data.get("ram_budget_bytes")
            saved_config.allow_concurrent_regions = bool(data.get("allow_concurrent_regions", True))
    exported = torch.export.load(exported_path)
    return compile_exported_program(exported, config=saved_config, name=_artifact_name(out))


def _artifact_name(directory: Path) -> str:
    portable = directory / "portable.json"
    if portable.exists():
        return str(json.loads(portable.read_text(encoding="utf-8")).get("name", "model"))
    return "model"
