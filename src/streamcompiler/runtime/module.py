"""PyTorch-compatible compiled module wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamcompiler.compile.pipeline import PortableArtifact, SpecializedArtifact
from streamcompiler.config import CompileConfig
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.observability import write_chrome_trace
from streamcompiler.runtime.executor import Coordinator
from streamcompiler.simulator import simulate_plan


class CompiledModule:
    def __init__(
        self,
        *,
        portable: PortableArtifact,
        specialized: SpecializedArtifact,
        config: CompileConfig,
    ) -> None:
        self.portable = portable
        self.specialized = specialized
        self.config = config

    def explain(self) -> str:
        return self.specialized.plan.explain()

    def visualize(self, path: str) -> str:
        machine = discover_resource_graph()
        sim = simulate_plan(self.specialized.plan, machine)
        if path.endswith(".json"):
            write_chrome_trace(self.specialized.plan, sim, Path(path))
            return path
        plan = self.specialized.plan
        rows = []
        for item in sim.timeline:
            rows.append(
                "<tr>"
                f"<td>{item['region']}</td><td>{item['device']}</td><td>{item['backend']}</td>"
                f"<td>{item['dtype']}</td><td>{item['start_s']:.6f}</td>"
                f"<td>{item['end_s'] - item['start_s']:.6f}</td></tr>"
            )
        decisions = "".join(
            f"<li><b>{'SELECTED' if d.selected else 'EXCLUDED'}</b> {d.resource}: {d.reason}</li>"
            for d in plan.decisions
        )
        html = (
            "<html><body><h1>StreamCompiler plan</h1>"
            f"<pre>{plan.explain()}</pre>"
            f"<h2>Resource decisions</h2><ul>{decisions}</ul>"
            "<table border=1><tr><th>region</th><th>device</th><th>backend</th>"
            "<th>dtype</th><th>start</th><th>dur</th></tr>"
            + "".join(rows)
            + "</table></body></html>"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        write_chrome_trace(plan, sim, Path(path.rsplit(".", 1)[0] + ".trace.json"))
        return path

    def profile(self) -> dict[str, Any]:
        return dict(self.specialized.profile)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        machine = discover_resource_graph()
        coordinator = Coordinator(self.specialized, machine)
        return coordinator.execute()
