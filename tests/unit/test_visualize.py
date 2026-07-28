"""CompiledModule.visualize must tolerate mixed timeline event kinds."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import streamcompiler as sc


def test_visualize_html_and_chrome_json(tmp_path: Path) -> None:
    model = nn.Linear(4, 2).eval()
    x = torch.randn(1, 4)
    compiled = sc.compile(model, (x,))
    try:
        html = tmp_path / "plan.html"
        trace = tmp_path / "plan.json"
        compiled.visualize(str(html))
        compiled.visualize(str(trace))
        assert "StreamCompiler plan" in html.read_text(encoding="utf-8")
        assert "analytic simulation" in html.read_text(encoding="utf-8")
        text = trace.read_text(encoding="utf-8")
        assert '"simulated"' in text
        assert "traceEvents" in text
    finally:
        compiled.close()
