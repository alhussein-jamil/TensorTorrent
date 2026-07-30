"""CLI behaviour tests: the commands documented in docs/deployment.md must run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.cli.main import main


def _saved_artifact(tmp_path: Path) -> Path:
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
    out = tmp_path / "artifact"
    compiled = sc.compile(
        model,
        (torch.randn(4, 32),),
        artifact_dir=out,
        config=sc.CompileConfig(allow_gpu=False),
    )
    compiled.save(out)
    return out


def test_doctor_reports_the_compiled_path(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    report = tmp_path / "doctor.json"
    assert main(["doctor", "--json", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    names = {check["name"]: check["status"] for check in payload["checks"]}
    assert names["numerical_equivalence_eager"] == "numerical_correctness_validated"
    if torch.cuda.is_available():
        assert names["backend_available:cuda"] == "backend_available"
    else:
        assert names["backend_available:cuda"] == "unsupported_capability"


def test_autotune_measures_regions_when_the_exported_program_is_present(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Autotuning a saved artifact must benchmark, not fall back to priors."""
    out = _saved_artifact(tmp_path)
    assert main(["autotune", "--cpu-only", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "region_costs=measured" in printed
    assert "priors_only" not in printed
    assert (out / "specialized" / "specialized.json").exists()


def test_autotune_without_an_exported_program_says_it_is_planning_from_priors(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    out = _saved_artifact(tmp_path)
    (out / "exported.pt2").unlink()
    assert main(["autotune", "--cpu-only", str(out)]) == 0
    captured = capsys.readouterr()
    assert "planning from priors only" in captured.err
    assert "priors_only" in captured.out


def test_autotune_rejects_a_directory_without_a_portable_artifact(tmp_path: Path) -> None:
    assert main(["autotune", str(tmp_path)]) == 2


def test_benchmark_topology_emits_link_rows(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out = tmp_path / "topo.json"
    assert main(["benchmark-topology", "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["transfer_matrix"]
    # Unmeasured links must say so rather than reporting a fabricated bandwidth.
    for link in payload["transfer_matrix"]:
        assert link["measured"] is True or link["bytes_per_s"] is None
