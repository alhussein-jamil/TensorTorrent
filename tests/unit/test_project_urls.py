"""Project metadata must point at the real GitHub repository."""

from __future__ import annotations

from pathlib import Path


def test_pyproject_urls_point_at_alhussein_jamil_streamcompiler() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'Homepage = "https://github.com/alhussein-jamil/streamcompiler"' in text
    assert 'Repository = "https://github.com/alhussein-jamil/streamcompiler"' in text
    assert "github.com/streamcompiler/streamcompiler" not in text
