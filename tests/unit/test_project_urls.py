"""Project metadata must point at the real GitHub repository."""

from __future__ import annotations

from pathlib import Path


def test_pyproject_urls_point_at_alhussein_jamil_tensortorrent() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'Homepage = "https://github.com/alhussein-jamil/TensorTorrent"' in text
    assert 'Repository = "https://github.com/alhussein-jamil/TensorTorrent"' in text
    assert "github.com/tensortorrent/tensortorrent" not in text
