from __future__ import annotations

from pathlib import Path

import pytest
from tools.check_version import ROOT, project_versions, validate_version


def _write_project(root: Path, *, python: str = "1.2.3", rust: str = "1.2.3", package: str = "1.2.3") -> None:
    (root / "python/streamcompiler").mkdir(parents=True)
    (root / "python/streamcompiler/__init__.py").write_text(f'__version__ = "{python}"\n', encoding="utf-8")
    (root / "pyproject.toml").write_text(f'[project]\nversion = "{package}"\n', encoding="utf-8")
    (root / "Cargo.toml").write_text(f'[workspace.package]\nversion = "{rust}"\n', encoding="utf-8")
    (root / "CHANGELOG.md").write_text(f"# Changelog\n\n## {package}\n", encoding="utf-8")


def test_repository_versions_match() -> None:
    assert validate_version(ROOT) == project_versions(ROOT)["pyproject.toml"]


def test_version_mismatch_is_rejected(tmp_path: Path) -> None:
    _write_project(tmp_path, rust="1.2.4")

    with pytest.raises(ValueError, match="Version mismatch"):
        validate_version(tmp_path)


def test_release_tag_and_changelog_are_validated(tmp_path: Path) -> None:
    _write_project(tmp_path)

    assert validate_version(tmp_path, tag="v1.2.3") == "1.2.3"
    with pytest.raises(ValueError, match="expected 'v1.2.3'"):
        validate_version(tmp_path, tag="v1.2.4")
