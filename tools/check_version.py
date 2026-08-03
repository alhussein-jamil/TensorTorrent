"""Validate release versions across Python, Rust, and Git tags."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def _toml_value(path: Path, section: str, key: str) -> str:
    current_section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue
        if current_section != section:
            continue
        match = re.fullmatch(rf"{re.escape(key)}\s*=\s*[\"']([^\"']+)[\"']", line)
        if match:
            return match.group(1)
    raise ValueError(f"Missing {key!r} in [{section}] of {path}")


def project_versions(root: Path = ROOT) -> dict[str, str]:
    init_text = (root / "python/streamcompiler/__init__.py").read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*[\"\']([^\"\']+)[\"\']\s*$", init_text, re.MULTILINE)
    if match is None:
        raise ValueError("Missing __version__ in python/streamcompiler/__init__.py")
    return {
        "pyproject.toml": _toml_value(root / "pyproject.toml", "project", "version"),
        "Cargo.toml": _toml_value(root / "Cargo.toml", "workspace.package", "version"),
        "streamcompiler.__version__": match.group(1),
    }


def validate_version(root: Path = ROOT, *, tag: str | None = None) -> str:
    versions = project_versions(root)
    unique = set(versions.values())
    if len(unique) != 1:
        details = ", ".join(f"{source}={version}" for source, version in versions.items())
        raise ValueError(f"Version mismatch: {details}")

    version = unique.pop()
    if SEMVER.fullmatch(version) is None:
        raise ValueError(f"Project version is not valid SemVer: {version!r}")

    if tag is not None:
        expected_tag = f"v{version}"
        if tag != expected_tag:
            raise ValueError(f"Release tag {tag!r} does not match project version; expected {expected_tag!r}")
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        if re.search(rf"^## {re.escape(version)}$", changelog, re.MULTILINE) is None:
            raise ValueError(f"CHANGELOG.md has no release section for {version}")

    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="validate a release tag such as v1.2.3")
    args = parser.parse_args()
    try:
        version = validate_version(tag=args.tag)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"version_ok={version}")


if __name__ == "__main__":
    main()
