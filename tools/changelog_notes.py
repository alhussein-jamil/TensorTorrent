"""Extract a CHANGELOG.md section for GitHub Release notes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTION = re.compile(r"^## (?P<title>.+)\s*$")


def changelog_section(version: str, *, root: Path = ROOT) -> str:
    changelog = root / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8")
    lines = text.splitlines()
    start: int | None = None
    end = len(lines)
    for idx, line in enumerate(lines):
        match = SECTION.match(line)
        if match is None:
            continue
        title = match.group("title").strip()
        if start is None and title == version:
            start = idx
            continue
        if start is not None:
            end = idx
            break
    if start is None:
        raise ValueError(f"CHANGELOG.md has no release section for {version}")
    body = "\n".join(lines[start:end]).strip()
    if not body:
        raise ValueError(f"CHANGELOG.md section for {version} is empty")
    return body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="release tag such as v1.2.3")
    parser.add_argument(
        "--repo",
        default="alhussein-jamil/TensorTorrent",
        help="GitHub owner/repo for the compare link",
    )
    args = parser.parse_args()
    tag = args.tag
    if not tag.startswith("v"):
        raise SystemExit(f"tag must start with 'v', got {tag!r}")
    version = tag[1:]
    try:
        notes = changelog_section(version)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(notes.rstrip())
    print()
    print(f"Full changelog: https://github.com/{args.repo}/blob/main/CHANGELOG.md")


if __name__ == "__main__":
    main()
