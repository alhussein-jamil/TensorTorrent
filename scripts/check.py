"""Developer Makefile-style helpers as a plain script (no Make required)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    py = sys.executable
    os.environ["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    run([py, "-m", "ruff", "check", "src", "tests"])
    run([py, "-m", "ruff", "format", "--check", "src", "tests"])
    run([py, "-m", "mypy", "src"])
    run([py, "-m", "pytest", "-q"])
    run([py, "-m", "streamcompiler.cli.main", "doctor"])
    if (ROOT / "native").is_dir() or (ROOT / "CMakeLists.txt").is_file():
        raise SystemExit("native sources present without a CI build path; refuse all_ok")
    print("all_ok")


if __name__ == "__main__":
    main()
