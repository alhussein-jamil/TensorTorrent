"""Developer Makefile-style helpers as a plain script (no Make required)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    py = sys.executable
    paths = [str(ROOT / "python"), str(ROOT)]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths) + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    run([py, "-m", "ruff", "check", "python", "tests", "server"])
    run([py, "-m", "ruff", "format", "--check", "python", "tests", "server"])
    run([py, "-m", "mypy", "python"])
    if shutil.which("cargo"):
        run(["cargo", "fmt", "--check"])
        run(
            [
                "cargo",
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ]
        )
        run(["cargo", "test", "--workspace"])
    run([py, "-m", "pytest", "-q"])
    run([py, "-m", "streamcompiler.cli.main", "doctor"])
    if not (ROOT / "rust" / "sc-python" / "Cargo.toml").is_file():
        raise SystemExit("native Rust extension crate missing; refuse all_ok")
    run(
        [
            py,
            "-c",
            "from streamcompiler.native import require_native; require_native(); print('native_ok')",
        ]
    )
    print("all_ok")


if __name__ == "__main__":
    main()
