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
    # PyO3 otherwise probes whichever `python` happens to be on PATH. That can
    # select an unsupported system interpreter even though this check is running
    # in the project's supported virtual environment.
    os.environ["PYO3_PYTHON"] = py
    paths = [str(ROOT / "python"), str(ROOT)]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths) + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    run([py, "tools/check_version.py"])
    run([py, "-m", "ruff", "check", "python", "tests", "tools"])
    run([py, "-m", "ruff", "format", "--check", "python", "tests", "tools"])
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
    # Hardware stress tests are target-specific and can allocate most VRAM or
    # large spill files. Keep the deterministic developer gate architecture-
    # neutral; run `make hardware-test` explicitly on deployment targets.
    run([py, "-m", "pytest", "-q", "-m", "not hardware"])
    run([py, "-m", "tensortorrent.cli.main", "doctor"])
    if not (ROOT / "crates" / "tt-python" / "Cargo.toml").is_file():
        raise SystemExit("native Rust extension crate missing; refuse all_ok")
    run(
        [
            py,
            "-c",
            "from tensortorrent.native import require_native; require_native(); print('native_ok')",
        ]
    )
    print("all_ok")


if __name__ == "__main__":
    main()
