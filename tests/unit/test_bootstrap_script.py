"""Bootstrap --check-only must classify this host without mutating the tree."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_bootstrap_check_only_succeeds_on_supported_unix() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/bootstrap.py", "--check-only"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "platform:" in proc.stdout
    assert "unsupported" not in proc.stdout.split("platform:", 1)[1].splitlines()[0]
