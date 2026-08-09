"""Host-RAM hygiene for multi-GiB DeepMLP / HF benchmark runs."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from benchmarks.harness import TimedRun, release_host_memory
from benchmarks.workloads import DeepMLP, param_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]

_WEIGHT_PEAK_FACTOR = 3
_HOST_HEADROOM_BYTES = 4 * (1024**3)

SMOKE_PUBLIC_SUITES: tuple[str, ...] = ("fit", "budget", "hetero")
FULL_PUBLIC_SUITES: tuple[str, ...] = (
    "fit",
    "deepmlp",
    "transformer",
    "budget",
    "crossover",
    "hetero",
)

SMOKE_CROSSOVER_MULTIPLES: tuple[float, ...] = (0.15, 0.25, 0.35)
FULL_CROSSOVER_MULTIPLES: tuple[float, ...] = (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)
DEFAULT_CROSSOVER_MULTIPLES: tuple[float, ...] = (0.25, 0.6, 0.95, 1.15, 1.5)


def host_available_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


def abort_if_host_tight(need_bytes: int, *, label: str) -> TimedRun | None:
    """Fail closed when free host RAM cannot hold a safe compile peak."""
    avail = host_available_bytes()
    if avail is None:
        return None
    peak_est = int(need_bytes * _WEIGHT_PEAK_FACTOR)
    if avail < peak_est + _HOST_HEADROOM_BYTES:
        return TimedRun(
            ok=False,
            note=(
                f"skip {label}: est_peak~{peak_est / 1e9:.1f}GB "
                f"({_WEIGHT_PEAK_FACTOR}×weights) +4GiB headroom, "
                f"avail={avail / 1e9:.1f}GB (abort to avoid RAM crash)"
            ),
        )
    return None


@contextlib.contextmanager
def deepmlp_weight_file(width: int, depth: int, *, seed: int = 0):
    """Persist DeepMLP weights to a tempfile; free the live module."""
    torch.manual_seed(seed)
    ref = DeepMLP(width, depth).eval()
    pbytes = param_bytes(ref)
    fd, path = tempfile.mkstemp(prefix="tt_bench_w_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(ref.state_dict(), path)
        del ref
        release_host_memory()
        yield path, pbytes
    finally:
        with contextlib.suppress(OSError):
            Path(path).unlink(missing_ok=True)


def load_deepmlp(path: str, width: int, depth: int) -> nn.Module:
    m = DeepMLP(width, depth).eval()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m


def run_json_worker(
    module: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
) -> tuple[int, dict[str, Any] | None, str]:
    """Run ``python -m <module>`` with JSON stdin; parse last stdout JSON object."""
    proc = subprocess.run(
        [sys.executable, "-m", module],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
        cwd=str(REPO_ROOT),
    )
    err = (proc.stderr or proc.stdout or "")[-400:]
    if proc.returncode != 0:
        return proc.returncode, None, err
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return proc.returncode, None, err or "worker produced no output"
    try:
        return proc.returncode, json.loads(lines[-1]), err
    except json.JSONDecodeError:
        return proc.returncode, None, err or "worker stdout was not JSON"


def crossover_multiples(*, smoke: bool, full: bool) -> tuple[float, ...]:
    if smoke:
        return SMOKE_CROSSOVER_MULTIPLES
    if full:
        return FULL_CROSSOVER_MULTIPLES
    return DEFAULT_CROSSOVER_MULTIPLES


def public_suite_names(*, smoke: bool) -> Sequence[str]:
    return SMOKE_PUBLIC_SUITES if smoke else FULL_PUBLIC_SUITES
