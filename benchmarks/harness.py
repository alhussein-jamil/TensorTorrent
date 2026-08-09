"""Shared timing, memory, and environment helpers for public benchmarks."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


@dataclass
class TimedRun:
    ok: bool
    median_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0
    stdev_ms: float = 0.0
    samples_ms: list[float] = field(default_factory=list)
    peak_device_bytes: int = 0
    peak_host_bytes: int = 0
    compile_s: float = 0.0
    note: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peaks() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()


def release_host_memory() -> None:
    """Drop Python + CUDA caches between heavy benchmark steps."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
    gc.collect()


def peak_device_bytes() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def peak_host_bytes() -> int:
    # Linux: ru_maxrss is KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def timed_callable(fn: Callable[[], Any], *, iters: int, warmup: int) -> list[float]:
    for _ in range(max(0, warmup)):
        fn()
    sync()
    samples: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def summarize_samples(
    samples: list[float],
    *,
    ok: bool = True,
    note: str = "",
    extras: dict[str, Any] | None = None,
    **more: Any,
) -> TimedRun:
    if not samples:
        payload = dict(extras or {})
        payload.update(more)
        return TimedRun(ok=ok, note=note or "no samples", extras=payload)
    ordered = sorted(samples)
    p95_idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    p25_idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.25) - 1))
    p75_idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.75) - 1))
    payload = dict(extras or {})
    payload.update(more)
    payload["p25_ms"] = float(ordered[p25_idx])
    payload["p75_ms"] = float(ordered[p75_idx])
    if torch.cuda.is_available():
        payload["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    return TimedRun(
        ok=ok,
        median_ms=float(statistics.median(samples)),
        p95_ms=float(ordered[p95_idx]),
        mean_ms=float(statistics.fmean(samples)),
        stdev_ms=float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        samples_ms=[float(s) for s in samples],
        peak_device_bytes=peak_device_bytes(),
        peak_host_bytes=peak_host_bytes(),
        note=note,
        extras=payload,
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        env["host_ram_total_bytes"] = int(vm.total)
        env["host_ram_available_bytes"] = int(vm.available)
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu0"] = {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "major": props.major,
            "minor": props.minor,
        }
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if driver:
                env["cuda_driver_version"] = driver.splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        try:
            import tensortorrent as tt

            env["tensortorrent"] = getattr(tt, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            env["tensortorrent"] = "unavailable"
    return env


def results_dir(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parent / "results"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, TimedRun):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def evidence_class(kind: str) -> str:
    """Normalize evidence labels used in docs and JSON."""
    allowed = {"MEASURED", "SIMULATED", "SUPPORTED_BUT_UNMEASURED", "PLANNED"}
    key = kind.strip().upper().replace(" ", "_")
    if key not in allowed:
        raise ValueError(f"unknown evidence class {kind!r}; expected one of {sorted(allowed)}")
    return key


def to_plain(obj: Any) -> Any:
    """Recursively convert TimedRun (and nested containers) to plain dicts."""
    if isinstance(obj, TimedRun):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def write_suite_json(out_dir: Path, payload: Any, *names: str) -> None:
    """Write the same suite payload under one or more filenames (compat aliases)."""
    plain = to_plain(payload)
    for name in names:
        write_json(out_dir / name, plain)
