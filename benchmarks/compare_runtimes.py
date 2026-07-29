#!/usr/bin/env python3
"""Compare eager vs native StreamCompiler vs legacy Python DAG (opt-in).

CPU-only. Writes machine-readable JSON. Never labels simulated accelerator work
as measured.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.native import require_native


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _median_p90(times: list[float]) -> tuple[float, float]:
    s = sorted(times)
    med = statistics.median(s)
    p90 = s[int(0.90 * (len(s) - 1))]
    return med, p90


def _bench_call(fn, x: torch.Tensor, *, warmup: int = 10, iters: int = 100) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            fn(x)
        times: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(x)
            times.append(time.perf_counter() - t0)
    med, p90 = _median_p90(times)
    return {
        "median_s": med,
        "p90_s": p90,
        "throughput_fps": 1.0 / med if med > 0 else 0.0,
        "iters": iters,
        "warmup": warmup,
        "profile_status": "measured",
    }


def main() -> None:
    require_native()
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64)).eval()
    x = torch.randn(32, 256)

    # Eager
    eager = _bench_call(model, x)

    # Legacy Python DAG (developer oracle path — not production)
    from streamcompiler.testing.legacy_runtime import run_schedule_legacy_python

    legacy_mod = sc.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        # Warm schedule install
        ex = legacy_mod.executor._schedule_executor

        def _legacy_once(inp: torch.Tensor) -> None:
            run_schedule_legacy_python(ex, list(legacy_mod._program.flatten_inputs((inp,), {})))

        legacy = _bench_call(_legacy_once, x)
        legacy["path"] = "python_dag_oracle"
        legacy["profile_status"] = "measured"
    finally:
        legacy_mod.close()

    # Native (only production path)
    native_mod = sc.compile(
        model,
        (x,),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        native = _bench_call(native_mod, x)
        from streamcompiler.native import require_native as rn

        rn().reset_debug_counters()
        with torch.inference_mode():
            native_mod(x)
        counters = dict(rn().debug_counters())
        report = native_mod.last_report
        store = getattr(report, "parameter_store", {}) or {}
        native["debug_counters"] = counters
        native["native_runtime"] = store.get("native_runtime")
        native["native_data_plane"] = store.get("native_data_plane")
        native["compute_callbacks_last_forward"] = counters.get("compute_callbacks")
        native["non_compute_python_callbacks_last_forward"] = counters.get(
            "non_compute_python_callbacks"
        )
        native["python_callbacks_last_forward"] = counters.get("instruction_callbacks")
        native["gil_acquisitions_last_forward"] = counters.get("gil_acquisitions")
    finally:
        native_mod.close()

    payload = {
        "date": datetime.now(timezone.utc).isoformat(),
        "commit": _git_hash(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "rustc": subprocess.check_output(["rustc", "--version"]).decode().strip(),
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "build_mode": "maturin-dev",
        "model": "Sequential(Linear 256→256, ReLU, Linear 256→64)",
        "batch": 32,
        "results": {
            "eager_pytorch": eager,
            "streamcompiler_legacy_python_dag": legacy,
            "streamcompiler_native": native,
        },
        "notes": [
            "CPU-only VM; no CUDA/ROCm claimed.",
            "Legacy Python DAG is oracle/bench-only via testing.legacy_runtime — never auto-activates.",
            "Resident path: non_compute_python_callbacks=0; Load=persistent_residency.",
        ],
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"native_forward_{payload['commit'][:12]}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["results"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
