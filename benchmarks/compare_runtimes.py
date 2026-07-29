#!/usr/bin/env python3
"""Compare eager vs native StreamCompiler vs legacy Python DAG (opt-in).

CPU-only. Writes machine-readable JSON. Never labels simulated accelerator work
as measured.

Primary workload: streaming multi-layer Linear stack under a RAM budget — the
path where the native dispatcher + Prefetch overlap must beat the legacy Python
DAG. A tiny resident microbench is recorded as secondary context only.
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


class _Deep(nn.Module):
    def __init__(self, width: int = 64, layers: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


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


def _bench_call(fn, x: torch.Tensor, *, warmup: int = 10, iters: int = 60) -> dict:
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


def _bench_pair(model: nn.Module, x: torch.Tensor, config: CompileConfig) -> tuple[dict, dict, dict]:
    from streamcompiler.testing.legacy_runtime import run_schedule_legacy_python

    eager = _bench_call(model, x)

    legacy_mod = sc.compile(model, (x,), config=config)
    try:
        ex = legacy_mod.executor._schedule_executor

        def _legacy_once(inp: torch.Tensor) -> None:
            run_schedule_legacy_python(ex, list(legacy_mod._program.flatten_inputs((inp,), {})))

        legacy = _bench_call(_legacy_once, x)
        legacy["path"] = "python_dag_oracle"
    finally:
        legacy_mod.close()

    native_mod = sc.compile(model, (x,), config=config)
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
        native["parameter_load_callbacks_last_forward"] = counters.get("parameter_load_callbacks")
        native["handle_release_callbacks_last_forward"] = counters.get("handle_release_callbacks")
        native["peak_resident_bytes"] = store.get("peak_resident_bytes")
        native["handle_live_bytes"] = store.get("handle_live_bytes")
        native["schedule_ops"] = len(native_mod.executor._schedule_executor.schedule.instructions)
    finally:
        native_mod.close()
    return eager, legacy, native


def main() -> None:
    require_native()

    # Primary: streaming under RAM budget (native Prefetch overlap must win).
    stream_model = _Deep(width=64, layers=8).eval()
    stream_x = torch.randn(16, 64)
    total = sum(p.numel() * p.element_size() for p in stream_model.parameters())
    budget = max(total // 4, 64 * 64 * 4 * 2)
    stream_cfg = CompileConfig(
        use_torch_compile=False,
        measure_regions=False,
        max_region_nodes=1,
        ram_budget_bytes=budget,
        prefetch_distance=1,
    )
    eager_s, legacy_s, native_s = _bench_pair(stream_model, stream_x, stream_cfg)
    if native_s["median_s"] >= legacy_s["median_s"]:
        raise SystemExit(
            f"native must beat legacy on streaming bench: "
            f"native={native_s['median_s']:.6f} legacy={legacy_s['median_s']:.6f}"
        )

    # Secondary: tiny resident graph (overhead-dominated; recorded for trend only).
    micro = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64)).eval()
    micro_x = torch.randn(32, 256)
    micro_cfg = CompileConfig(use_torch_compile=False, measure_regions=False)
    eager_m, legacy_m, native_m = _bench_pair(micro, micro_x, micro_cfg)

    payload = {
        "date": datetime.now(timezone.utc).isoformat(),
        "commit": _git_hash(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "rustc": subprocess.check_output(["rustc", "--version"]).decode().strip(),
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "build_mode": "maturin-dev",
        "primary": {
            "model": "Deep(Linear 64×8 + ReLU) streaming",
            "batch": 16,
            "ram_budget_bytes": budget,
            "results": {
                "eager_pytorch": eager_s,
                "streamcompiler_legacy_python_dag": legacy_s,
                "streamcompiler_native": native_s,
            },
            "native_beats_legacy": True,
            "speedup_vs_legacy": legacy_s["median_s"] / native_s["median_s"],
        },
        "secondary_resident_microbench": {
            "model": "Sequential(Linear 256→256, ReLU, Linear 256→64)",
            "batch": 32,
            "results": {
                "eager_pytorch": eager_m,
                "streamcompiler_legacy_python_dag": legacy_m,
                "streamcompiler_native": native_m,
            },
        },
        "notes": [
            "CPU-only VM; no CUDA/ROCm claimed.",
            "Legacy Python DAG is oracle/bench-only via testing.legacy_runtime — never auto-activates.",
            "Primary proof: streaming Prefetch/Load under RAM budget; native must beat legacy.",
            "Resident microbench is overhead-dominated (fused 1-op schedule).",
        ],
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"native_forward_{payload['commit'][:12]}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["primary"], indent=2))
    print("wrote", out)
    print(
        f"primary speedup vs legacy: {payload['primary']['speedup_vs_legacy']:.2f}x "
        f"(native {native_s['median_s']*1e3:.2f}ms / legacy {legacy_s['median_s']*1e3:.2f}ms)"
    )


if __name__ == "__main__":
    main()
