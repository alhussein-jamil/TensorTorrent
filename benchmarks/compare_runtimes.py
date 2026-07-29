#!/usr/bin/env python3
"""Compare eager PyTorch vs native StreamCompiler vs legacy Python DAG.

CPU-only. Writes machine-readable JSON. Never labels simulated accelerator work
as measured.

Primary workload: streaming multi-layer Linear stack under a RAM budget — the
path where Prefetch overlap must keep residency under budget. A tiny resident
microbench is recorded as secondary context only.
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
from streamcompiler.cost_model import prediction_error
from streamcompiler.native import require_native
from streamcompiler.testing.legacy_runtime import run_schedule_legacy_python


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
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        untracked = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
        ).decode()
        if dirty != 0 or untracked.strip():
            return f"{commit}-dirty"
        return commit
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


def _bench_legacy(compiled: sc.CompiledModule, flat: list[torch.Tensor], *, warmup: int = 5, iters: int = 30) -> dict:
    se = compiled.executor._schedule_executor
    if se is None:
        raise RuntimeError("ScheduleExecutor missing for legacy bench")

    def _run(_x: torch.Tensor) -> None:
        run_schedule_legacy_python(se, list(flat))

    return _bench_call(_run, flat[0], warmup=warmup, iters=iters)


def _bench_pair(model: nn.Module, x: torch.Tensor, config: CompileConfig) -> tuple[dict, dict, dict]:
    eager = _bench_call(model, x)

    # Separate compiles so legacy and native do not share store/pin state.
    legacy_mod = sc.compile(model, (x,), config=config)
    try:
        flat = list(legacy_mod._program.flatten_inputs((x,), {}))
        legacy = _bench_legacy(legacy_mod, flat)
        legacy["path"] = "python_dag_oracle"
    finally:
        legacy_mod.close()

    native_mod = sc.compile(model, (x,), config=config)
    closed = False
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
        native["non_compute_python_callbacks_last_forward"] = counters.get("non_compute_python_callbacks")
        native["python_callbacks_last_forward"] = counters.get("instruction_callbacks")
        native["gil_acquisitions_last_forward"] = counters.get("gil_acquisitions")
        native["parameter_load_callbacks_last_forward"] = counters.get("parameter_load_callbacks")
        native["handle_release_callbacks_last_forward"] = counters.get("handle_release_callbacks")
        native["copy_sync_callbacks_last_forward"] = counters.get("copy_sync_callbacks")
        native["peak_resident_bytes"] = store.get("peak_resident_bytes")
        native["handle_live_bytes"] = store.get("handle_live_bytes")
        native["bytes_read"] = store.get("bytes_read") or store.get("native_bytes_read")
        native["cache_hits"] = store.get("cache_hits")
        native["schedule_ops"] = len(native_mod.executor._schedule_executor.schedule.instructions)

        # Shutdown must not hang or leak live handles.
        native_mod.close()
        native["shutdown_ok"] = True
        closed = True
        if not native.get("native_runtime") or not native.get("native_data_plane"):
            raise SystemExit(f"native path not engaged: {store}")
    finally:
        if not closed:
            native_mod.close()
    return eager, native, legacy


def main() -> None:
    require_native()

    # Primary: streaming under RAM budget.
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
    eager_s, native_s, legacy_s = _bench_pair(stream_model, stream_x, stream_cfg)
    peak = int(native_s.get("peak_resident_bytes") or 0)
    if peak > budget:
        raise SystemExit(f"streaming residency exceeded budget: peak={peak} budget={budget}")
    if native_s["median_s"] >= legacy_s["median_s"]:
        raise SystemExit(
            f"native did not beat legacy on streaming primary: "
            f"native={native_s['median_s'] * 1e3:.3f}ms legacy={legacy_s['median_s'] * 1e3:.3f}ms"
        )

    # Secondary: tiny resident graph (overhead-dominated; recorded for trend only).
    micro = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64)).eval()
    micro_x = torch.randn(32, 256)
    micro_cfg = CompileConfig(use_torch_compile=False, measure_regions=False)
    eager_m, native_m, legacy_m = _bench_pair(micro, micro_x, micro_cfg)

    # Simulator parity smoke on the streaming schedule (native DES).
    sim_error = None
    sim_makespan = None
    pred_errs: dict = {"prediction_error_s": None, "prediction_relative_error": None}
    try:
        from streamcompiler.hardware.discovery import discover_resource_graph
        from streamcompiler.simulator import simulate_schedule

        tmp = sc.compile(stream_model, (stream_x,), config=stream_cfg)
        try:
            sched = tmp.specialized.schedule
            machine = discover_resource_graph()
            sim = simulate_schedule(sched, machine)
            if getattr(sim, "error", None):
                sim_error = str(sim.error)
            elif getattr(sim, "feasible", True) is False:
                sim_error = "infeasible"
            else:
                sim_error = None
            from streamcompiler.cost_model.calibration import runtime_predicted_makespan_s
            from streamcompiler.ir.graph import OpCode

            analytic = float(getattr(sim, "makespan_s", 0.0) or 0.0)
            n_compute = sum(1 for i in sched.instructions if i.opcode == OpCode.COMPUTE)
            sim_makespan = runtime_predicted_makespan_s(analytic, n_compute=n_compute)
            pred_errs = prediction_error(native_s["median_s"], sim_makespan)
        finally:
            tmp.close()
    except Exception as exc:  # noqa: BLE001
        sim_error = f"{type(exc).__name__}: {exc}"
        sim_makespan = None

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
                "streamcompiler_native": native_s,
                "streamcompiler_legacy_python_dag": legacy_s,
            },
            "native_under_budget": True,
            "native_beats_legacy": True,
            "speedup_vs_eager": eager_s["median_s"] / native_s["median_s"],
            "speedup_vs_legacy": legacy_s["median_s"] / native_s["median_s"],
            "simulator_error": sim_error,
            "simulator_makespan_s": sim_makespan,
            "prediction_error_s": pred_errs["prediction_error_s"],
            "prediction_relative_error": pred_errs["prediction_relative_error"],
        },
        "secondary_resident_microbench": {
            "model": "Sequential(Linear 256→256, ReLU, Linear 256→64)",
            "batch": 32,
            "results": {
                "eager_pytorch": eager_m,
                "streamcompiler_native": native_m,
                "streamcompiler_legacy_python_dag": legacy_m,
            },
        },
        "notes": [
            "CPU-only VM; no CUDA/ROCm claimed.",
            "Primary proof: streaming Prefetch/Load stays under RAM budget on the native path.",
            "Legacy Python DAG is bench-only via streamcompiler.testing.legacy_runtime.",
            "prediction_error_s = wall_median - sim_makespan when the simulator runs.",
            "Resident microbench is overhead-dominated (fused 1-op schedule).",
            "handle_release / copy_sync callbacks are batched (one GIL per wave/instruction).",
        ],
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"native_forward_{payload['commit'][:12]}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["primary"], indent=2))
    print("wrote", out)
    print(
        f"primary vs eager: {payload['primary']['speedup_vs_eager']:.2f}x "
        f"(native {native_s['median_s'] * 1e3:.2f}ms / eager {eager_s['median_s'] * 1e3:.2f}ms) "
        f"vs legacy {payload['primary']['speedup_vs_legacy']:.2f}x "
        f"peak_resident={peak} budget={budget}"
    )


if __name__ == "__main__":
    main()
