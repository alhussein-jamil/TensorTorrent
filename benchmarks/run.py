#!/usr/bin/env python3
"""Reproduce TensorTorrent public benchmarks.

Examples::

    python -m benchmarks.run --smoke
    python -m benchmarks.run --suite beyond_vram
    python -m benchmarks.run --suite all --iters 20
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.harness import TimedRun, collect_environment, results_dir, write_json
from benchmarks.suites import (
    run_beyond_vram_suite,
    run_fit_suite,
    run_hetero_suite,
    run_memory_pressure_suite,
    run_model_size_scaling_suite,
    try_plot,
)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, TimedRun):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def _print_fit(payload: dict[str, Any]) -> None:
    print(f"\n=== fit ({payload.get('device')}) ===")
    print("| workload | eager ms | compile ms | TT ms | TT/eager | peak VRAM MB |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in payload.get("results", []):
        approaches = row["approaches"]

        def med(name: str, apps: dict = approaches) -> str:
            r = apps.get(name)
            if not r or not r.get("ok"):
                return "FAIL"
            return f"{r['median_ms']:.2f}"

        eager = approaches.get("eager") or {}
        tt = approaches.get("tensortorrent") or {}
        rel = ""
        if eager.get("ok") and tt.get("ok") and eager.get("median_ms"):
            rel = f"{tt['median_ms'] / eager['median_ms']:.2f}×"
        peak = tt.get("peak_device_bytes", 0) / 1e6 if tt.get("ok") else 0
        print(
            f"| {row['workload']} | {med('eager')} | {med('torch_compile')} | {med('tensortorrent')} | {rel} | {peak:.1f} |"
        )


def _print_beyond(payload: dict[str, Any]) -> None:
    print("\n=== beyond_vram ===")
    print(f"params={payload.get('params_bytes', 0) / 1e9:.2f}G ({payload.get('params_over_vram', 0):.2f}× VRAM)")
    print("| approach | median ms | peak VRAM GB | peak host GB | status |")
    print("|---|---:|---:|---:|---|")
    for name, run in (payload.get("approaches") or {}).items():
        if run.get("ok"):
            print(
                f"| {name} | {run['median_ms']:.1f} | {run['peak_device_bytes'] / 1e9:.2f} | "
                f"{run['peak_host_bytes'] / 1e9:.2f} | ok |"
            )
        else:
            print(f"| {name} | — | — | — | {run.get('note', 'FAIL')[:80]} |")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite",
        choices=("smoke", "fit", "beyond_vram", "pressure", "scaling", "hetero", "all"),
        default="all",
        help="which suite to run (smoke is also --smoke)",
    )
    ap.add_argument("--smoke", action="store_true", help="fast development suite")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--iters", type=int, default=0, help="0 = suite default")
    ap.add_argument("--warmup", type=int, default=0, help="0 = suite default")
    ap.add_argument("--vram-multiple", type=float, default=1.5)
    ap.add_argument("--out", type=str, default="", help="results directory (default: benchmarks/results/<ts>)")
    args = ap.parse_args(argv)

    smoke = bool(args.smoke or args.suite == "smoke")
    suite = "all" if args.suite == "smoke" else args.suite
    out_dir = Path(args.out) if args.out else results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = collect_environment()
    write_json(out_dir / "environment.json", env)
    print(f"results → {out_dir}")
    print(f"commit={env.get('commit', '?')[:12]} torch={env.get('torch')} cuda={env.get('cuda_device_count')}")

    suites: dict[str, Any] = {}
    fit_iters = args.iters or (5 if smoke else 30)
    fit_warmup = args.warmup or (1 if smoke else 5)
    heavy_iters = args.iters or (2 if smoke else 5)
    heavy_warmup = args.warmup or 1

    if suite in ("fit", "all"):
        payload = run_fit_suite(device=args.device, iters=fit_iters, warmup=fit_warmup, smoke=smoke)
        suites["fit"] = payload
        write_json(out_dir / "fit.json", _to_plain(payload))
        _print_fit(_to_plain(payload))

    if suite in ("beyond_vram", "all"):
        payload = run_beyond_vram_suite(
            vram_multiple=args.vram_multiple,
            iters=heavy_iters,
            warmup=heavy_warmup,
            smoke=smoke,
        )
        suites["beyond_vram"] = payload
        write_json(out_dir / "beyond_vram.json", _to_plain(payload))
        _print_beyond(_to_plain(payload))

    if suite in ("pressure", "all"):
        payload = run_memory_pressure_suite(iters=heavy_iters, warmup=heavy_warmup, smoke=smoke)
        suites["memory_pressure"] = payload
        write_json(out_dir / "memory_pressure.json", _to_plain(payload))
        print("\n=== memory_pressure ===")
        for row in _to_plain(payload).get("results", []):
            tt_run = row.get("tensortorrent") or {}
            status = f"{tt_run.get('median_ms', 0):.1f} ms" if tt_run.get("ok") else tt_run.get("note", "FAIL")
            print(f"  budget={row['budget_fraction'] * 100:.0f}% → {status}")

    if suite in ("scaling", "all"):
        payload = run_model_size_scaling_suite(iters=max(2, heavy_iters - 1), warmup=heavy_warmup, smoke=smoke)
        suites["model_size_scaling"] = payload
        write_json(out_dir / "model_size_scaling.json", _to_plain(payload))
        print("\n=== model_size_scaling ===")
        for row in _to_plain(payload).get("results", []):
            eg = (row.get("approaches") or {}).get("gpu_eager") or {}
            tt_run = (row.get("approaches") or {}).get("tensortorrent") or {}
            eg_s = f"{eg['median_ms']:.1f}ms" if eg.get("ok") else "OOM/FAIL"
            tt_s = f"{tt_run['median_ms']:.1f}ms" if tt_run.get("ok") else tt_run.get("note", "FAIL")[:40]
            print(f"  {row['vram_multiple']:.2f}× → eager={eg_s} TT={tt_s}")

    if suite in ("hetero", "all"):
        payload = run_hetero_suite(smoke=smoke)
        suites["heterogeneous"] = payload
        write_json(out_dir / "heterogeneous.json", _to_plain(payload))
        print("\n=== heterogeneous ===")
        for row in _to_plain(payload).get("results", []):
            print(f"  {row.get('case')}: {row.get('evidence')} {row.get('note', '')}")

    summary = {
        "environment": env,
        "suite": suite,
        "smoke": smoke,
        "suites": _to_plain(suites),
    }
    write_json(out_dir / "summary.json", summary)
    plots = try_plot(out_dir, {"suite": "all", "suites": _to_plain(suites)})
    if plots:
        print("plots:", *plots)

    # Non-zero if a critical MEASURED path failed hard.
    beyond = suites.get("beyond_vram") or {}
    tt_beyond = (beyond.get("approaches") or {}).get("tensortorrent")
    if (
        isinstance(tt_beyond, TimedRun)
        and not tt_beyond.ok
        and suite in ("beyond_vram", "all")
        and not smoke
        and env.get("cuda_available")
    ):
        print("CRITICAL: beyond_vram TensorTorrent failed", tt_beyond.note, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
