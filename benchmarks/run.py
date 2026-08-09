#!/usr/bin/env python3
"""Legacy benchmark entry; prefer ``python -m benchmarks.public``.

Keeps suite names ``beyond_vram`` / ``pressure`` / ``scaling``. ``--suite all``
runs each suite in a child process.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.harness import (
    TimedRun,
    collect_environment,
    release_host_memory,
    results_dir,
    to_plain,
    write_json,
    write_suite_json,
)
from benchmarks.report import try_plot_all
from benchmarks.suites import (
    run_beyond_vram_suite,
    run_fit_suite,
    run_hetero_suite,
    run_memory_budget_curve_suite,
    run_memory_pressure_suite,
    run_model_size_scaling_suite,
)


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


def _run_all_isolated(out_dir: Path, args: argparse.Namespace, *, smoke: bool) -> int:
    """One child per suite — same RAM contract as ``benchmarks.public --suite all``."""
    if smoke:
        names = ("fit", "budget", "hetero")
    else:
        names = ("fit", "beyond_vram", "pressure", "budget", "crossover", "hetero")
    rc = 0
    for name in names:
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.run",
            "--suite",
            name,
            "--out",
            str(out_dir),
            "--device",
            args.device,
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--vram-multiple",
            str(args.vram_multiple),
        ]
        if smoke:
            cmd.append("--smoke")
        print(f"\n=== subprocess suite={name} ===", flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            rc = proc.returncode
            print(f"suite {name} failed rc={proc.returncode}", file=sys.stderr)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite",
        choices=("smoke", "fit", "beyond_vram", "pressure", "budget", "scaling", "crossover", "hetero", "all"),
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

    if suite == "all":
        return _run_all_isolated(out_dir, args, smoke=smoke)

    env = collect_environment()
    write_json(out_dir / "environment.json", env)
    print(f"results → {out_dir}")
    print(f"commit={env.get('commit', '?')[:12]} torch={env.get('torch')} cuda={env.get('cuda_device_count')}")

    suites: dict[str, Any] = {}
    fit_iters = args.iters or (5 if smoke else 30)
    fit_warmup = args.warmup or (1 if smoke else 5)
    heavy_iters = args.iters or (2 if smoke else 5)
    heavy_warmup = args.warmup or 1

    if suite == "fit":
        payload = run_fit_suite(device=args.device, iters=fit_iters, warmup=fit_warmup, smoke=smoke)
        suites["fit"] = payload
        write_suite_json(out_dir, payload, "fit.json")
        _print_fit(to_plain(payload))

    if suite == "beyond_vram":
        payload = run_beyond_vram_suite(
            vram_multiple=args.vram_multiple,
            iters=heavy_iters,
            warmup=heavy_warmup,
            smoke=smoke,
        )
        suites["beyond_vram"] = payload
        write_suite_json(out_dir, payload, "beyond_vram.json")
        _print_beyond(to_plain(payload))

    if suite == "pressure":
        payload = run_memory_pressure_suite(iters=heavy_iters, warmup=heavy_warmup, smoke=smoke)
        suites["memory_pressure"] = payload
        write_suite_json(out_dir, payload, "memory_pressure.json")
        print("\n=== memory_pressure ===")
        for row in to_plain(payload).get("results", []):
            tt_run = row.get("tensortorrent") or {}
            status = f"{tt_run.get('median_ms', 0):.1f} ms" if tt_run.get("ok") else tt_run.get("note", "FAIL")
            print(f"  budget={row['budget_fraction'] * 100:.0f}% → {status}")

    if suite == "budget":
        payload = run_memory_budget_curve_suite(iters=heavy_iters, warmup=heavy_warmup, smoke=smoke)
        suites["memory_budget_curve"] = payload
        write_suite_json(out_dir, payload, "memory_budget_curve.json")
        print("\n=== memory_budget_curve ===")
        for row in to_plain(payload).get("results", []):
            tt_run = row.get("tensortorrent") or {}
            gib = row.get("vram_budget_gib", 0)
            status = f"{tt_run.get('median_ms', 0):.1f} ms" if tt_run.get("ok") else tt_run.get("note", "FAIL")
            print(f"  budget={gib:.1f} GiB → {status}")

    if suite in ("scaling", "crossover"):
        full_crossover = suite == "crossover"
        payload = run_model_size_scaling_suite(
            iters=max(2, heavy_iters - 1),
            warmup=heavy_warmup,
            smoke=smoke,
            full_crossover=full_crossover,
        )
        suites["model_size_scaling"] = payload
        names = ["model_size_scaling.json"]
        if full_crossover:
            suites["model_size_crossover"] = payload
            names.append("model_size_crossover.json")
        write_suite_json(out_dir, payload, *names)
        print("\n=== model_size_scaling ===")
        for row in to_plain(payload).get("results", []):
            eg = (row.get("approaches") or {}).get("gpu_eager") or {}
            tt_run = (row.get("approaches") or {}).get("tensortorrent") or {}
            eg_s = f"{eg['median_ms']:.1f}ms" if eg.get("ok") else "OOM/FAIL"
            tt_s = f"{tt_run['median_ms']:.1f}ms" if tt_run.get("ok") else tt_run.get("note", "FAIL")[:40]
            print(f"  {row['vram_multiple']:.2f}× → eager={eg_s} TT={tt_s}")

    if suite == "hetero":
        payload = run_hetero_suite(smoke=smoke)
        suites["heterogeneous"] = payload
        write_suite_json(out_dir, payload, "heterogeneous.json")
        print("\n=== heterogeneous ===")
        for row in to_plain(payload).get("results", []):
            print(f"  {row.get('case')}: {row.get('evidence')} {row.get('note', '')}")

    release_host_memory()
    summary = {
        "environment": env,
        "suite": suite,
        "smoke": smoke,
        "suites": to_plain(suites),
    }
    write_json(out_dir / "summary.json", summary)
    plots = try_plot_all(out_dir, summary)
    if plots:
        print("plots:", *plots)

    beyond = suites.get("beyond_vram") or {}
    tt_beyond = (beyond.get("approaches") or {}).get("tensortorrent")
    if (
        isinstance(tt_beyond, TimedRun)
        and not tt_beyond.ok
        and suite == "beyond_vram"
        and not smoke
        and env.get("cuda_available")
    ):
        print("CRITICAL: beyond_vram TensorTorrent failed", tt_beyond.note, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    raise SystemExit(main())
