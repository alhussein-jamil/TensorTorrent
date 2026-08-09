"""Public launch benchmark entry: real transformer + capacity curves.

Usage::

    python -m benchmarks.public --suite deepmlp
    python -m benchmarks.public --suite transformer
    python -m benchmarks.public --suite budget
    python -m benchmarks.public --suite crossover
    python -m benchmarks.public --suite all
    python -m benchmarks.smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.harness import (
    collect_environment,
    release_host_memory,
    results_dir,
    to_plain,
    write_json,
    write_suite_json,
)
from benchmarks.memory_hygiene import host_available_bytes, public_suite_names
from benchmarks.report import render_markdown_tables, try_plot_all
from benchmarks.suites import (
    run_beyond_vram_suite,
    run_fit_suite,
    run_hetero_suite,
    run_memory_budget_curve_suite,
    run_model_size_scaling_suite,
    run_transformer_beyond_vram_suite,
)


def _run_all_as_subprocesses(out_dir: Path, args: argparse.Namespace) -> int:
    """Run each suite in its own process so RSS cannot accumulate."""
    suites = list(public_suite_names(smoke=bool(args.smoke)))
    rc = 0
    for name in suites:
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.public",
            "--suite",
            name,
            "--out",
            str(out_dir),
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--model-id",
            args.model_id,
            "--seq-len",
            str(args.seq_len),
        ]
        if args.smoke:
            cmd.append("--smoke")
        print(f"\n=== subprocess suite={name} ===", flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            rc = proc.returncode
            print(f"suite {name} failed rc={proc.returncode}", file=sys.stderr)

    suites_payload: dict[str, Any] = {}
    for fname, key in (
        ("fit.json", "fit"),
        ("beyond_vram_deepmlp.json", "beyond_vram_deepmlp"),
        ("transformer_beyond_vram.json", "transformer_beyond_vram"),
        ("memory_budget_curve.json", "memory_budget_curve"),
        ("model_size_crossover.json", "model_size_crossover"),
        ("heterogeneous.json", "heterogeneous"),
    ):
        path = out_dir / fname
        if path.exists():
            suites_payload[key] = json.loads(path.read_text(encoding="utf-8"))
    env = collect_environment()
    summary = {"environment": env, "suite": "all", "smoke": bool(args.smoke), "suites": suites_payload}
    write_json(out_dir / "summary.json", summary)
    md = render_markdown_tables(summary)
    (out_dir / "REPORT.md").write_text(md, encoding="utf-8")
    print(md)
    plots = try_plot_all(out_dir, summary)
    if plots:
        print("plots:", *plots)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suite",
        choices=(
            "all",
            "transformer",
            "deepmlp",
            "budget",
            "crossover",
            "fit",
            "hetero",
        ),
        default="deepmlp",
        help="default deepmlp (not all) — keeps host RAM bounded",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--seq-len", type=int, default=16)
    args = ap.parse_args(argv)

    smoke = bool(args.smoke)
    out_dir = Path(args.out) if args.out else results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.suite == "all":
        return _run_all_as_subprocesses(out_dir, args)

    env = collect_environment()
    write_json(out_dir / "environment.json", env)
    print(f"results → {out_dir}")
    print(f"commit={str(env.get('commit', '?'))[:12]} torch={env.get('torch')}")
    avail = host_available_bytes()
    if avail is not None:
        print(f"host_ram_available_gib={avail / (1024**3):.1f}")

    suites: dict[str, Any] = {}
    heavy_iters = args.iters or (1 if smoke else 3)
    heavy_warmup = args.warmup or (0 if smoke else 1)
    fit_iters = args.iters or (3 if smoke else 20)
    suite = args.suite

    if suite == "fit":
        payload = run_fit_suite(device="cuda", iters=fit_iters, warmup=1 if smoke else 3, smoke=smoke)
        suites["fit"] = payload
        write_suite_json(out_dir, payload, "fit.json")

    if suite == "deepmlp":
        payload = run_beyond_vram_suite(
            vram_multiple=1.5,
            iters=heavy_iters,
            warmup=heavy_warmup,
            smoke=smoke,
            instrument=True,
        )
        suites["beyond_vram_deepmlp"] = payload
        write_suite_json(out_dir, payload, "beyond_vram_deepmlp.json")

    if suite == "transformer" and not smoke:
        payload = run_transformer_beyond_vram_suite(
            model_id=args.model_id,
            seq_len=args.seq_len,
            iters=max(1, min(heavy_iters, 2)),
            warmup=heavy_warmup,
        )
        suites["transformer_beyond_vram"] = payload
        write_suite_json(out_dir, payload, "transformer_beyond_vram.json")
    elif suite == "transformer" and smoke:
        print("smoke skips full HF transformer; omit --smoke")

    if suite == "budget":
        payload = run_memory_budget_curve_suite(iters=heavy_iters, warmup=heavy_warmup, smoke=smoke)
        suites["memory_budget_curve"] = payload
        write_suite_json(out_dir, payload, "memory_budget_curve.json")

    if suite == "crossover":
        payload = run_model_size_scaling_suite(
            iters=max(1, heavy_iters),
            warmup=heavy_warmup,
            smoke=smoke,
            full_crossover=not smoke,
        )
        suites["model_size_crossover"] = payload
        write_suite_json(out_dir, payload, "model_size_crossover.json")

    if suite == "hetero":
        payload = run_hetero_suite(smoke=smoke)
        suites["heterogeneous"] = payload
        write_suite_json(out_dir, payload, "heterogeneous.json")

    release_host_memory()
    summary = {"environment": env, "suite": suite, "smoke": smoke, "suites": to_plain(suites)}
    write_json(out_dir / "summary.json", summary)
    md = render_markdown_tables(summary)
    (out_dir / "REPORT.md").write_text(md, encoding="utf-8")
    print(md)
    plots = try_plot_all(out_dir, summary)
    if plots:
        print("plots:", *plots)

    if not smoke and suite == "transformer" and env.get("cuda_available") and "transformer_beyond_vram" in suites:
        tt_run = (suites["transformer_beyond_vram"].get("approaches") or {}).get("tensortorrent")
        ok = getattr(tt_run, "ok", None)
        if ok is None and isinstance(tt_run, dict):
            ok = tt_run.get("ok")
        if tt_run is not None and not ok:
            note = getattr(tt_run, "note", None) or (tt_run.get("note") if isinstance(tt_run, dict) else "")
            print("CRITICAL: transformer TensorTorrent failed:", note, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    raise SystemExit(main())
