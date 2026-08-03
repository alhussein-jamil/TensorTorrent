#!/usr/bin/env python3
"""Benchmark TensorTorrent against the runtimes people would otherwise use.

The other scripts in this directory compare against eager PyTorch. Eager is a
floor, not a competitor: beating it says nothing about whether TensorTorrent is
worth choosing over the alternatives. This script measures the same model on
the same machine across every runtime that is actually a candidate:

* eager PyTorch            — the floor
* torch.compile (Inductor) — the default answer to "make my model faster"
* AOTInductor              — PyTorch's own ahead-of-time compiler, the closest
                             peer to what TensorTorrent does
* ONNX Runtime (CPU EP)    — the established graph-compiler runtime
* TensorTorrent            — this project

Every runtime is checked for numerical agreement with eager before its timings
are reported, so a fast-but-wrong backend cannot look good. Runtimes that fail
to build or run are reported as failures rather than quietly dropped.

Usage:
    uv run python bench/compare_baselines.py
    uv run python bench/compare_baselines.py --iters 100 --json results.json

Honest reading of the output: this measures single-process CPU inference on
whatever machine you run it on. It says nothing about GPU placement, parameter
streaming, or multi-device scheduling — the features that motivate
TensorTorrent in the first place. Those need a machine with real accelerators
and a model too large for them; see docs/product/deployment.md.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Models: small enough to run anywhere, shaped to stress different limits.
# --------------------------------------------------------------------------


class MLPStack(nn.Module):
    """Compute-bound: a deep chain of dense layers."""

    def __init__(self, width: int = 512, depth: int = 8) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Realistic shape: attention + MLP, the unit most inference work is made of."""

    def __init__(self, dim: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        return x + self.mlp(self.norm2(x))


class WideBranching(nn.Module):
    """Memory/transfer-bound with independent branches the planner can overlap."""

    def __init__(self, width: int = 1024) -> None:
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.left = nn.Linear(width, width)
        self.right = nn.Linear(width, width)
        self.head = nn.Linear(width * 2, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        return self.head(torch.cat([torch.relu(self.left(h)), torch.tanh(self.right(h))], dim=-1))


WORKLOADS: dict[str, tuple[Callable[[], nn.Module], tuple[int, ...]]] = {
    # Small: dominated by per-forward dispatch overhead.
    "mlp_stack_512x8": (lambda: MLPStack(), (32, 512)),
    "transformer_block_256": (lambda: TransformerBlock(), (8, 64, 256)),
    "wide_branching_1024": (lambda: WideBranching(), (16, 1024)),
    # Large: enough compute per forward that a fixed scheduling cost should
    # amortise. If a runtime only wins here, that is the honest story to tell.
    "mlp_stack_2048x16": (lambda: MLPStack(width=2048, depth=16), (64, 2048)),
    "transformer_block_1024": (lambda: TransformerBlock(dim=1024, heads=8), (16, 256, 1024)),
}


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass
class Result:
    runtime: str
    workload: str
    ok: bool
    median_ms: float = 0.0
    p95_ms: float = 0.0
    stdev_ms: float = 0.0
    compile_s: float = 0.0
    max_abs_err: float | None = None
    note: str = ""
    samples: list[float] = field(default_factory=list, repr=False)


DEVICE = "cpu"


def _sync() -> None:
    """Block until queued device work is done.

    CUDA kernel launches are asynchronous: without this every GPU timing would
    measure launch latency rather than execution, and every runtime would look
    identically (and impossibly) fast.
    """
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _time_calls(fn: Callable[[], Any], iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync()
    out: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _summarise(res: Result, samples: list[float]) -> Result:
    res.samples = samples
    res.median_ms = statistics.median(samples)
    res.p95_ms = sorted(samples)[min(len(samples) - 1, max(0, math.ceil(len(samples) * 0.95) - 1))]
    res.stdev_ms = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return res


def _err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())


# --- one function per runtime; each returns a Result -----------------------


def run_eager(model: nn.Module, x: torch.Tensor, iters: int, warmup: int, wl: str) -> tuple[Result, torch.Tensor]:
    with torch.no_grad():
        ref = model(x)
        samples = _time_calls(lambda: model(x), iters, warmup)
    return _summarise(Result("eager", wl, True, max_abs_err=0.0), samples), ref


def run_torch_compile(model: nn.Module, x: torch.Tensor, ref: torch.Tensor, iters: int, warmup: int, wl: str) -> Result:
    try:
        t0 = time.perf_counter()
        compiled = torch.compile(model, backend="inductor")
        with torch.no_grad():
            out = compiled(x)  # triggers compilation
        compile_s = time.perf_counter() - t0
        with torch.no_grad():
            samples = _time_calls(lambda: compiled(x), iters, warmup)
        res = Result("torch.compile", wl, True, compile_s=compile_s, max_abs_err=_err(out, ref))
        return _summarise(res, samples)
    except Exception as exc:  # noqa: BLE001 - a failing baseline is a result
        return Result("torch.compile", wl, False, note=f"{type(exc).__name__}: {exc}"[:160])


def run_aot_inductor(model: nn.Module, x: torch.Tensor, ref: torch.Tensor, iters: int, warmup: int, wl: str) -> Result:
    try:
        t0 = time.perf_counter()
        ep = torch.export.export(model, (x,))
        path = torch._inductor.aoti_compile_and_package(ep)  # type: ignore[attr-defined]
        runner = torch._inductor.aoti_load_package(path)  # type: ignore[attr-defined]
        compile_s = time.perf_counter() - t0
        out = runner(x)
        out_t = out[0] if isinstance(out, (list, tuple)) else out
        samples = _time_calls(lambda: runner(x), iters, warmup)
        res = Result("AOTInductor", wl, True, compile_s=compile_s, max_abs_err=_err(out_t, ref))
        return _summarise(res, samples)
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        if "CUDA_HOME" in detail or "nvcc" in detail:
            detail = (
                "needs a system CUDA toolkit (nvcc) on PATH or CUDA_HOME; the PyPI "
                "torch wheels ship headers and libs but no compiler"
            )
        return Result("AOTInductor", wl, False, note=detail[:160])


def run_onnxruntime(model: nn.Module, x: torch.Tensor, ref: torch.Tensor, iters: int, warmup: int, wl: str) -> Result:
    try:
        import io

        import numpy as np
        import onnxruntime as ort

        t0 = time.perf_counter()
        buf = io.BytesIO()
        try:
            torch.onnx.export(model.cpu(), (x.cpu(),), buf, input_names=["x"], output_names=["y"], dynamo=False)
        finally:
            model.to(DEVICE)  # restore even if export fails so later runtimes see the right device
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        available = ort.get_available_providers()
        if DEVICE == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(buf.getvalue(), opts, providers=providers)
        # The onnxruntime CPU wheel silently ignores CUDAExecutionProvider, so a
        # GPU sweep would otherwise show a CPU measurement in a table of GPU
        # numbers. Name the provider actually in use instead of hiding it.
        active = sess.get_providers()
        label = "onnxruntime"
        if DEVICE == "cuda" and "CUDAExecutionProvider" not in active:
            label = "onnxruntime[CPU-EP]"
        compile_s = time.perf_counter() - t0
        feed = {"x": x.detach().cpu().numpy()}
        out = sess.run(None, feed)[0]
        samples = _time_calls(lambda: sess.run(None, feed), iters, warmup)
        res = Result(label, wl, True, compile_s=compile_s, max_abs_err=_err(torch.from_numpy(np.asarray(out)), ref))
        if label.endswith("[CPU-EP]"):
            res.note = "onnxruntime-gpu not installed; measured on CPU, not comparable to the GPU rows"
        return _summarise(res, samples)
    except Exception as exc:  # noqa: BLE001
        return Result("onnxruntime", wl, False, note=f"{type(exc).__name__}: {exc}"[:160])


def run_tensortorrent(model: nn.Module, x: torch.Tensor, ref: torch.Tensor, iters: int, warmup: int, wl: str) -> Result:
    try:
        import tensortorrent as tt

        t0 = time.perf_counter()
        compiled = tt.compile(model, example_inputs=(x,))
        compile_s = time.perf_counter() - t0
        with torch.no_grad():
            out = compiled(x)
            samples = _time_calls(lambda: compiled(x), iters, warmup)
        res = Result("tensortorrent", wl, True, compile_s=compile_s, max_abs_err=_err(out, ref))
        summarised = _summarise(res, samples)
        with contextlib.suppress(Exception):  # cleanup must not fail the benchmark
            compiled.close()
        return summarised
    except Exception as exc:  # noqa: BLE001
        return Result("tensortorrent", wl, False, note=f"{type(exc).__name__}: {exc}"[:160])


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def environment() -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "device": DEVICE,
        "gpu": gpu,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }


def markdown(results: list[Result], env: dict[str, Any]) -> str:
    lines = [
        "# Runtime comparison",
        "",
        f"- device **{env['device']}**" + (f" — {env['gpu']} (x{env['gpu_count']})" if env.get("gpu") else ""),
        f"- python {env['python']}, torch {env['torch']}, threads {env['torch_threads']}",
        f"- {env['platform']}",
        f"- CUDA available: {env['cuda_available']}",
        "",
        "Latency is per forward pass, lower is better. `rel` is relative to eager",
        "on the same workload (below 1.00 is faster than eager). `err` is the max",
        "absolute deviation from the eager result.",
        "",
    ]
    for wl in dict.fromkeys(r.workload for r in results):
        rows = [r for r in results if r.workload == wl]
        base = next((r.median_ms for r in rows if r.runtime == "eager" and r.ok), 0.0)
        lines += [
            f"## {wl}",
            "",
            "| runtime | median ms | p95 ms | rel | compile s | err | status |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            if not r.ok:
                lines.append(f"| {r.runtime} | – | – | – | – | – | FAILED: {r.note} |")
                continue
            rel = f"{r.median_ms / base:.2f}x" if base else "–"
            err = "–" if r.max_abs_err is None else f"{r.max_abs_err:.2e}"
            status = f"ok — {r.note}" if r.note else "ok"
            lines.append(
                f"| {r.runtime} | {r.median_ms:.3f} | {r.p95_ms:.3f} | {rel} | {r.compile_s:.2f} | {err} | {status} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--markdown", type=str, default="")
    ap.add_argument("--workload", type=str, default="", help="run only this workload")
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="auto uses CUDA when available; the baselines run on this device",
    )
    args = ap.parse_args()

    global DEVICE
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    print(f"benchmarking on: {DEVICE}", flush=True)

    torch.manual_seed(0)
    results: list[Result] = []
    items = WORKLOADS.items()
    if args.workload:
        items = [(k, v) for k, v in WORKLOADS.items() if k == args.workload]  # type: ignore[assignment]

    for name, (factory, shape) in items:
        print(f"== {name} {tuple(shape)}", flush=True)
        model = factory().eval().to(DEVICE)
        x = torch.randn(*shape, device=DEVICE)
        eager, ref = run_eager(model, x, args.iters, args.warmup, name)
        results.append(eager)
        print(f"   eager          {eager.median_ms:8.3f} ms", flush=True)
        for label, fn in (
            ("torch.compile", run_torch_compile),
            ("AOTInductor", run_aot_inductor),
            ("onnxruntime", run_onnxruntime),
            ("tensortorrent", run_tensortorrent),
        ):
            r = fn(model, x, ref, args.iters, args.warmup, name)  # type: ignore[operator]
            results.append(r)
            status = f"{r.median_ms:8.3f} ms" if r.ok else f"FAILED ({r.note[:60]})"
            print(f"   {label:<14} {status}", flush=True)

    env = environment()
    report = markdown(results, env)
    print()
    print(report)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(report)
    if args.json:
        payload = {
            "environment": env,
            "results": [{k: v for k, v in r.__dict__.items() if k != "samples"} for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
