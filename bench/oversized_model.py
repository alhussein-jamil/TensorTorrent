#!/usr/bin/env python3
"""The benchmark that decides whether TensorTorrent is worth using.

`compare_baselines.py` measures a model that fits comfortably on one device.
On that ground TensorTorrent has nothing to offer: it adds scheduling overhead
and wins nothing (see docs/product/benchmarks.md). Its entire reason to exist
is the case that script cannot construct — a model **too large for the
accelerator**, where the alternative is not "run it faster" but "run it at
all".

This script builds a model deliberately larger than available VRAM and
compares the ways you could actually run it:

* TensorTorrent  — parameter streaming and activation spill under a budget
* Accelerate     — `device_map="auto"` with CPU/disk offload, the baseline
                   almost everyone reaches for first
* CPU-only eager — always works, sets the "how much did the GPU buy you" floor
* GPU-only eager — expected to OOM; recorded as a failure, because "the
                   alternative OOMs" is exactly the claim being tested

Sizing is derived from the GPU actually present, so the model is genuinely
oversized on the machine you run it on rather than a hardcoded guess.

Usage:
    uv run python bench/oversized_model.py                    # auto-size
    uv run python bench/oversized_model.py --vram-multiple 2  # 2x VRAM
    uv run python bench/oversized_model.py --json results.json

Report honestly: if TensorTorrent is slower than Accelerate offload here, that
is the headline result and it should be published, not buried.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class Outcome:
    approach: str
    ok: bool
    median_ms: float = 0.0
    p95_ms: float = 0.0
    peak_host_gb: float = 0.0
    peak_device_gb: float = 0.0
    load_s: float = 0.0
    note: str = ""


class BigMLP(nn.Module):
    """Parameter-heavy by construction: width^2 floats per layer."""

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = torch.relu(blk(x))
        return self.head(x)


def param_bytes(width: int, depth: int) -> int:
    return (width * width + width) * depth * 4 + (width * 8 + 8) * 4


def size_for_target(target_bytes: int, width: int = 4096) -> tuple[int, int]:
    """Pick a depth whose parameters exceed ``target_bytes`` at this width."""
    per_layer = (width * width + width) * 4
    depth = max(2, int(target_bytes / per_layer) + 1)
    return width, depth


def device_total_bytes() -> int:
    if torch.cuda.is_available():
        return int(torch.cuda.get_device_properties(0).total_memory)
    return 0


def _reset_peaks() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_device_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def _peak_host_gb() -> float:
    try:
        import resource

        # ru_maxrss is KiB on Linux.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:  # noqa: BLE001 - diagnostics only
        return 0.0


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn: Any, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _finish(o: Outcome, samples: list[float]) -> Outcome:
    o.median_ms = statistics.median(samples)
    o.p95_ms = sorted(samples)[min(len(samples) - 1, max(0, math.ceil(len(samples) * 0.95) - 1))]
    o.peak_device_gb = _peak_device_gb()
    o.peak_host_gb = _peak_host_gb()
    return o


# --- approaches ------------------------------------------------------------


def try_gpu_eager(model: nn.Module, x: torch.Tensor, iters: int, warmup: int) -> Outcome:
    """Expected to OOM. That failure is the point of the comparison."""
    if not torch.cuda.is_available():
        return Outcome("gpu eager", False, note="no CUDA device")
    _reset_peaks()
    try:
        m = model.cuda().eval()
        xd = x.cuda()
        with torch.no_grad():
            samples = _timed(lambda: m(xd), iters, warmup)
        out = _finish(Outcome("gpu eager", True), samples)
        m.cpu()
        _reset_peaks()
        return out
    except torch.cuda.OutOfMemoryError as exc:
        _reset_peaks()
        return Outcome("gpu eager", False, note=f"CUDA OOM (expected): {str(exc)[:90]}")
    except Exception as exc:  # noqa: BLE001
        _reset_peaks()
        return Outcome("gpu eager", False, note=f"{type(exc).__name__}: {exc}"[:120])


def try_cpu_eager(model: nn.Module, x: torch.Tensor, iters: int, warmup: int) -> Outcome:
    _reset_peaks()
    try:
        m = model.cpu().eval()
        with torch.no_grad():
            samples = _timed(lambda: m(x.cpu()), iters, warmup)
        return _finish(Outcome("cpu eager", True), samples)
    except Exception as exc:  # noqa: BLE001
        return Outcome("cpu eager", False, note=f"{type(exc).__name__}: {exc}"[:120])


def try_accelerate(model: nn.Module, x: torch.Tensor, iters: int, warmup: int) -> Outcome:
    """HuggingFace Accelerate device_map='auto' — the usual answer to this problem."""
    _reset_peaks()
    try:
        from accelerate import cpu_offload, dispatch_model, infer_auto_device_map

        t0 = time.perf_counter()
        if torch.cuda.is_available():
            dmap = infer_auto_device_map(model, no_split_module_classes=["Linear"])
            m = dispatch_model(model, device_map=dmap)
            xd = x.cuda()
        else:
            m = cpu_offload(model, execution_device=torch.device("cpu"))
            xd = x.cpu()
        load_s = time.perf_counter() - t0
        with torch.no_grad():
            samples = _timed(lambda: m(xd), iters, warmup)
        out = _finish(Outcome("accelerate device_map", True), samples)
        out.load_s = load_s
        return out
    except ImportError:
        return Outcome("accelerate device_map", False, note="pip install accelerate to enable this baseline")
    except Exception as exc:  # noqa: BLE001
        return Outcome("accelerate device_map", False, note=f"{type(exc).__name__}: {exc}"[:120])


def try_tensortorrent(model: nn.Module, x: torch.Tensor, iters: int, warmup: int, vram_budget: int | None) -> Outcome:
    _reset_peaks()
    try:
        import tensortorrent as tt

        # Streaming config: one layer resident at a time on GPU, with prefetch,
        # host budget large enough to hold the pack file. Matches the
        # configuration used by tests/hardware/test_vram_size_sweep.py which
        # exercises the same oversize-model streaming path.
        layer_bytes = x.shape[-1] * x.shape[-1] * 4 + x.shape[-1] * 4
        ram_budget = max(layer_bytes * 4, 128 << 20)
        t0 = time.perf_counter()
        cfg = tt.CompileConfig(
            use_torch_compile=False,
            measure_regions=False,
            allow_gpu=True,
            allow_cpu=True,
            ram_budget_bytes=ram_budget,
            vram_budget_bytes=vram_budget,
            max_region_nodes=1,
            prefetch_distance=1,
        )
        compiled = tt.compile(model.cpu().eval(), example_inputs=(x.cpu(),), config=cfg)
        load_s = time.perf_counter() - t0
        with torch.no_grad():
            samples = _timed(lambda: compiled(x.cpu()), iters, warmup)
        out = _finish(Outcome("tensortorrent", True), samples)
        out.load_s = load_s
        with contextlib.suppress(Exception):  # cleanup must not fail the benchmark
            compiled.close()
        return out
    except Exception as exc:  # noqa: BLE001
        return Outcome("tensortorrent", False, note=f"{type(exc).__name__}: {exc}"[:160])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--depth", type=int, default=0, help="0 derives a depth that overflows VRAM")
    ap.add_argument("--vram-multiple", type=float, default=1.5, help="target params as a multiple of total VRAM")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    total_vram = device_total_bytes()
    if args.depth:
        width, depth = args.width, args.depth
    elif total_vram:
        width, depth = size_for_target(int(total_vram * args.vram_multiple), args.width)
    else:
        width, depth = args.width, 8  # no GPU: keep it runnable, results are indicative only

    pbytes = param_bytes(width, depth)
    print(f"device VRAM      : {total_vram / 1e9:.2f} GB" if total_vram else "device VRAM      : none (CPU only)")
    print(f"model            : width={width} depth={depth}")
    print(f"parameter bytes  : {pbytes / 1e9:.2f} GB", flush=True)
    if total_vram:
        print(f"params / VRAM    : {pbytes / total_vram:.2f}x", flush=True)
    print()

    torch.manual_seed(0)
    x = torch.randn(args.batch, width)

    results: list[Outcome] = []
    for label, fn in (
        ("gpu eager", lambda m: try_gpu_eager(m, x, args.iters, args.warmup)),
        ("cpu eager", lambda m: try_cpu_eager(m, x, args.iters, args.warmup)),
        ("accelerate", lambda m: try_accelerate(m, x, args.iters, args.warmup)),
        ("tensortorrent", lambda m: try_tensortorrent(m, x, args.iters, args.warmup, total_vram or None)),
    ):
        print(f"-- {label} ...", flush=True)
        model = BigMLP(width, depth)  # fresh model per approach; they mutate placement
        r = fn(model)
        results.append(r)
        if r.ok:
            print(
                f"   {r.median_ms:9.1f} ms   dev peak {r.peak_device_gb:5.2f} GB   host peak {r.peak_host_gb:5.2f} GB"
            )
        else:
            print(f"   FAILED: {r.note}")
        del model
        _reset_peaks()

    print()
    print("| approach | median ms | p95 ms | device peak GB | host peak GB | load s | status |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        if r.ok:
            print(
                f"| {r.approach} | {r.median_ms:.1f} | {r.p95_ms:.1f} | "
                f"{r.peak_device_gb:.2f} | {r.peak_host_gb:.2f} | {r.load_s:.1f} | ok |"
            )
        else:
            print(f"| {r.approach} | – | – | – | – | – | {r.note} |")

    if args.json:
        payload = {
            "environment": {
                "platform": platform.platform(),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "vram_bytes": total_vram,
            },
            "model": {"width": width, "depth": depth, "param_bytes": pbytes},
            "results": [r.__dict__ for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
