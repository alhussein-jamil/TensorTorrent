#!/usr/bin/env python3
"""Four experiments that decide what gets built next. Nothing else.

Each one answers a question I cannot answer without a GPU, and each result
changes a decision rather than just adding a number.

    uv run python tools/answer_open_questions.py

Roughly 5-10 minutes. Experiment 2 allocates a model ~1.2x your VRAM, so give
it disk and don't run it while gaming.
"""

from __future__ import annotations

import contextlib
import gc
import os
import statistics
import time
from typing import Any

import torch
import torch.nn as nn

RULE = "=" * 74


def hdr(n: int, title: str, decides: str) -> None:
    print(f"\n{RULE}\nEXPERIMENT {n}: {title}\ndecides: {decides}\n{RULE}", flush=True)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn: Any, iters: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    sync()
    xs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        xs.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(xs)


def free_vram() -> int:
    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def reset() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class MLP(nn.Module):
    def __init__(self, width: int, depth: int, out: int = 10) -> None:
        super().__init__()
        seq: list[nn.Module] = []
        for _ in range(depth):
            seq += [nn.Linear(width, width), nn.ReLU()]
        seq.append(nn.Linear(width, out))
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Big(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.head = nn.Linear(width, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = torch.relu(b(x))
        return self.head(x)


# ---------------------------------------------------------------------------
# 1. Does the direct path help on GPU, and is it even correct there?
# ---------------------------------------------------------------------------


def experiment_1() -> None:
    hdr(
        1,
        "direct path on GPU (TT_DIRECT_PATH=1)",
        "whether the fast path becomes the default, overriding anti-patterns 7 and 8",
    )
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return
    print("NOTE: the parameter-placement hoist (tensor.to(cuda)) has never run on")
    print("      a GPU. Correctness here matters more than the speed number.\n")

    import tensortorrent as tt

    for width, depth, batch in ((512, 8, 32), (1024, 4, 16)):
        model = MLP(width, depth).eval().cuda()
        x = torch.randn(batch, width, device="cuda")
        with torch.no_grad():
            ref = model(x)
            eager_ms = bench(lambda m=model, xx=x: m(xx))

        row = {}
        for label, flag in (("scheduler", "0"), ("direct", "1")):
            os.environ["TT_DIRECT_PATH"] = flag
            try:
                c = tt.compile(model, example_inputs=(x,))
                engaged = getattr(c.executor, "direct_plan", None) is not None
                with torch.no_grad():
                    got = c(x)
                    err = float((got.detach().cpu() - ref.detach().cpu()).abs().max())
                    ms = bench(lambda cc=c, xx=x: cc(xx))
                row[label] = (ms, err, engaged)
                with contextlib.suppress(Exception):
                    c.close()
            except Exception as exc:  # noqa: BLE001
                row[label] = (float("nan"), float("nan"), False)
                print(f"  {label}: FAILED {type(exc).__name__}: {exc}"[:150])
            reset()
        os.environ.pop("TT_DIRECT_PATH", None)

        print(f"  mlp {width}x{depth} batch {batch}:  eager {eager_ms:7.3f} ms")
        for label, (ms, err, engaged) in row.items():
            flag = "engaged" if engaged else "not engaged"
            print(f"    {label:<10} {ms:7.3f} ms  ({ms / eager_ms:.2f}x eager)  err {err:.2e}  [{flag}]")
        del model
        reset()

    print("\n  -> if direct is well under 1.00x here and err is ~1e-7, the fast path")
    print("     is worth making default and I will do it through the planner so")
    print("     there is still one executor.")


# ---------------------------------------------------------------------------
# 2. Was the 63.9 s streaming result caused by a starved host budget?
# ---------------------------------------------------------------------------


def experiment_2() -> None:
    hdr(
        2,
        "streaming host budget sweep",
        "whether the 63.9 s oversized result was self-inflicted by a tiny ram_budget",
    )
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return
    print("This is the most important experiment. My claim is that the benchmark")
    print("starved the host tier (~268 MB) so 12 GiB was re-read from disk every")
    print("forward at ~193 MB/s. If a real host budget collapses the time, the")
    print("tiering work is justified. If it does not, my diagnosis is wrong and I")
    print("should not build on it.\n")

    import tensortorrent as tt

    total = int(torch.cuda.get_device_properties(0).total_memory)
    width = 4096
    per_layer = (width * width + width) * 4
    depth = max(2, int(total * 1.2 / per_layer) + 1)
    pbytes = per_layer * depth
    print(f"  model {width}x{depth} = {pbytes / 1e9:.2f} GB params vs {total / 1e9:.2f} GB VRAM\n")

    x = torch.randn(8, width)
    gib = 1 << 30
    for label, ram in (("starved (268 MB)", 268 << 20), ("8 GiB", 8 * gib), ("32 GiB", 32 * gib)):
        model = Big(width, depth).eval()
        try:
            cfg = tt.CompileConfig(
                use_torch_compile=False,
                measure_regions=False,
                ram_budget_bytes=ram,
                vram_budget_bytes=total,
                max_region_nodes=1,
                prefetch_distance=1,
            )
            t0 = time.perf_counter()
            c = tt.compile(model, example_inputs=(x,), config=cfg)
            compile_s = time.perf_counter() - t0
            with torch.no_grad():
                ms = bench(lambda cc=c, xx=x: cc(xx), iters=3, warmup=1)
            print(f"  ram_budget {label:<18} forward {ms:10.1f} ms   compile {compile_s:6.1f} s")
            with contextlib.suppress(Exception):
                c.close()
        except Exception as exc:  # noqa: BLE001
            print(f"  ram_budget {label:<18} FAILED {type(exc).__name__}: {str(exc)[:90]}")
        del model
        reset()

    print("\n  -> a large drop from starved to 8/32 GiB confirms the diagnosis and")
    print("     means ram_budget_bytes=None must default to the resolved host budget.")


# ---------------------------------------------------------------------------
# 3. Did the stricter eager-vs-Inductor rule wrongly reject Inductor on GPU?
# ---------------------------------------------------------------------------


def experiment_3() -> None:
    hdr(
        3,
        "backend selection after the anti-pattern 9 fix",
        "whether requiring Inductor to actually win loses real GPU speedups",
    )
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return
    print("On CPU Inductor rarely wins, so the stricter rule was clearly right.")
    print("On GPU it usually does win. The risk is that measurement noise now")
    print("rejects an Inductor build that genuinely helps.\n")

    import tensortorrent as tt

    for width, depth, batch in ((512, 8, 32), (2048, 8, 64)):
        model = MLP(width, depth).eval().cuda()
        x = torch.randn(batch, width, device="cuda")
        with torch.no_grad():
            eager_ms = bench(lambda m=model, xx=x: m(xx))
        try:
            c = tt.compile(model, example_inputs=(x,))
            impl = "unknown"
            for r in getattr(c, "regions", []) or []:
                a = getattr(r, "attributes", {}) or {}
                impl = str(a.get("impl", a.get("reason", impl)))
                break
            with torch.no_grad():
                ms = bench(lambda cc=c, xx=x: cc(xx))
            print(
                f"  mlp {width}x{depth}: eager {eager_ms:7.3f} ms | tt {ms:7.3f} ms ({ms / eager_ms:.2f}x) | kept: {impl}"
            )
            with contextlib.suppress(Exception):
                c.close()
        except Exception as exc:  # noqa: BLE001
            print(f"  mlp {width}x{depth}: FAILED {type(exc).__name__}: {exc}"[:140])
        del model
        reset()

    print("\n  -> if 'kept' says eager FX on a workload where Inductor should win,")
    print("     the rule is too strict and needs a noise margin.")


# ---------------------------------------------------------------------------
# 4. Where is the crossover between GPU-resident and streaming?
# ---------------------------------------------------------------------------


def experiment_4() -> None:
    hdr(
        4,
        "fit-to-overflow crossover",
        "whether TensorTorrent degrades smoothly as a model outgrows VRAM",
    )
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return
    print("An always-improve runtime should degrade gracefully, not fall off a")
    print("cliff the moment the model stops fitting.\n")

    import tensortorrent as tt

    free = free_vram()
    width = 2048
    per_layer = (width * width + width) * 4
    x = torch.randn(8, width)
    for frac in (0.25, 0.60, 0.90, 1.20):
        depth = max(2, int(free * frac / per_layer))
        model = Big(width, depth).eval()
        try:
            c = tt.compile(model, example_inputs=(x,))
            with torch.no_grad():
                ms = bench(lambda cc=c, xx=x: cc(xx), iters=5, warmup=2)
            dev = torch.cuda.max_memory_allocated() / 1e9
            print(f"  {frac:.2f}x free VRAM (depth {depth:3d}): {ms:9.1f} ms   dev peak {dev:5.2f} GB")
            with contextlib.suppress(Exception):
                c.close()
        except Exception as exc:  # noqa: BLE001
            print(f"  {frac:.2f}x free VRAM (depth {depth:3d}): FAILED {type(exc).__name__}: {str(exc)[:80]}")
        del model
        reset()

    print("\n  -> a smooth curve is the goal. A cliff at the fit boundary tells me")
    print("     where the planner should start tiering earlier.")


def main() -> None:
    print(RULE)
    print("TensorTorrent — the four questions I need a GPU to answer")
    print(RULE)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu    : {p.name}  {p.total_memory / 1e9:.2f} GB total, {free_vram() / 1e9:.2f} GB free")
    else:
        print("gpu    : NONE — every experiment will skip")
    print(f"torch  : {torch.__version__}")

    for fn in (experiment_1, experiment_2, experiment_3, experiment_4):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - one failure must not hide the rest
            print(f"\n  EXPERIMENT ABORTED: {type(exc).__name__}: {exc}"[:200])
        reset()

    print(f"\n{RULE}\ndone — paste the output back\n{RULE}")


if __name__ == "__main__":
    main()
