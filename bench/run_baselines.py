"""Benchmark StreamCompiler against eager PyTorch on the current machine.

Every number printed here is measured in this process. Runs are interleaved so
CPU frequency drift affects both paths equally, and the reported latency is the
minimum over repetitions, which is the most stable statistic for short kernels.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

import streamcompiler as sc


@dataclass
class Comparison:
    name: str
    batch: int
    regions: int
    eager_ms: float
    streamcompiler_ms: float
    ratio: float
    max_abs_err: float
    concurrency: str

    workers: int = 1

    def line(self) -> str:
        return (
            f"{self.name:<22} batch={self.batch:<4} regions={self.regions:<3} "
            f"eager={self.eager_ms:7.3f}ms sc={self.streamcompiler_ms:7.3f}ms "
            f"ratio={self.ratio:5.2f}x err={self.max_abs_err:.2e} workers={self.workers}"
        )


class Mlp(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ManyBranches(nn.Module):
    """Wide independent branches: the shape where region concurrency can pay off."""

    def __init__(self, width: int, branches: int) -> None:
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.branches = nn.ModuleList([nn.Linear(width, width) for _ in range(branches)])
        self.head = nn.Linear(width, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        acc = torch.zeros_like(h)
        for branch in self.branches:
            acc = acc + torch.relu(branch(h))
        return self.head(acc)


class Branching(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.left = nn.Linear(width, width)
        self.right = nn.Linear(width, width)
        self.head = nn.Linear(width, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        return self.head(torch.relu(self.left(h)) + torch.tanh(self.right(h)))


def _time(fn: Callable[[], object], *, warmup: int, iters: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter() - start) / iters)
    return best


def compare(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    *,
    warmup: int = 10,
    iters: int = 20,
    reps: int = 5,
) -> Comparison:
    model = model.eval()
    compiled = sc.compile(model, (x,))

    def run_eager() -> object:
        with torch.no_grad():
            return model(x)

    def run_compiled() -> object:
        return compiled(x)

    # Interleave so neither path benefits from a warmer machine.
    eager = _time(run_eager, warmup=warmup, iters=iters, reps=reps)
    streamed = _time(run_compiled, warmup=warmup, iters=iters, reps=reps)
    eager = min(eager, _time(run_eager, warmup=0, iters=iters, reps=reps))
    streamed = min(streamed, _time(run_compiled, warmup=0, iters=iters, reps=reps))

    with torch.no_grad():
        expected = model(x)
    actual = compiled(x)
    error = float((actual - expected).abs().max())
    decision = compiled.specialized.validation["concurrency"]
    return Comparison(
        name=name,
        batch=int(x.shape[0]),
        regions=len(compiled.regions),
        eager_ms=eager * 1e3,
        streamcompiler_ms=streamed * 1e3,
        ratio=streamed / eager if eager else float("inf"),
        max_abs_err=error,
        concurrency=str(decision["reason"]),
        workers=compiled.executor.max_workers,
    )


def streaming_report(width: int = 256, layers: int = 8, batch: int = 8) -> dict[str, object]:
    """Measure the cost and the memory saving of streaming weights from disk."""
    model = Mlp(width, layers).eval()
    x = torch.randn(batch, width)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    resident = sc.compile(model, (x,))
    streamed = sc.compile(
        model,
        (x,),
        config=sc.CompileConfig(ram_budget_bytes=total // 4, prefetch_distance=1),
    )
    with torch.no_grad():
        expected = model(x)
    error = float((streamed(x) - expected).abs().max())
    resident_ms = _time(lambda: resident(x), warmup=5, iters=10, reps=3) * 1e3
    streamed_ms = _time(lambda: streamed(x), warmup=5, iters=10, reps=3) * 1e3
    stats = streamed._executor.parameter_store.stats()
    return {
        "total_parameter_bytes": total,
        "budget_bytes": stats["budget_bytes"],
        "peak_resident_bytes": stats["peak_resident_bytes"],
        "bytes_read": stats["bytes_read"],
        "reads": stats["reads"],
        "evictions": stats["evictions"],
        "prefetch_submitted": stats["prefetch_submitted"],
        "resident_ms": resident_ms,
        "streaming_ms": streamed_ms,
        "slowdown": streamed_ms / resident_ms if resident_ms else None,
        "max_abs_err": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/benchmarks", help="output directory")
    parser.add_argument("--quick", action="store_true", help="fewer repetitions")
    args = parser.parse_args()

    reps = 2 if args.quick else 5
    cases: list[Comparison] = [
        compare("linear", nn.Linear(512, 512), torch.randn(32, 512), reps=reps),
        compare("mlp_256x4", Mlp(256, 4), torch.randn(32, 256), reps=reps),
        compare("mlp_1024x4", Mlp(1024, 4), torch.randn(64, 1024), reps=reps),
        compare("branching_512", Branching(512), torch.randn(64, 512), reps=reps),
        compare("branching_1024", Branching(1024), torch.randn(128, 1024), reps=reps),
        compare("branches8_1024", ManyBranches(1024, 8), torch.randn(64, 1024), reps=reps),
        compare("branches4_2048", ManyBranches(2048, 4), torch.randn(256, 2048), reps=reps),
    ]

    print(
        f"host: {platform.processor() or platform.machine()} "
        f"torch={torch.__version__} threads={torch.get_num_threads()}"
    )
    for case in cases:
        print(case.line())
    for case in cases:
        if case.workers > 1:
            print(f"  {case.name} concurrency: {case.concurrency}")
    stream = streaming_report()
    print(
        f"streaming: budget={stream['budget_bytes']}B "
        f"peak_resident={stream['peak_resident_bytes']}B "
        f"read={stream['bytes_read']}B in {stream['reads']} reads, "
        f"{stream['slowdown']:.2f}x slower than resident, err={stream['max_abs_err']:.2e}"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": {
            "processor": platform.processor() or platform.machine(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "measured": True,
        "comparisons": [asdict(c) for c in cases],
        "streaming": stream,
    }
    (out / "streamcompiler_vs_eager.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
