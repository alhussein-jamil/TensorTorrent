#!/usr/bin/env python3
"""Compile + forward timing breakdown for TensorTorrent perf work.

Reports specialize phase timers (capture / measure / plan / region_compile /
simulate) plus a short forward latency sample. Use before/after the same
command on the same machine when validating perf changes.

Usage:
    uv run python bench/perf_breakdown.py
    uv run python bench/perf_breakdown.py --smoke
    uv run python bench/perf_breakdown.py --json /tmp/perf.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import torch
import torch.nn as nn

from tensortorrent.config import CompileConfig


class _MLP(nn.Module):
    def __init__(self, width: int = 256, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _median_forward(module: Any, x: torch.Tensor, *, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(2):
            module(x)
        samples: list[float] = []
        for _ in range(max(1, iters)):
            start = time.perf_counter()
            module(x)
            samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def run(*, smoke: bool, iters: int, device: str) -> dict[str, Any]:
    import tensortorrent as tt

    width = 128 if smoke else 256
    depth = 2 if smoke else 4
    model = _MLP(width=width, depth=depth).eval()
    x = torch.randn(4, width)
    allow_gpu = device != "cpu"
    config = CompileConfig(
        allow_cpu=True,
        allow_gpu=allow_gpu,
        measure_regions=True,
        use_torch_compile=not smoke,
        prefer_direct_path=True,
    )
    t0 = time.perf_counter()
    compiled = tt.compile(model, example_inputs=(x,), config=config)
    compile_s = time.perf_counter() - t0
    timing = dict(compiled.specialized.profile.get("specialize_timing") or {})
    planner = dict(compiled.specialized.profile.get("planner_search") or {})
    forward_s = _median_forward(compiled, x, iters=iters)
    eager = _MLP(width=width, depth=depth).eval()
    eager.load_state_dict(model.state_dict())
    eager_s = _median_forward(eager, x, iters=iters)
    report = {
        "workload": f"mlp {width}x{depth}",
        "device_pin": device,
        "compile_wall_s": compile_s,
        "specialize_timing": timing,
        "planner_search": {
            "states_expanded": planner.get("states_expanded"),
            "candidate_subsets": planner.get("candidate_subsets"),
            "parallel_subsets": planner.get("parallel_subsets"),
            "local_improvements": planner.get("local_improvements"),
        },
        "forward_median_s": forward_s,
        "eager_median_s": eager_s,
        "rel_vs_eager": (forward_s / eager_s) if eager_s > 0 else None,
        "strategy": compiled.specialized.plan.strategy,
        "devices_used": list(compiled.specialized.plan.devices_used),
    }
    close = getattr(compiled, "close", None)
    if callable(close):
        close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Tiny model / fewer iters")
    parser.add_argument("--iters", type=int, default=0, help="Forward samples (default 5/15)")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--json", type=str, default="", help="Write report JSON path")
    args = parser.parse_args()
    iters = args.iters or (3 if args.smoke else 11)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    report = run(smoke=args.smoke, iters=iters, device=device)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    # Soft gate for smoke: specialize timing must be present and compile must finish.
    if args.smoke:
        timing = report.get("specialize_timing") or {}
        if "total_s" not in timing:
            raise SystemExit("smoke failed: specialize_timing.total_s missing")
        if report["compile_wall_s"] <= 0:
            raise SystemExit("smoke failed: compile_wall_s not positive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
