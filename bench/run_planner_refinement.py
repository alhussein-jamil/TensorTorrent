"""Microbenchmark local-search plan copying without requiring real hardware."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

from tensortorrent.planner.local_search import rebalance_partitions, refine_prefetch_distance
from tensortorrent.planner.maximal import ExecutionPlan, Placement


def _plan(size: int) -> ExecutionPlan:
    placements = [
        Placement(
            region_id=f"region-{index}",
            device=f"gpu{index % 2}",
            backend_id="virtual",
            dtype="float32",
            kernel_id="compute",
            estimated_latency_s=1.0 if index % 2 == 0 else 0.1,
            depends_on=(f"region-{index - 1}",) if index else (),
            output_bytes=4096,
            state_bytes=8192,
        )
        for index in range(size)
    ]
    return ExecutionPlan(
        graph_name="planner-bench",
        fingerprint="bench",
        objective="latency",
        placements=placements,
        decisions=[],
        devices_used=("gpu0", "gpu1"),
        communication_backend="host_staged",
        predicted_latency_s=float(size),
        notes=["benchmark"],
    )


def _median_us(operation: Callable[[], object], iterations: int, samples: int = 9) -> float:
    timings: list[float] = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            operation()
        timings.append((time.perf_counter_ns() - start) / iterations / 1_000)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1_000)
    args = parser.parse_args()
    plan = _plan(args.placements)

    print(f"placements={args.placements} iterations={args.iterations}")
    print(
        f"refine_prefetch_distance median_us={_median_us(lambda: refine_prefetch_distance(plan, 2), args.iterations):.3f}"
    )
    print(f"rebalance_partitions median_us={_median_us(lambda: rebalance_partitions(plan), args.iterations):.3f}")


if __name__ == "__main__":
    main()
