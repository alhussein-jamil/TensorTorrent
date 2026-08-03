"""Demonstrate disk-backed parameter streaming under a tight RAM budget.

Prints the metrics the CPU milestone requires: disk bytes, cache behaviour,
timed I/O overlap with compute, stalls, peak resident parameters, latency, and
numerical error versus eager PyTorch.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

import tensortorrent as tt


class Deep(nn.Module):
    def __init__(self, width: int = 256, layers: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.head = nn.Linear(width, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x)


def main() -> None:
    torch.manual_seed(0)
    model = Deep().eval()
    x = torch.randn(16, 256)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    budget = max(total // 5, model.layers[0].weight.numel() * 4 * 2)
    print(f"parameter_bytes={total} ram_budget_bytes={budget}")

    with torch.no_grad():
        expected = model(x)

    compiled = tt.compile(
        model,
        (x,),
        config=tt.CompileConfig(
            ram_budget_bytes=budget,
            prefetch_distance=1,
            max_region_nodes=2,
        ),
    )
    assert compiled.executor.parameter_store.stats()["kind"] == "streaming"

    # Warmup so page cache and threads settle before the timed call.
    for _ in range(2):
        compiled(x)

    start = time.perf_counter()
    with torch.no_grad():
        actual = compiled(x)
    latency_s = time.perf_counter() - start
    report = compiled.last_execution_report()
    stats = report["parameter_store"]
    err = (actual - expected).abs().max().item()

    print("--- streaming run ---")
    print(f"latency_s={latency_s:.6f}")
    print(f"max_abs_err={err:.3e}")
    print(f"bytes_read={stats['bytes_read']}")
    print(f"reads={stats['reads']} cache_hits={stats['cache_hits']} cache_misses={stats['cache_misses']}")
    print(f"duplicate_reads_avoided={stats['duplicate_reads_avoided']}")
    print(f"prefetch_submitted={stats['prefetch_submitted']} prefetch_hits={stats['prefetch_hits']}")
    print(f"evictions={stats['evictions']} waits_for_prefetch={stats['waits_for_prefetch']}")
    print(f"io_time_s={stats['io_time_s']:.6f}")
    print(f"io_overlapped_with_compute_s={stats['io_overlapped_with_compute_s']:.6f}")
    print(f"exposed_io_s={stats['exposed_io_s']:.6f}")
    print(f"acquire_stall_s={stats['acquire_stall_s']:.6f}")
    print(f"peak_resident_bytes={stats['peak_resident_bytes']} budget_bytes={stats['budget_bytes']}")
    print(f"peak_activation_bytes={report['peak_activation_bytes']}")
    storage = compiled.specialized.profile.get("storage", {})
    if storage:
        print(f"measured_pack_pread_MiB_s={storage['bytes_per_s'] / (1 << 20):.1f} (block={storage['nbytes']} bytes)")
    for note in compiled.specialized.plan.notes:
        if "storage_pread" in note or "region_costs" in note:
            print(f"plan_note: {note}")
    torch.testing.assert_close(actual, expected)
    print("numerical_match=ok")
    compiled.close()


if __name__ == "__main__":
    main()
