# Benchmarks

TensorTorrent's benchmark suite measures two different questions:

1. **overhead on workloads that already fit one device**, and
2. **feasibility/performance when placement or memory hierarchy is required**.

Results below are snapshots from one machine. They are not claims that TensorTorrent beats eager PyTorch on every host or model.

## Reproduce

```bash
uv sync --extra dev --extra bench

uv run python bench/compare_baselines.py --device cpu --iters 50
uv run python bench/compare_baselines.py --device cuda --iters 50
uv run python bench/planner_native_bench.py
uv run python bench/perf_breakdown.py --device cpu
```

For the full local run:

```bash
bash tools/run_everything.sh
```

The harness records logs/JSON and a summary under `bench-results/<timestamp>/`.

## Methodology notes

- `compare_baselines.py --device cpu` disables GPU placement so comparisons are same-device.
- CUDA comparisons use GPU eager as the eager baseline.
- ONNX Runtime GPU measurements require `onnxruntime-gpu`; the CPU package is labeled as CPU EP rather than being presented as a GPU result.
- AOTInductor CUDA measurements require a system CUDA toolkit.
- Tiny forward times are sensitive to host load; use repeated medians and interpret sub-millisecond differences cautiously.
- Direct-path results are a distinct execution mode and should not be mixed with schedule-path measurements without labeling them.

## Published snapshot

Hardware for the following tables: Intel i7-12700H, 61 GiB RAM, RTX 3070 Ti Laptop GPU (8 GiB), PyTorch 2.13.0+cu130. Measurements recorded 2026-08-04 unless otherwise noted.

### CPU

Latency per forward in milliseconds. `rel` is TensorTorrent / eager; lower is better.

| Workload | Eager | `torch.compile` | AOTInductor | ONNX Runtime | TensorTorrent | rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP 512×8 | 0.59 | 0.62 | 0.65 | 0.34 | 1.11 | 1.88× |
| Transformer 256 | 2.66 | 2.59 | 2.42 | 2.07 | 2.97 | 1.12× |
| Wide 1024 | 0.57 | 0.65 | 0.48 | 0.25 | 1.22 | 2.15× |
| MLP 2048×16 | 65.07 | 59.62 | 59.48 | 34.49 | 63.11 | 0.97× |
| Transformer 1024 | 591.34 | 614.14 | 604.97 | 408.07 | 589.24 | 1.00× |

The schedule/control overhead is visible on small CPU workloads. Once forward time reaches tens or hundreds of milliseconds, the measured TensorTorrent path approaches eager parity on this host.

### Resident direct path

`prefer_direct_path=True` is the default for eligible resident static plans. The following snapshot used an explicit `TT_DIRECT_PATH=1` pin:

| Workload | Eager | TensorTorrent | rel |
| --- | ---: | ---: | ---: |
| MLP 512×8 | 0.91 | 0.62 | 0.68× |
| Transformer 256 | 3.64 | 5.27 | 1.45× |
| Wide 1024 | 1.46 | 0.52 | 0.35× |
| MLP 2048×16 | 63.21 | 57.07 | 0.90× |
| Transformer 1024 | 582.69 | 602.90 | 1.03× |

These numbers come from a separate run from the first CPU table. Use them to characterize the direct path, not as a cross-table microbenchmark comparison.

### CUDA

`rel` is against GPU eager.

| Workload | Eager | TensorTorrent schedule | rel | TensorTorrent direct | rel |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP 512×8 | 0.23 | 0.59 | 2.51× | 0.24 | 1.24× |
| Transformer 256 | 0.37 | 0.63 | 1.70× | 0.39 | 1.02× |
| Wide 1024 | 0.14 | 0.49 | 3.47× | 0.15 | 1.07× |
| MLP 2048×16 | 2.68 | 3.01 | 1.13× | 2.06 | 0.94× |
| Transformer 1024 | 22.44 | 23.27 | 1.04× | 24.56 | 1.16× |

The direct path removes most scheduler overhead for simple resident plans, but it is not uniformly faster than eager or the schedule path on every workload.

### Model larger than VRAM

Measured 2026-08-07 with a model sized to 1.50× device VRAM: 12.35 GiB of parameters on an 8.22 GiB device.

| Approach | Median ms | Device peak GB | Host peak GB | Result |
| --- | ---: | ---: | ---: | --- |
| GPU eager | — | — | — | CUDA OOM |
| Accelerate `device_map="auto"` | — | — | — | baseline unavailable on that host |
| TensorTorrent VRAM streaming | 1,115 | 0.61 | 25.08 | completed |
| CPU eager | 297 | 0.00 | 12.68 | completed |

This is a capacity result, not a claim that streaming beats CPU execution. On this laptop, moving the parameter set across PCIe each forward dominates runtime; CPU eager is faster because the model still fits host RAM.

## Planner benchmark

```bash
uv run python bench/planner_native_bench.py
```

The benchmark reports native serial versus auto-parallel planner timing, finalist behavior, and scalar/serial-batch/parallel-batch DES timing across synthetic planning problems.

Planner parallelism is intentionally adaptive: `planner_workers=0` permits multicore work, but small searches stay serial when scheduling overhead would dominate.

## What is still missing from public evidence

The current benchmark snapshot does not establish broad multi-device superiority. Stronger external evidence should include several real machines and scenarios such as unequal 2/4-GPU placement, CPU+GPU cooperation, different interconnects, and accelerator-vendor coverage.

Treat those as validation work, not as implied results from the current tables.
