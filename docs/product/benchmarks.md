# Benchmarks

TensorTorrent's benchmark suite measures two different questions:

1. **overhead on workloads that already fit one device**, and
2. **feasibility/performance when placement or memory hierarchy is required**.

Evidence labels used below:

| Label | Meaning |
| --- | --- |
| **MEASURED** | Numbers taken on the recorded host in this document |
| **SIMULATED** | Discrete-event / planner estimates only (not wall-clock) |
| **SUPPORTED BUT UNMEASURED** | Code path exists; this host lacked the hardware or run |
| **PLANNED** | Intended coverage; not claimed |

Never treat SIMULATED or PLANNED rows as published performance.

## Reproduce

```bash
uv sync --extra dev --extra bench

# Fast development smoke
python -m benchmarks.run --smoke

# Full public suite (JSON under benchmarks/results/<timestamp>/)
python -m benchmarks.run --suite all

# Individual suites
python -m benchmarks.run --suite fit --device cuda
python -m benchmarks.run --suite beyond_vram --vram-multiple 1.5
python -m benchmarks.run --suite pressure
python -m benchmarks.run --suite scaling
python -m benchmarks.run --suite hetero
```

Focused microbenches remain under `bench/` (`compare_baselines.py`, `planner_native_bench.py`, `perf_breakdown.py`, `oversized_model.py`). The public entry point is `python -m benchmarks.run`.

## Methodology

- Warm up before timing; synchronize CUDA before/after timed regions.
- Report median (and p95 in JSON); distinguish compile/planning time from steady-state forward time.
- Record host, GPU, RAM, torch/CUDA versions, and git commit in `environment.json`.
- Peak device memory via `torch.cuda.max_memory_allocated`; host via `ru_maxrss` (process lifetime — can include earlier approaches in the same process).
- Numerical checks against eager reference (`torch.allclose`, atol=rtol=1e-3).
- Same dtype / batch / model config across approaches in a row.
- Failures and OOMs recorded explicitly.
- GPU-eager OOM probes run in a child process so allocator fragmentation cannot poison TensorTorrent.

## Published snapshot (MEASURED)

Hardware: Intel i7-12700H, 61 GiB RAM, RTX 3070 Ti Laptop GPU (8.22 GiB), PyTorch 2.13.0+cu130. Measured 2026-08-09.

### Model larger than VRAM

DeepMLP, width=4096, depth sized to **1.50×** device VRAM (12.35 GiB parameters). TensorTorrent uses host-resident weights with per-region CUDA Transfer/Evict (`allow_cpu=False`).

| Approach | Median ms | Peak VRAM GB | Peak host GB (RSS) | Result | Evidence |
| --- | ---: | ---: | ---: | --- | --- |
| GPU eager | — | — | — | CUDA OOM | MEASURED |
| Accelerate `device_map="auto"` | — | — | — | CUDA OOM | MEASURED |
| TensorTorrent (CUDA) | 1579 | 0.61 | 38.0 | completed, numerically matched | MEASURED |
| CPU eager | 1036 | 0.08 | 38.0 | completed | MEASURED |

On this laptop, PCIe streaming loses to CPU eager when the model still fits host RAM. That is a capacity result, not a throughput win. Peak host RSS is process-lifetime and includes multiple 12 GiB model copies built during the suite.

### Fit-in-VRAM overhead (CUDA)

`rel` = TensorTorrent / eager. Lower is better for TensorTorrent.

| Workload | Eager ms | `torch.compile` ms | TensorTorrent ms | rel | Peak VRAM MB | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MLP 512×8 | 0.23 | 0.29 | 1.36 | 5.79× | 17 | MEASURED |
| Transformer 256 | 0.24 | 0.27 | 1.14 | 4.78× | 15 | MEASURED |
| MLP 2048×8 | 0.67 | 0.65 | 1.39 | 2.07× | 143 | MEASURED |

When the model fits one GPU, TensorTorrent is slower than eager / `torch.compile` on this host. Overhead shrinks as the forward gets heavier.

### Memory-pressure scaling (MEASURED)

Same ~0.45×-VRAM DeepMLP under artificial `vram_budget_bytes` fractions of physical VRAM:

| Budget | Median ms | Evidence |
| --- | ---: | --- |
| 100% | 19.5 | MEASURED |
| 75% | 19.8 | MEASURED |
| 50% | 20.3 | MEASURED |
| 35% | 401 | MEASURED |
| 25% | 398 | MEASURED |

Below ~50% of device VRAM, Transfer/Evict dominates and latency jumps by ~20× on this PCIe link.

### Model-size scaling around the VRAM wall (MEASURED, smoke widths)

| Size vs VRAM | GPU eager | TensorTorrent | Evidence |
| --- | --- | --- | --- |
| 0.30× | 12.6 ms | 16.3 ms | MEASURED |
| 0.90× | CUDA OOM | 1240 ms | MEASURED |
| 1.10× | CUDA OOM | 1578 ms | MEASURED |

### Multi-GPU / other vendors

| Configuration | Evidence |
| --- | --- |
| 2× GPU on one host | SUPPORTED BUT UNMEASURED (this laptop has 1 GPU) |
| ROCm / XPU | SUPPORTED BUT UNMEASURED here |
| GPU+CPU allowed placement | MEASURED by `--suite hetero` (placement recorded in JSON) |

## Planner microbench (separate)

```bash
uv run python bench/planner_native_bench.py
```

Planner timings are planning-cost measurements, not forward throughput.

## What is still missing from public evidence

- Multi-GPU utilization / interconnect numbers on real multi-GPU hosts
- Recognizable Hugging Face transformers at generation length (not only DeepMLP)
- Head-to-head vs offload stacks when those baselines do not OOM
- Better H2D/compute overlap under beyond-VRAM (PCIe-bound on this laptop)
- Per-approach host-peak RSS (current `ru_maxrss` is process-lifetime)
