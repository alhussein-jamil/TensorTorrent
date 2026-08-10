# Benchmarks

Measured capacity and fit-in-VRAM results for TensorTorrent.

TensorTorrent targets models that approach or exceed accelerator memory. Native
PyTorch is expected to be faster for small models that fit comfortably on one
GPU — planning and runtime add overhead there. Beyond VRAM, TensorTorrent
provides capacity and can be competitive with host-offload runtimes.

Host for these numbers: NVIDIA GeForce RTX 3070 Ti Laptop GPU (7.66 GiB) · PyTorch 2.13.0+cu130.
Provenance (commit, packages, raw samples): [`raw/`](raw/).

## Qwen3-8B BF16 — fixed-shape logits forward (`seq_len=16`)

Not autoregressive generation. Parameters **16.38 GB** on **7.66 GiB** physical VRAM.

| Approach | Median ms | Peak VRAM | Notes |
| --- | ---: | ---: | --- |
| GPU eager | — | — | infeasible (params > VRAM) |
| CPU eager | 4131 | 0 | ok |
| **TensorTorrent auto** | **1678** | **6.83 GB** | `transfer_evict` · cosine 0.9997 · argmax 15/16 |
| TensorTorrent forced GPU | 1700 | 6.53 GB | detailed; not the default UX |
| Accelerate (`device_map=auto`) | 1616 | 6.44 GB | tested config only |

![Qwen memory footprint](figures/qwen_memory_footprint.svg)

## DeepMLP — 1.50× VRAM (12.35 GB params)

| Approach | Median ms | Peak VRAM | Notes |
| --- | ---: | ---: | --- |
| GPU eager | — | — | GPU OOM |
| CPU eager | 446 | 0.08 GB | ok |
| **TensorTorrent auto** | **444** | **0.00 GB** | `direct_export_free` · devices `['cpu_numa_0']` |
| TensorTorrent forced GPU stream | 740 | 6.72 GB | detailed |
| Accelerate (`device_map=auto`) | 807 | 5.38 GB | tested config only |

## Model-size crossover (DeepMLP)

![Crossover latency](figures/crossover_latency.svg)

| Size × VRAM | GPU eager | TensorTorrent | Strategy |
| --- | --- | --- | --- |
| 0.50× | fits | 15 ms | `direct` |
| 0.75× | fits | 25 ms | `resident` |
| 0.90× | OOM | 677 ms | `resident` |
| 1.00× | OOM | 207 ms | `transfer_evict` |
| 1.10× | OOM | 278 ms | `transfer_evict` |
| 1.25× | OOM | 478 ms | `transfer_evict` |
| 1.50× | OOM | 695 ms | `transfer_evict` |

## Fit-in-VRAM

When the model fits, native PyTorch is faster:

![Fit-in-VRAM overhead](figures/fit_overhead.svg)

| Workload | Eager ms | TensorTorrent ms | Peak VRAM |
| --- | ---: | ---: | ---: |
| mlp_512x8 | 0.23 | 0.28 | 17 MB |
| transformer_256 | 0.26 | 0.34 | 20 MB |
| mlp_2048x8 | 0.70 | 0.75 | 146 MB |

## Unmeasured here

Autoregressive generation · multi-GPU · other Accelerate configs · ROCm/XPU.

Full tabular dump: [`REPORT.md`](REPORT.md). Methodology: [`docs/product/benchmarks.md`](../../docs/product/benchmarks.md).

## Reproduce

```bash
uv sync --extra dev --extra bench
python -m benchmarks.public --suite all --out benchmarks/results/current
python -m benchmarks.tooling.freeze --src benchmarks/results/current --dst benchmarks/evidence/raw
python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence
```
