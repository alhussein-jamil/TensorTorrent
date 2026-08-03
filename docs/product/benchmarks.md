# Benchmarks

**Read this first.** These are the only numbers TensorTorrent currently has,
and they do not show TensorTorrent winning. They are published anyway, because
a runtime with no published comparison against its alternatives has no basis
for a performance claim.

## What was measured

Single-process **CPU** inference, comparing TensorTorrent against the runtimes
a user would otherwise reach for: eager PyTorch, `torch.compile` (Inductor),
AOTInductor, and ONNX Runtime. Every runtime is checked for numerical agreement
with eager before its timings count. Reproduce with:

```bash
uv run python bench/compare_baselines.py --iters 30 --markdown docs/product/benchmarks.md
```

## What the numbers say

TensorTorrent **does not beat any of these runtimes on single-device CPU.** It
carries a roughly fixed per-forward scheduling cost — the Rust dispatcher plus
Python region callbacks — which behaves predictably:

| workload scale | TensorTorrent vs eager |
| --- | --- |
| small (1–2 ms of work) | ~1.7–2.4x slower |
| medium (7 ms) | ~1.3x slower |
| large (84 ms) | ~1.05x slower |
| very large (750 ms) | ~1.02x, effectively parity |

The overhead amortises as compute per forward grows, which is the expected
shape for a scheduling runtime. The honest conclusion is that on one CPU
device TensorTorrent has nothing to offer over PyTorch: at best it reaches
parity, and ONNX Runtime beats everything on three of the five workloads.

Two things do stand out in TensorTorrent's favour, and both are modest:
numerical agreement with eager is tight everywhere (max deviation 7.15e-07,
generally at float32 rounding), and compile time is far lower than
AOTInductor's on the small and medium workloads.

## What has NOT been measured

Everything that motivates the project:

- multi-GPU placement, and mixed-vendor (CUDA + ROCm + XPU) execution
- NUMA-aware placement on a multi-socket host
- parameter streaming from NVMe for a model larger than host memory
- activation spill under a real memory budget
- concurrent request serving under load

`bench/oversized_model.py` is the benchmark built to settle this: it sizes a
model past the GPU's VRAM and compares TensorTorrent against `accelerate`
`device_map="auto"`, CPU eager, and plain GPU eager (which should OOM — that
failure is the claim being tested). `tools/run_everything.sh` runs it together
with the hardware suite and both device sweeps. Neither has been executed on a
GPU yet.

None of these can run on a CPU-only machine, so none of them are validated by
the table below. Until they are benchmarked on real hardware against
llama.cpp, Accelerate `device_map`, DeepSpeed ZeRO-Inference, and ktransformers,
the heterogeneous claims remain design intent rather than demonstrated results.

---

- python 3.12.13, torch 2.13.0+cu130, threads 1
- Linux-6.12.76-x86_64-with-glibc2.34
- CUDA available: False

Latency is per forward pass, lower is better. `rel` is relative to eager
on the same workload (below 1.00 is faster than eager). `err` is the max
absolute deviation from the eager result.

## mlp_stack_512x8

| runtime | median ms | p95 ms | rel | compile s | err | status |
|---|---|---|---|---|---|---|
| eager | 1.580 | 1.615 | 1.00x | 0.00 | 0.00e+00 | ok |
| torch.compile | 1.676 | 1.823 | 1.06x | 4.70 | 0.00e+00 | ok |
| AOTInductor | 1.503 | 1.564 | 0.95x | 5.89 | 0.00e+00 | ok |
| onnxruntime | 1.069 | 1.177 | 0.68x | 0.36 | 0.00e+00 | ok |
| tensortorrent | 3.756 | 4.057 | 2.38x | 0.70 | 0.00e+00 | ok |

## transformer_block_256

| runtime | median ms | p95 ms | rel | compile s | err | status |
|---|---|---|---|---|---|---|
| eager | 7.104 | 7.317 | 1.00x | 0.00 | 0.00e+00 | ok |
| torch.compile | 7.088 | 7.326 | 1.00x | 0.19 | 4.77e-07 | ok |
| AOTInductor | 6.879 | 7.127 | 0.97x | 5.73 | 4.77e-07 | ok |
| onnxruntime | 9.031 | 9.159 | 1.27x | 0.09 | 7.15e-07 | ok |
| tensortorrent | 9.389 | 9.697 | 1.32x | 1.17 | 4.77e-07 | ok |

## wide_branching_1024

| runtime | median ms | p95 ms | rel | compile s | err | status |
|---|---|---|---|---|---|---|
| eager | 1.976 | 2.205 | 1.00x | 0.00 | 0.00e+00 | ok |
| torch.compile | 2.015 | 2.371 | 1.02x | 0.09 | 4.47e-08 | ok |
| AOTInductor | 1.910 | 1.928 | 0.97x | 5.02 | 4.47e-08 | ok |
| onnxruntime | 0.878 | 0.899 | 0.44x | 0.04 | 2.31e-07 | ok |
| tensortorrent | 3.380 | 3.444 | 1.71x | 0.70 | 0.00e+00 | ok |

## mlp_stack_2048x16

| runtime | median ms | p95 ms | rel | compile s | err | status |
|---|---|---|---|---|---|---|
| eager | 83.964 | 84.960 | 1.00x | 0.00 | 0.00e+00 | ok |
| torch.compile | 83.520 | 87.187 | 0.99x | 1.49 | 0.00e+00 | ok |
| AOTInductor | 84.750 | 91.634 | 1.01x | 9.30 | 0.00e+00 | ok |
| onnxruntime | 75.544 | 81.841 | 0.90x | 2.56 | 9.31e-09 | ok |
| tensortorrent | 87.891 | 89.112 | 1.05x | 4.99 | 0.00e+00 | ok |

## transformer_block_1024

| runtime | median ms | p95 ms | rel | compile s | err | status |
|---|---|---|---|---|---|---|
| eager | 752.414 | 784.729 | 1.00x | 0.00 | 0.00e+00 | ok |
| torch.compile | 749.990 | 771.355 | 1.00x | 4.09 | 7.15e-07 | ok |
| AOTInductor | 766.273 | 783.153 | 1.02x | 6.24 | 7.15e-07 | ok |
| onnxruntime | 954.229 | 999.977 | 1.27x | 1.33 | 9.54e-07 | ok |
| tensortorrent | 768.419 | 775.937 | 1.02x | 35.45 | 7.15e-07 | ok |
