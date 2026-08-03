# Benchmarks

Two things are measured here and they point in opposite directions. Both are
published, because a runtime that shows only its good numbers has not shown
anything.

1. On a single device, TensorTorrent reaches **parity at scale** but is still
   ~2.2× slower on models doing only a millisecond or two of work.
2. On a model larger than VRAM, TensorTorrent **completes where the standard
   alternatives fail outright** — which is the entire reason the project exists.

Reproduce on a machine with a GPU:

```bash
uv sync --extra dev --extra bench
bash tools/run_everything.sh
```

---

## The result that matters: a model larger than VRAM

Hardware: RTX 3070 Ti Laptop, 8 GiB VRAM (**4.7 GiB actually free** — the
desktop session holds the rest), i7-12700H, 61 GiB host RAM, torch 2.13.0+cu130,
driver 595.84.

Model: 12.35 GiB of parameters against 8.22 GiB of VRAM — **1.50×**.

| approach | result | device peak | host peak |
| --- | --- | --- | --- |
| GPU eager | **CUDA OOM** | – | – |
| Accelerate `device_map="auto"` | **CUDA OOM** | – | – |
| **TensorTorrent** | **completes** — 63,932 ms | 0.02 GiB | 25.19 GiB |
| CPU eager | completes — 397 ms | – | 12.68 GiB |

TensorTorrent runs a model that neither plain GPU execution nor the offloading
baseline most people reach for can run at all. That claim is now demonstrated
rather than asserted.

**The caveat matters as much as the result.** This model still fits in 61 GiB of
host RAM, so CPU eager also completes — and is roughly **160× faster** than the
streaming path. On this hardware TensorTorrent's win is narrow: it is the right
tool only when the model exceeds host RAM as well, or when the GPU must be used
for reasons other than speed. The 63.9 s figure is also far slower than the
memory traffic alone justifies, which makes streaming-path throughput the
clearest optimisation target the project has.

The honest publishable comparison is TensorTorrent against Accelerate at sizes
where Accelerate OOMs. Against CPU eager on a RAM-rich host, TensorTorrent
currently loses badly.

## Single device: parity at scale, still behind on small models

CPU, i7-12700H. Latency per forward in milliseconds, lower is better; `rel` is
relative to eager. Measured after the per-forward caching work in `70ab9ff`,
which hoists schedule-invariant scans out of the hot path.

| workload | eager | torch.compile | AOTInductor | ONNX Runtime | TensorTorrent | rel |
| --- | --- | --- | --- | --- | --- | --- |
| mlp 512×8 | 1.57 | 1.62 | 1.48 | **1.07** | 3.50 | 2.22× |
| transformer 256 | 7.40 | 7.53 | 8.16 | 9.49 | 8.67 | 1.17× |
| wide 1024 | 1.87 | 1.98 | 1.94 | **0.91** | 3.31 | 1.77× |
| mlp 2048×16 | 84.55 | 83.78 | 84.46 | **72.41** | 85.59 | 1.01× |
| transformer 1024 | 810.45 | 778.52 | 786.41 | 934.80 | **778.75** | 0.96× |

### What the caching change bought

| workload | before | after |
| --- | --- | --- |
| mlp 512×8 | 2.38× | 2.22× |
| transformer 256 | 1.32× | **1.17×** |
| wide 1024 | 1.71× | 1.77× (within noise) |
| mlp 2048×16 | 1.05× | **1.01×** |
| transformer 1024 | 1.02× | 0.96× |

Real on the medium and large workloads, nothing measurable on `wide_1024`.

**On the largest workload TensorTorrent is at parity with eager, not faster.**
Across three independent repeats it measured 768.8 / 772.1 / 778.8 ms — a 1.4%
spread — while eager ranged 752–840 ms across runs. Eager's own variance
brackets TensorTorrent's result, so the 0.96× is parity, and the honest
observation is that TensorTorrent's run-to-run variance is far tighter. Quoting
"faster than PyTorch" off a single 0.96× reading would not survive a repeat.

The remaining gap is concentrated where it always was: a fixed per-forward cost
that still dominates models doing only 1–2 ms of work (2.2× on `mlp_512x8`).
ONNX Runtime remains fastest on three of five workloads.

Numerical agreement with eager stays tight throughout (max deviation 7.15e-07,
generally float32 rounding), and compile time remains well below AOTInductor's.

## Reading the output

Some results are environmental rather than code failures. The harness labels
them instead of hiding them:

- **`onnxruntime[CPU-EP]`** — the `onnxruntime` CPU wheel silently ignores
  `CUDAExecutionProvider`. During a GPU sweep the row is renamed so a CPU
  measurement never sits unmarked among GPU numbers. Install `onnxruntime-gpu`
  for a real GPU comparison.
- **AOTInductor needs a system CUDA toolkit.** The PyPI torch wheels ship
  headers and libraries but no `nvcc`, so this baseline fails without a distro
  CUDA install. The failure note says exactly that.
- **The oversize sweep needs scratch disk.** Streaming cases write a pack about
  the size of the model; the tests check free space and skip with a clear reason
  rather than spending an hour to fail on a quota error.

## Still not measured

- mixed-vendor execution (CUDA + ROCm + XPU in one schedule) — the boldest claim
  in the project and the one with no evidence at all
- multi-GPU placement; the run above used a single GPU
- NUMA-aware placement on a multi-socket host
- a model larger than **host RAM**, where streaming would be the only option
- concurrent request serving under load

Until those are measured against llama.cpp, DeepSpeed ZeRO-Inference, and
ktransformers on comparable hardware, they remain design intent.
