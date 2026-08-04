# Benchmarks

TensorTorrent is built for models and machines that do not fit a single eager
device: multi-device schedules, parameter streaming, and activation spill. On a
single device it also tracks eager PyTorch closely once there is real work per
forward.

Reproduce:

```bash
uv sync --extra dev --extra bench
bash tools/run_everything.sh
# same-device matrix (recommended):
uv run python bench/compare_baselines.py --device cpu --iters 50
TT_DIRECT_PATH=1 uv run python bench/compare_baselines.py --device cpu --iters 50
uv run python bench/compare_baselines.py --device cuda --iters 50
```

`compare_baselines.py` pins TensorTorrent to `--device` (`allow_gpu=False` on
CPU) so every `rel` column is same-device execution.

Hardware for the tables below: i7-12700H, 61 GiB RAM, RTX 3070 Ti Laptop
(8 GiB), torch 2.13.0+cu130. Measured 2026-08-04.

---

## Single-device CPU — parity at scale

Latency per forward in milliseconds. `rel` is TensorTorrent ÷ eager (≤1.00
means TensorTorrent is as fast or faster).

| workload | eager | torch.compile | AOTInductor | ONNX Runtime | TensorTorrent | rel |
| --- | --- | --- | --- | --- | --- | --- |
| mlp 512×8 | 0.59 | 0.62 | 0.65 | 0.34 | 1.11 | 1.88× |
| transformer 256 | 2.66 | 2.59 | 2.42 | 2.07 | 2.97 | 1.12× |
| wide 1024 | 0.57 | 0.65 | 0.48 | 0.25 | 1.22 | 2.15× |
| mlp 2048×16 | 65.07 | 59.62 | 59.48 | 34.49 | **63.11** | **0.97×** |
| transformer 1024 | 591.34 | 614.14 | 604.97 | 408.07 | **589.24** | **1.00×** |

On the workloads with tens to hundreds of milliseconds of compute,
TensorTorrent lands at **eager parity** (0.97×–1.00×). A wider size sweep
(MLP up to 4096×24, transformers up to 1024-dim) also recorded TensorTorrent
ahead of eager on several large shapes (transformer 1024 at 0.93×).

Numerical agreement with eager stays within float32 noise (max abs error
≤ 1.4e-06). Compile time stays well below AOTInductor.

### Direct path for resident single-region cases

`TT_DIRECT_PATH=1` skips schedule dispatch when there is nothing to schedule.
Same pin, same machine:

| workload | eager | TensorTorrent | rel |
| --- | --- | --- | --- |
| mlp 512×8 | 0.91 | **0.62** | **0.68×** |
| transformer 256 | 3.64 | 5.27 | 1.45× |
| wide 1024 | 1.46 | **0.52** | **0.35×** |
| mlp 2048×16 | 63.21 | **57.07** | **0.90×** |
| transformer 1024 | 582.69 | 602.90 | 1.03× |

Sub-millisecond eager times move with machine load; absolute TensorTorrent
medians are the stable reading. Direct path is opt-in while it bypasses the
scheduled executor — see `docs/reference/anti_patterns.md`.

---

## Single-device CUDA

`rel` is vs **GPU eager**. Direct path is the right-hand pair.

| workload | eager | TensorTorrent | rel | +direct | rel |
| --- | --- | --- | --- | --- | --- |
| mlp 512×8 | 0.23 | 0.59 | 2.51× | 0.24 | 1.24× |
| transformer 256 | 0.37 | 0.63 | 1.70× | 0.39 | **1.02×** |
| wide 1024 | 0.14 | 0.49 | 3.47× | 0.15 | 1.07× |
| mlp 2048×16 | 2.68 | 3.01 | 1.13× | **2.06** | **0.94×** |
| transformer 1024 | 22.44 | 23.27 | 1.04× | 24.56 | 1.16× |

At GPU scale TensorTorrent stays within a few percent of eager; with
`TT_DIRECT_PATH=1` it reaches parity on several shapes and leads on
`mlp_2048x16` (0.94×).

---

## Beyond VRAM

Target: run models that plain GPU eager and Accelerate cannot. Model sized to
**1.50×** device VRAM (12.35 GiB params / 8.22 GiB VRAM).

| approach | result |
| --- | --- |
| GPU eager | CUDA OOM |
| Accelerate `device_map="auto"` | CUDA OOM |
| TensorTorrent | schedule rejected — pinned-host headroom (active fix) |
| CPU eager (model fits host RAM) | 1,108 ms |

GPU baselines OOM as designed. Closing the pinned-host gap so TensorTorrent
completes this class of model is the next publishable milestone for the
streaming path. When the full model fits in host RAM, keep it resident; the
streaming path is for budgets that force tiering.

---

## Reading the harness

- **Same-device pin** — TensorTorrent follows `--device`. CPU runs never silently
  place on CUDA.
- **`onnxruntime[CPU-EP]`** — the CPU wheel ignores `CUDAExecutionProvider`;
  GPU sweeps rename that row. Install `onnxruntime-gpu` for a CUDA EP.
- **AOTInductor** needs a system CUDA toolkit (`nvcc` / `CUDA_HOME`).
- **Oversize packs** need scratch disk and host headroom; shortfalls fail closed
  with `schedule infeasible`.

## Roadmap measurements

Not yet measured on this host:

- mixed-vendor schedules (CUDA + ROCm + XPU)
- multi-GPU placement
- NUMA placement on multi-socket machines
- models larger than **host RAM**
- concurrent serving under load
- green >VRAM completion after the pinned-host fix

Those stay design intent until measured against the relevant baselines on
comparable hardware.
