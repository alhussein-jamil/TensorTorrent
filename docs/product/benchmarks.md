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
# compile-phase breakdown (capture/measure/plan/region_compile/simulate):
make bench-perf
uv run python bench/perf_breakdown.py --device cpu
```

Compile knobs that trade specialize wall time vs plan quality
(see `CompileConfig`):

- `measure_workers` — accelerator measure shards (`0` = auto; CPU always serial)
- `region_compile_workers` — default `1` (serial); parallel Inductor rarely wins under GIL
- `planner_parallel_subsets` — default off; enable when multi-device subset search profiles faster

Specialize profiles expose `profile["specialize_timing"]` for local before/after.

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

### Direct path for resident static plans

`prefer_direct_path` (default on) skips schedule dispatch when there is nothing
dynamic to schedule. This includes one-region plans and CPU+accelerator branch
plans retained only after synchronized compile-time timing beats both the
schedule executor and full fusion.
Tables below still show an explicit `TT_DIRECT_PATH=1` pin for comparison
against older schedule-only runs. Same pin, same machine:

| workload | eager | TensorTorrent | rel |
| --- | --- | --- | --- |
| mlp 512×8 | 0.91 | **0.62** | **0.68×** |
| transformer 256 | 3.64 | 5.27 | 1.45× |
| wide 1024 | 1.46 | **0.52** | **0.35×** |
| mlp 2048×16 | 63.21 | **57.07** | **0.90×** |
| transformer 1024 | 582.69 | 602.90 | 1.03× |

Sub-millisecond eager times move with machine load; absolute TensorTorrent
medians are the stable reading. Set `TT_DIRECT_PATH=0` when schedule telemetry,
mid-forward cancellation, training, or streaming semantics are required.

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

Target: run models that plain GPU eager cannot. Model sized to **1.50×**
device VRAM (12.35 GiB params / 8.22 GiB VRAM). Measured 2026-08-07 on the
same host (`bench/oversized_model.py`).

| approach | median ms | device peak GB | host peak GB | status |
| --- | --- | --- | --- | --- |
| GPU eager | – | – | – | CUDA OOM |
| Accelerate `device_map="auto"` | – | – | – | not installed on this host |
| TensorTorrent (VRAM stream) | 1,115 | 0.61 | 25.08 | ok |
| CPU eager (model fits host RAM) | 297 | 0.00 | 12.68 | ok |

GPU eager OOMs as designed. TensorTorrent keeps weights host-resident and
streams one fused region at a time through VRAM (device peak stays under the
budget). Wall time is dominated by PCIe H2D of the full 12 GiB parameter set
each forward (~1.1 s on this laptop link) — not scheduler tax. CPU eager wins
when the model fits host RAM (DRAM bandwidth ≫ PCIe); the streaming path is
for “run at all under a VRAM cap,” not for beating in-RAM CPU on this shape.

---

## Reading the harness

- **Same-device pin** — TensorTorrent follows `--device`. CPU runs never silently
  place on CUDA.
- **`onnxruntime[CPU-EP]`** — the CPU wheel ignores `CUDAExecutionProvider`;
  GPU sweeps rename that row. Install `onnxruntime-gpu` for a CUDA EP.
- **AOTInductor** needs a system CUDA toolkit (`nvcc` / `CUDA_HOME`).
- **Oversize packs** need scratch disk and host headroom; shortfalls fail closed
  with `schedule infeasible`.

## Capacity paths measured on this host

- Beyond-VRAM parameter streaming (model 1.5× device VRAM, host-resident weights)
- Host-RAM budget streaming via `CompileConfig.ram_budget_bytes` (fail-closed when
  a region cannot fit the budget)
- Concurrent serving with shared capacity leases (host/device/disk) under
  `ModelManager` / `CompiledModule.capacity_ledger`

Run `bash tools/run_everything.sh` after `uv sync --extra bench` to refresh local
tables.
