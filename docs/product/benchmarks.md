# Benchmarks

Public capacity and fit-in-VRAM evidence for TensorTorrent.

**Read first:** [benchmarks/evidence/v0.3.1/README.md](../../benchmarks/evidence/v0.3.1/README.md)
**Index:** [benchmarks/README.md](../../benchmarks/README.md)
**Raw JSON:** [benchmarks/evidence/v0.3.1/raw/](../../benchmarks/evidence/v0.3.1/raw/)

Evidence labels:

| Label | Meaning |
| --- | --- |
| **MEASURED** | Wall-clock numbers from the frozen host |
| **SIMULATED** | Planner / discrete-event estimates only |
| **SUPPORTED BUT UNMEASURED** | Path exists; not claimed here |
| **PLANNED** | Intended coverage; not claimed |

## Reproduce

```bash
uv sync --extra dev --extra bench
python -m benchmarks.smoke
python -m benchmarks.public --suite fit
python -m benchmarks.public --suite crossover
python -m benchmarks.public --suite transformer --model-id Qwen/Qwen3-8B --seq-len 16
```

Ephemeral JSON: `benchmarks/results/` (gitignored). Freeze + render:

```bash
python -m benchmarks.tooling.freeze --src benchmarks/results/<run> --dst benchmarks/evidence/v0.3.1/raw
python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence/v0.3.1
```

Dirty trees are refused by freeze (`--allow-dirty` is local debug only).

## Methodology

- Warm up before timing; synchronize CUDA around timed regions.
- Report median latency; separate compile/planning time from steady-state forward.
- `environment.json` records host, GPU, RAM, torch/CUDA, package version, **git commit**, and **`git_dirty`** (`false` required for published evidence).
- Peak device memory: `torch.cuda.max_memory_allocated`.
- Numerical checks vs eager reference. BF16 transformer: cosine + argmax thresholds.
- GPU-eager OOM probes and each crossover size run in child processes.
- When parameters exceed VRAM, GPU eager is **infeasible by parameter footprint** (not attempted) unless the suite probes allocation.
- Accelerate rows are the **tested** `device_map="auto"` configuration only.
- Host abort if free RAM < ~2.5× weights + 4 GiB headroom.

## Published snapshot (MEASURED)

Frozen at package **0.3.1**, measured commit `fb503e5`, `git_dirty=false`.
Host: Intel i7-12700H, 61 GiB RAM, RTX 3070 Ti Laptop (8.22 GiB), PyTorch 2.13.0+cu130.

Figures (from raw JSON):

![Crossover latency](../../benchmarks/evidence/v0.3.1/figures/crossover_latency.png)

![Qwen memory footprint](../../benchmarks/evidence/v0.3.1/figures/qwen_memory_footprint.png)

![Budget latency / transfer](../../benchmarks/evidence/v0.3.1/figures/budget_latency_transfer.png)

![Fit-in-VRAM overhead](../../benchmarks/evidence/v0.3.1/figures/fit_overhead.png)

Full tables and interpretation: [evidence report](../../benchmarks/evidence/v0.3.1/README.md).

### Qwen3-8B — fixed-shape logits forward (not generation)

Parameters **16.38 GB** (~1.99× VRAM). Peak TT allocated VRAM **~1.33 GB**. Cosine ≈ 0.9997, argmax 15/16.

### Still unmeasured

- Multi-GPU utilization on real multi-GPU hosts
- Autoregressive generation tokens/s
- Additional Accelerate offload configurations
- ROCm / XPU on this machine

## Planner microbench

```bash
uv run python benchmarks/micro/planner_native_bench.py
```

Planner timings are planning-cost measurements, not forward throughput.
