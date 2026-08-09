# Benchmark methodology

Public results and figures: [benchmarks/evidence/v0.3.1/](../../benchmarks/evidence/v0.3.1/).
How to run suites: [benchmarks/README.md](../../benchmarks/README.md).

- Warm up; synchronize CUDA around timed regions; report median latency.
- `environment.json`: host, GPU, torch/CUDA, package version, git commit, `git_dirty` (must be `false` when frozen).
- Peak device bytes: `torch.cuda.max_memory_allocated`.
- Beyond-VRAM GPU eager: **infeasible by parameter footprint** when params exceed VRAM (unless a probe allocates).
- Accelerate rows = the **tested** `device_map="auto"` config only.
- Child processes for GPU-eager OOM probes and each crossover size.
- Host abort if free RAM ≲ 2.5× weights + 4 GiB.

Evidence labels: **MEASURED** · **SIMULATED** · **SUPPORTED BUT UNMEASURED** · **PLANNED**.
