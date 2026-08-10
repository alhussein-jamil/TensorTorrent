# Benchmark methodology

Public results and figures: [benchmarks/evidence/](../../benchmarks/evidence/).
How to run suites: [benchmarks/README.md](../../benchmarks/README.md).

- Warm up; synchronize CUDA around timed regions; report median latency.
- `environment.json` (under `evidence/raw/`): host, GPU, torch/CUDA, package version, git commit, `git_dirty` (must be `false` when frozen as public evidence).
- Peak device bytes: `torch.cuda.max_memory_allocated`.
- Beyond-VRAM GPU eager: **infeasible by parameter footprint** when params exceed VRAM (unless a probe allocates).
- Accelerate rows = the **tested** `device_map="auto"` config only — not every possible Accelerate setup.
- Prefer **TensorTorrent auto** as the user-facing number when both auto and forced-GPU rows exist.
- Child processes for GPU-eager OOM probes and each crossover size.
- Host abort if free RAM ≲ 2.5× weights + 4 GiB.

Evidence labels: **MEASURED** · **SIMULATED** · **SUPPORTED BUT UNMEASURED** · **PLANNED**.

Reproduce current evidence:

```bash
uv sync --extra dev --extra bench
python -m benchmarks.public --suite all --out benchmarks/results/current
python -m benchmarks.tooling.freeze --src benchmarks/results/current --dst benchmarks/evidence/raw
python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence
```
