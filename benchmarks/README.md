# Benchmarks

TensorTorrent’s public benchmarks measure **capacity** and **fit-in-VRAM overhead**
on a single host — not “fastest framework” claims.

**Start here:** [evidence/v0.3.1/README.md](evidence/v0.3.1/README.md) (report + figures).
**Raw JSON:** [evidence/v0.3.1/raw/](evidence/v0.3.1/raw/).

```text
benchmarks/
  README.md           ← you are here
  public.py / smoke.py
  suites/             public suite implementations
  tooling/            harness, freeze, report, figure renderer
  micro/              optional microbenchmarks (not public evidence)
  evidence/v0.3.1/    frozen 0.3.1 snapshot
    README.md         polished report
    figures/          publication SVG/PNG
    raw/              immutable environment + suite JSON
  results/            ephemeral local runs (gitignored)
```

## What is measured

| Suite | Question |
| --- | --- |
| Fit | Overhead when the model already fits one GPU |
| Crossover | Resident → Transfer/Evict as size approaches / exceeds VRAM |
| DeepMLP / Qwen | Beyond-VRAM fixed-shape forwards |
| Budget | Latency vs absolute `vram_budget_bytes` |
| Hetero | GPU+CPU placement marker (multi-GPU unmeasured here) |

**MEASURED** on the frozen host: RTX 3070 Ti Laptop (~8 GiB), 61 GiB RAM, PyTorch 2.13.
Qwen3-8B row is a **fixed-shape logits forward** (`seq_len=16`) — not autoregressive generation.

## Central result

Qwen3-8B BF16 ≈ **16.38 GB** parameters on ~**8 GiB** VRAM: TensorTorrent completes the
fixed-shape forward with peak allocated VRAM ≈ **1.33 GB** by streaming / Transfer–Evict.
When the model fits comfortably, native PyTorch is generally faster.

## Reproduce

```bash
uv sync --extra dev --extra bench
python -m benchmarks.smoke
python -m benchmarks.public --suite all   # heavy; prefers one suite at a time
```

Freeze + refresh the polished report (clean git tree required for freeze):

```bash
python -m benchmarks.tooling.freeze --src benchmarks/results/<run> --dst benchmarks/evidence/v0.3.1/raw
python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence/v0.3.1
```

Microbenchmarks (planner timing, etc.): `python -m benchmarks.micro.planner_native_bench`.

Methodology detail: [docs/product/benchmarks.md](../docs/product/benchmarks.md).
