# Benchmarks

Capacity and fit-in-VRAM measurements for TensorTorrent.

**Results:** [evidence/v0.3.1/](evidence/v0.3.1/) (report + figures) · [raw JSON](evidence/v0.3.1/raw/)

```bash
uv sync --extra dev --extra bench
python -m benchmarks.smoke
python -m benchmarks.public --suite crossover   # or transformer / fit / …
```

Freeze (clean tree) then refresh the report:

```bash
python -m benchmarks.tooling.freeze --src benchmarks/results/<run> --dst benchmarks/evidence/v0.3.1/raw
python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence/v0.3.1
```

| Path | Role |
| --- | --- |
| `suites/` | public suite implementations |
| `tooling/` | harness, freeze, figure render |
| `micro/` | optional microbenches (not public evidence) |
| `results/` | ephemeral local runs (gitignored) |

Qwen3-8B numbers are a **fixed-shape logits forward** (`seq_len=16`), not generation.
Methodology notes: [docs/product/benchmarks.md](../docs/product/benchmarks.md).
