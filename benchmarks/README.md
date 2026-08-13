# Benchmarks

Capacity and fit-in-VRAM measurements for TensorTorrent.

**Results:** [evidence/](evidence/) (report + figures) · [raw JSON](evidence/raw/) · [detailed tables](evidence/REPORT.md)

```bash
uv sync --extra dev --extra bench
python -m benchmarks.smoke
python -m benchmarks.public --suite all --out benchmarks/results/current
# optional hard suite:
# python -m benchmarks.public --suite hard --out benchmarks/results/current-hard
```

Freeze a clean measured run, then refresh the report:

```bash
python -m benchmarks.tooling.freeze --src benchmarks/results/current --dst benchmarks/evidence/raw
python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence
```

| Path | Role |
| --- | --- |
| `suites/` | public suite implementations |
| `tooling/` | harness, freeze, figure render |
| `micro/` | optional microbenches (not public evidence) |
| `results/` | ephemeral local runs (gitignored) |
| `evidence/` | published report, figures, and raw JSON |

Qwen3-8B numbers are a **fixed-shape logits forward** (`seq_len=16`), not generation.
Optional generate suite (not in ``all``): ``python -m benchmarks.public --suite generate``
(static padded greedy vs HF/Accelerate KV ``generate()``).
Methodology notes: [docs/product/benchmarks.md](../docs/product/benchmarks.md).
