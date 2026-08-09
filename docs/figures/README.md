# Figures

Hand-authored SVGs match the brand (dark panel, cyan `#58B0D0` → purple `#8058C0`).

| File | Used by |
| --- | --- |
| `pipeline.svg` | README, architecture overview |
| `planner.svg` | planner docs |
| `runtime.svg` | runtime docs |
| `memory.svg` | heterogeneous hardware |
| `logo-banner.png` / `logo-icon.png` | README, docs index |
| `logo.svg` | editable logo source |
| `logo.png` / `social-banner.png` | packaging / social preview |

**Benchmark plots** live with the frozen evidence (not under `docs/figures/`):

[`benchmarks/evidence/v0.3.1/figures/`](../../benchmarks/evidence/v0.3.1/figures/)

Regenerate from raw JSON (no remasure required):

```bash
python -m benchmarks.tooling.render_evidence --evidence benchmarks/evidence/v0.3.1
```
