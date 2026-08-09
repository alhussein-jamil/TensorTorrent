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
| `benchmarks/*.png` | MEASURED plots from `benchmarks/published/2026-08-09/` |

Edit the `.svg` files directly. Regenerate benchmark PNGs with `python -m benchmarks.public` (requires `matplotlib` from `--extra bench`).
