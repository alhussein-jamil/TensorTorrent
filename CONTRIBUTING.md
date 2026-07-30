# Contributing

1. Small, tested changes. Prefer correctness and measured performance.
2. Vendor logic stays in `backends/` and `communication/`.
3. Do not bake the development host into the planner.
4. Cover planner, discovery, and validation changes with tests.
5. Local gate (after `uv sync --extra dev` + `uv run maturin develop --release`):

```bash
uv run python scripts/check.py
uv run make native-gate
```

6. Never present simulated numbers as measured hardware results.
