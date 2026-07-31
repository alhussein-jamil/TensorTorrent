# Contributing

1. Small, tested changes. Prefer correctness and measured performance.
2. Vendor logic stays in `backends/` (including `backends/communication.py`).
3. Do not bake the development host into the planner.
4. Cover planner, discovery, and validation changes with tests.
5. Local gate (after `uv sync --extra dev` + `uv run maturin develop --release`):

```bash
uv run make pre-commit-install   # once per clone
uv run make pre-commit           # hooks + clippy
uv run python tools/check.py
uv run make native-gate
```

Pre-commit covers whitespace, YAML/TOML/JSON, private keys, Ruff, codespell,
mypy, `cargo fmt`, `cargo check`, and (on push) `cargo clippy`.
