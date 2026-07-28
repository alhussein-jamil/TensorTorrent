# Contributing

1. Prefer small, tested changes that improve correctness, maintainability, or measured performance.
2. Keep vendor-specific logic inside `backends/` and `communication/`.
3. Never encode assumptions from the current development machine into the planner.
4. Add unit/integration coverage for planner, discovery, and validation changes.
5. Run:

```bash
pytest -q
streamcompiler doctor
ruff check src tests
```

6. Do not present simulated or fabricated numbers as measured hardware results.
