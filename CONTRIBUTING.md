# Contributing

1. Prefer small, tested changes that improve correctness, maintainability, or measured performance.
2. Keep vendor-specific logic inside `backends/` and `communication/`.
3. Never encode assumptions from the current development machine into the planner.
4. Add unit/integration coverage for planner, discovery, and validation changes.
5. Run the local gate (matches CI lint/types/tests/doctor):

```bash
python scripts/check.py
# optional package build smoke
python -m build
```

6. Do not present simulated or fabricated numbers as measured hardware results.
7. GPU presence is discovery only; concurrent multi-device execution stays unvalidated
   until a real overlapping run exists on deployment hardware.
