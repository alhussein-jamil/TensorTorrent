# Contributing

1. Small, tested changes. Prefer correctness and measured performance.
2. Vendor logic stays in `backends/` and `communication/`.
3. Do not bake the development host into the planner.
4. Cover planner, discovery, and validation changes with tests.
5. Local gate:

```bash
python scripts/check.py
make native-gate
```

6. Never present simulated numbers as measured hardware results.
