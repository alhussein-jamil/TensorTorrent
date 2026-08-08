# CLI reference

The `tensortorrent` command provides host diagnostics, profiling, specialization, and validation. `tensortorrent serve` dispatches to the serving CLI.

## `doctor`

```bash
tensortorrent doctor [--full] [--json PATH]
```

Reports backend/resource readiness and resolved budgets. `--full` runs extended probes. `--json` writes a machine-readable report.

## `profile`

```bash
tensortorrent profile [--all-resources] [--devices ID ...] [--output DIR]
```

Discovers and benchmarks selected machine resources. Default output is `artifacts/profile`.

## `validate-hardware`

```bash
tensortorrent validate-hardware [--stress] [--overnight] [--output PATH]
```

Runs the target-hardware validation suite. Default report path is `artifacts/validation_report.json`.

`--overnight` selects the extended soak path and implies stress work.

## `benchmark-topology`

```bash
tensortorrent benchmark-topology [--output PATH]
```

Writes a discovered/measured topology matrix. Default output is `artifacts/topology.json`.

## `autotune`

```bash
tensortorrent autotune MODEL_ARTIFACT \
  [--objective latency|throughput|memory|balanced|weighted] \
  [--profile] [--force] [--no-mixed-vendor] [--cpu-only]
```

Specializes a portable artifact for the current machine and caches the specialized result.

## `serve`

```bash
tensortorrent serve --artifact artifact/ --listen 127.0.0.1:8080
```

Equivalent serving entry point:

```bash
tensortorrent-serve --artifact artifact/ --listen 127.0.0.1:8080
```

Key options:

- `--artifact PATH`
- `--model-id ID`
- `--concurrency N`
- `--listen HOST:PORT`
- `--devices ID[,ID...]`
- `--health`
- `--metrics`
- `--allow-empty` for network diagnostics without a loaded artifact

See [Deployment](../product/deployment.md) for the service contract and environment configuration.
