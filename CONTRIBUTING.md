# Contributing to TensorTorrent

Spans capture, native planning, simulation, runtime, storage, backends. Keep the contracts between layers — don't optimize one in isolation and break another.

## Development setup

```bash
git clone https://github.com/alhussein-jamil/TensorTorrent.git
cd TensorTorrent
make sync
make pre-commit-install
```

`make sync` installs deps and builds the PyO3 extension with the `release-quick` Cargo profile.

## Before opening a pull request

```bash
make check
make native-gate
```

Targeted:

```bash
make test
make cargo-test
make cargo-clippy
make rust-fmt
make coverage
```

Hardware changes need a separate validation pass (not in the generic CI matrix — they can reserve a lot of VRAM/RAM/spill):

```bash
make hardware-test
tensortorrent validate-hardware --output artifacts/validation_report.json
```

## Pull-request expectations

Say what changed, why the old behavior was wrong, what invariant should hold, what tests moved, and before/after numbers for perf work. Keep unrelated cleanup out unless that cleanup is needed to make the change safe.

## Architecture rules

- Keep vendor-specific execution logic behind backend interfaces.
- Do not encode the development machine's topology into planner logic.
- Keep the native Rust planner authoritative for hot placement search.
- Keep one executable schedule representation for simulation and runtime.
- Do not hide transfers from the schedule/residency model.
- Preserve deterministic planner results across worker counts.
- Keep planner/DES parallelism bounded by the configured local worker pool.
- Add regression tests for bugs that cross planner, simulation, or runtime boundaries.
- Do not weaken hardware validation so an unsupported target appears healthy.

See [Architectural anti-patterns](docs/reference/anti_patterns.md).

## Performance changes

Performance changes require measurements. At minimum, separate:

- compile/specialization time,
- planner time,
- simulator time,
- forward latency/throughput,
- memory peaks when relevant.

Use the existing benchmark harness rather than one-off timing code when possible:

```bash
# Public capacity suite (canonical)
uv run python -m benchmarks.smoke
uv run python -m benchmarks.public --suite fit

# Planner / microbenches
uv run python benchmarks/micro/planner_native_bench.py
uv run python benchmarks/micro/perf_breakdown.py --device cpu
uv run python benchmarks/micro/compare_baselines.py --device cpu --iters 50
```

## Style and quality gates

The repository uses Ruff, mypy strict mode, codespell, `cargo fmt`, Cargo checks/Clippy, pytest, and pre-commit hooks. Do not suppress a gate globally to merge a local issue.

## Releases

Version tags use SemVer (`vMAJOR.MINOR.PATCH`). See [docs/RELEASING.md](docs/RELEASING.md).
