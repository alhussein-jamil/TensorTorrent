# Contributing to TensorTorrent

TensorTorrent spans PyTorch graph capture, native planning, simulation, runtime scheduling, storage, and hardware backends. Changes should preserve the contracts between those layers rather than optimize one layer in isolation.

## Development setup

```bash
git clone https://github.com/alhussein-jamil/TensorTorrent.git
cd TensorTorrent
make sync
make pre-commit-install
```

`make sync` installs development dependencies and builds the PyO3 extension with the faster `release-quick` Cargo profile.

## Before opening a pull request

Run the architecture-neutral gate:

```bash
make check
make native-gate
```

Useful targeted commands:

```bash
make test
make cargo-test
make cargo-clippy
make rust-fmt
make coverage
```

Hardware changes require target hardware validation as a separate step:

```bash
make hardware-test
tensortorrent validate-hardware --output artifacts/validation_report.json
```

Hardware tests are intentionally not part of the generic CI matrix because they may reserve substantial VRAM, RAM, or spill space.

## Pull-request expectations

A good PR explains:

1. the behavior being changed,
2. why the existing behavior is insufficient,
3. the invariant that should hold afterward,
4. tests added or changed,
5. before/after measurements for performance work.

Keep unrelated cleanup out of functional changes unless the cleanup is required to make the change safe.

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
uv run python bench/planner_native_bench.py
uv run python bench/perf_breakdown.py --device cpu
uv run python bench/compare_baselines.py --device cpu --iters 50
```

## Style and quality gates

The repository uses Ruff, mypy strict mode, codespell, `cargo fmt`, Cargo checks/Clippy, pytest, and pre-commit hooks. Do not suppress a gate globally to merge a local issue.

## Releases

Version tags use SemVer (`vMAJOR.MINOR.PATCH`). See [docs/RELEASING.md](docs/RELEASING.md).
