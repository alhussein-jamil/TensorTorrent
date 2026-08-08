<p align="center">
  <img src="docs/figures/logo.svg" width="112" alt="TensorTorrent logo">
</p>

<h1 align="center">TensorTorrent</h1>

<p align="center">
  Heterogeneous execution planning for PyTorch on a single machine.
</p>

<p align="center">
  <a href="https://github.com/alhussein-jamil/TensorTorrent/actions/workflows/ci.yml"><img src="https://github.com/alhussein-jamil/TensorTorrent/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/tensortorrent/"><img src="https://img.shields.io/pypi/v/tensortorrent" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB" alt="Python 3.10–3.13">
  <img src="https://img.shields.io/badge/platform-Linux-333333" alt="Linux">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0"></a>
</p>

TensorTorrent compiles a PyTorch model for the resources that are actually available on a host. It profiles compute and transfer paths, searches heterogeneous placements in a native Rust planner, simulates the strongest schedule candidates, then compiles and executes the winner.

The target is not “make one GPU faster.” The target is **make the whole machine schedulable**: CPUs, multiple accelerators, host memory, device memory, and storage.

> **Status:** alpha. CPU and virtual backends are covered by architecture-neutral CI. Accelerator support must be validated on the target host with `tensortorrent validate-hardware`.

<p align="center">
  <img src="docs/figures/pipeline.svg" alt="TensorTorrent compilation and execution pipeline" width="100%">
</p>

## Install

Install the PyTorch build appropriate for your machine first, then TensorTorrent:

```bash
pip install torch
pip install tensortorrent
```

Requirements: Linux, Python 3.10–3.13, PyTorch 2.4 or newer.

For development from source:

```bash
git clone https://github.com/alhussein-jamil/TensorTorrent.git
cd TensorTorrent
make sync
make doctor
```

See [Installation](docs/getting-started/installation.md) for CUDA/ROCm/XPU notes and source builds.

## Quick start

```python
import torch
import torch.nn as nn
import tensortorrent as tt

model = nn.Sequential(
    nn.Linear(256, 1024),
    nn.GELU(),
    nn.Linear(1024, 256),
).eval()

x = torch.randn(32, 256)
compiled = tt.compile(model, example_inputs=(x,))

y = compiled(x)
torch.testing.assert_close(y, model(x), check_device=False)

print(compiled.explain())
compiled.save("artifact/")
```

Reload without recompiling the portable graph:

```python
compiled = tt.load_compiled("artifact/")
y = compiled(x)
```

## What TensorTorrent does

- Captures and partitions a PyTorch graph into executable regions.
- Discovers CPU, CUDA, ROCm, Intel XPU, memory, storage, and transfer resources through backend capability interfaces.
- Measures region and transfer performance when profiling is enabled.
- Searches placements in native Rust with bounded multicore parallelism.
- Keeps multiple competitive placements, including alternatives from the same device subset.
- Builds prefetch/staging variants and evaluates them with a Rust discrete-event simulator.
- Models dependencies, serialized resources, transfer contention, memory residency, spill, and overlap.
- Compiles only the final DES-selected region implementations.
- Executes one immutable schedule through the runtime, with parameter streaming and activation spill when required by budgets.

## What it does not claim

TensorTorrent does **not** exhaustively simulate every possible placement. The planner narrows a large search space; DES ranks a bounded finalist set.

Hardware discovery also does **not** mean a backend is production-ready on that host. Validation is explicit and target-specific.

TensorTorrent is currently single-host. Multi-node cluster scheduling is outside the project scope.

## When it is useful

TensorTorrent is most relevant when placement or memory hierarchy matters:

- the model does not fit one GPU,
- several unequal compute devices are available,
- host/device transfer cost materially affects placement,
- RAM or VRAM budgets must be enforced,
- parameters need to stream from slower tiers,
- activations need bounded spill,
- you want a reproducible plan rather than hand-written `.to(device)` placement.

For a resident one-device graph, TensorTorrent can select a direct execution path to avoid paying schedule-dispatch overhead when the scheduler adds no value.

## Architecture at a glance

| Layer | Responsibility |
| --- | --- |
| Python frontend | PyTorch capture, graph normalization, partitioning, public API |
| Hardware layer | discovery, topology, budget resolution, measurement |
| `tt-planner` | native placement search and top-K finalist generation |
| `tt-runtime` simulator | detailed schedule feasibility and objective ranking |
| Region compilation | eager FX / `torch.compile` / AOT selection where configured |
| Rust runtime | schedule dispatch, residency, transfers, events, cancellation, telemetry |
| `tt-storage` | parameter packs, prefetch, cache, activation spill |

The important boundary is deliberate: **Python integrates with PyTorch; Rust owns the hot planning and scheduling machinery.** Torch-backed compute regions may still call Python to execute the region body.

Read [Architecture](docs/architecture/architecture.md), [Planner](docs/architecture/planner.md), and [Runtime](docs/architecture/runtime.md) for the detailed model.

## Configure the objective

```python
config = tt.CompileConfig(
    objective=tt.Objective.THROUGHPUT,
    target_inflight_requests=4,
)
compiled = tt.compile(model, example_inputs=(x,), config=config)
```

Supported objectives are `latency`, `throughput`, `memory`, `balanced`, and `weighted`.

Most users should leave planner concurrency on automatic:

```python
config = tt.CompileConfig(
    planner_workers=0,              # auto
    planner_parallel_subsets=True,  # default
)
```

See the complete [configuration reference](docs/reference/configuration.md).

## Large models and memory budgets

TensorTorrent resolves host memory, device memory, CPU, and spill-disk budgets before specialization. Explicit limits override automatically discovered limits.

```python
config = tt.CompileConfig(
    vram_budget_bytes=6 * (1 << 30),
    ram_budget_bytes=24 * (1 << 30),
    allow_nvme_streaming=True,
)
```

See [Running models under memory pressure](docs/guides/large-models.md) and [Resource budgets](docs/product/resource_budgets.md).

## CLI

```bash
tensortorrent doctor
tensortorrent profile --output artifacts/profile
tensortorrent validate-hardware --output artifacts/validation_report.json
tensortorrent benchmark-topology --output artifacts/topology.json
```

See [CLI reference](docs/reference/cli.md).

## Benchmarks

Benchmark results in this repository are snapshots from a named machine, not universal performance claims. The harness includes eager PyTorch, `torch.compile`, AOTInductor, ONNX Runtime, and Accelerate where installed.

```bash
uv sync --extra dev --extra bench
uv run python bench/compare_baselines.py --device cpu --iters 50
uv run python bench/planner_native_bench.py
```

See [Benchmarks](docs/product/benchmarks.md) for the measured results and methodology.

## Development

```bash
make sync
make check
make native-gate
```

Real-hardware tests are intentionally separate:

```bash
make hardware-test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for repository conventions.

## Documentation

Start at the [documentation index](docs/README.md).

- [Installation](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [Architecture](docs/architecture/architecture.md)
- [Planner](docs/architecture/planner.md)
- [Runtime](docs/architecture/runtime.md)
- [Backends](docs/architecture/backends.md)
- [Large models](docs/guides/large-models.md)
- [Training](docs/guides/training.md)
- [Configuration](docs/reference/configuration.md)
- [Deployment](docs/product/deployment.md)
- [Benchmarks](docs/product/benchmarks.md)
- [FAQ](docs/reference/faq.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
