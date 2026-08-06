<p align="center">
  <img src="docs/figures/logo.svg" width="144" alt="TensorTorrent logo">
</p>

<h1 align="center">TensorTorrent</h1>

<p align="center">
  A heterogeneous PyTorch compiler and runtime for one machine with many CPUs,
  GPUs, and memory tiers.
</p>

<p align="center">
  <a href="https://github.com/alhussein-jamil/TensorTorrent/actions/workflows/ci.yml"><img src="https://github.com/alhussein-jamil/TensorTorrent/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://github.com/alhussein-jamil/TensorTorrent/tags"><img src="https://img.shields.io/github/v/tag/alhussein-jamil/TensorTorrent?sort=semver&amp;label=version" alt="Latest version tag"></a>
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB" alt="Python 3.10 to 3.13">
  <img src="https://img.shields.io/badge/rust-1.85%2B-DEA584" alt="Rust 1.85 or newer">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 license"></a>
</p>

TensorTorrent exports a PyTorch model, partitions its graph, places regions
across available compute, and runs the resulting schedule through a Rust data
plane. Parameters can stream from slower storage and activations can spill when
the model exceeds device or host memory.

Python compiles. Rust schedules. One immutable `ExecutableArtifact` describes
the program.

> [!IMPORTANT]
> TensorTorrent is alpha software. The supported target is Linux with Python
> 3.10–3.13 and PyTorch 2.4 or newer. Validate every deployment machine before
> serving production traffic. APIs, artifact formats, and env var names may
> change between releases.

## Installation

Install the PyTorch build you want first (CPU / CUDA / ROCm), then TensorTorrent.
`pip` reuses an already-installed `torch>=2.4` instead of replacing it.

```bash
# Example: CPU torch from the official index
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then TensorTorrent (wheels for CPython 3.10–3.13 on Linux)
pip install tensortorrent
```

For CUDA or ROCm builds, follow
[pytorch.org/get-started](https://pytorch.org/get-started/locally/).
The empty `tensortorrent[cuda]` / `tensortorrent[rocm]` extras are markers only —
they do not install an accelerator torch; bring your own.

Requires Linux and Python 3.10–3.13. Wheels are on
[PyPI](https://pypi.org/project/tensortorrent/) and
[GitHub Releases](https://github.com/alhussein-jamil/TensorTorrent/releases);
other platforms can build from the sdist with a Rust 1.85+ toolchain.

To develop from source use [uv](https://docs.astral.sh/uv/) (see Quick start).

## Quick start

```bash
git clone https://github.com/alhussein-jamil/TensorTorrent.git
cd TensorTorrent
make sync
make doctor
```

Compile a module and compare it with eager PyTorch:

```python
import torch
import torch.nn as nn
import tensortorrent as tt  # import alias: tt

model = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).eval()
x = torch.randn(32, 256)

compiled = tt.compile(model, example_inputs=(x,))
torch.testing.assert_close(compiled(x), model(x), check_device=False)

compiled.save("artifact/")
reloaded = tt.load_compiled("artifact/")
```

Run `uv run python examples/public_api_demo.py` for hardware discovery, compile,
and schedule output in one executable example.

## What it handles

| Area | Implementation |
| --- | --- |
| PyTorch export and graph partitioning | [`python/tensortorrent/compile`](python/tensortorrent/compile) |
| CPU, CUDA, ROCm, Intel XPU, and plugin discovery | [`python/tensortorrent/backends`](python/tensortorrent/backends) |
| Resource budget resolver (host memory, VRAM, CPU, disk) | [`python/tensortorrent/hardware/budget.py`](python/tensortorrent/hardware/budget.py) |
| NUMA-aware host allocation and CPU budget enforcement | [`crates/tt-backend-cpu`](crates/tt-backend-cpu) |
| Scheduling, residency, transfer, stall watchdog, and cancellation | [`crates/tt-runtime`](crates/tt-runtime) |
| Parameter streaming and activation spill | [`crates/tt-storage`](crates/tt-storage) |
| Atomic, checksummed artifact bundles | [`python/tensortorrent/artifact_io.py`](python/tensortorrent/artifact_io.py) |
| Concurrent request serving (HTTP, auth, metrics) | [`python/tensortorrent/serve`](python/tensortorrent/serve) |
| Virtual accelerators for deterministic tests | [`crates/tt-backend-virtual`](crates/tt-backend-virtual) |

The runtime supports NCCL, RCCL, oneCCL, Gloo, and explicit host-staged
collective fallbacks where the installed hardware and libraries allow them.

## Architecture

```mermaid
flowchart LR
    M[PyTorch module] --> E[Export and normalize]
    E --> P[Partition and place]
    P --> A[ExecutableArtifact]
    A --> R[Rust dispatcher]
    R --> C[CPU / GPU regions]
    R --> S[Memory / storage tiers]
```

The Python control plane owns export, normalization, partitioning, region
compilation, public APIs, and diagnostics. The Rust data plane owns the
artifact, schedule, workers, residency, transfers, storage, cancellation, and
telemetry. Torch compute regions may call back into Python; scheduling and data
movement remain in Rust.

See the [architecture guide](docs/architecture/architecture.md) for ownership
boundaries and [backend contracts](docs/architecture/backends.md) for extension
points.

## Module composition

Compile a sequence as one graph to avoid opaque transfers between separately
compiled artifacts:

```python
compiled = tt.compile_modules(
    [encoder, projector, decoder],
    example_inputs=(x,),
    names=["encoder", "projector", "decoder"],
)
```

For branches, joins, structured arguments, or nested outputs, build a
`ModuleGraph` from `ModuleNode`, `GraphInput`, and `NodeOutput`. Invalid names,
forward references, and output paths are rejected before export.

## Opt-in training

Compilation is inference-only by default. Set `allow_training=True` to use the
same heterogeneous schedule with autograd:

```python
config = tt.CompileConfig(allow_training=True)
compiled = tt.compile(model, example_inputs=(x,), config=config)

optimizer = torch.optim.Adam(compiled.parameters())
compiled.train()
optimizer.zero_grad()
loss = compiled(x).sum()
loss.backward()
optimizer.step()
compiled.eval()
```

Training cannot currently be combined with NVMe parameter streaming,
activation spill budgets, or process workers. See the full
[product scope](docs/product/PRODUCT.md) for intentional limits.

## Does it actually work?

On a single device TensorTorrent reaches **eager parity at scale** — matching or
beating PyTorch on large MLPs and transformers. Eligible resident single-region
graphs use the direct path by default. Measured resident CPU+accelerator branch
plans can use the same low-overhead path after synchronized timing beats both
schedule execution and full fusion (`prefer_direct_path`; override with
`TT_DIRECT_PATH=0/1`). The product
focus beyond that is multi-device placement, parameter streaming, and activation
spill.

Measured tables, the same-device harness pin, and open roadmap items live in
[Benchmarks](docs/product/benchmarks.md).

## Resource budgets and guardrails

Every memory limit, CPU count, and disk quota flows through a single resolver
that reads cgroup v2/v1 limits, live OS availability, and explicit config
values — in that precedence order. Containers automatically see their cgroup
limits, not host totals. The resolver provenance is shown by
`tensortorrent doctor`.

See [Resource budgets and guardrails](docs/product/resource_budgets.md) for the
full precedence chain, spill lifecycle, stall watchdog, and worked examples.

## Development

```bash
make sync                 # create the environment and build the native extension
make check                # lint, types, Rust tests, Python tests, doctor
make audit                # cargo-audit (Rust) + pip-audit (Python)
make coverage             # run tests with coverage gate (Python 3.12)
make native-gate          # native extension smoke and execution checks
make hardware-test        # explicit: may consume most available VRAM or spill space
```

On a machine with a GPU, run everything that needs real hardware in one go:

```bash
bash tools/run_everything.sh     # tests + hardware suite + all benchmarks
```

It writes logs, JSON, and a `SUMMARY.md` to `bench-results/<timestamp>/`.
Install the benchmark baselines first with `uv sync --extra bench` so the
ONNX Runtime and Accelerate comparisons run instead of reporting as missing.

CI runs on pull requests and pushes to `main` (not on every feature-branch
push or release tag). The matrix covers Python 3.10 and 3.13 on Linux x86-64
plus ARM64, including a coverage gate and `cargo-audit` / `pip-audit`
dependency audits. Hardware tests stay opt-in because they are target-specific
and resource-intensive.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing planner, discovery, or
backend behavior.

## Repository map

```text
python/tensortorrent/   Python control plane, public API, and serving
crates/tt-*/            Rust IR, runtime, memory, storage, backends, and FFI
tests/                  Unit, integration, end-to-end, property, and hardware tests
docs/                   Product, architecture, deployment, and reference guides
examples/               Small public API programs
bench/                  Runtime and planner comparisons
tools/                  Local quality and native-extension gates
deploy/                 Docker Compose and Kubernetes examples
Dockerfile              CPU-only production container
Dockerfile.cuda         CUDA GPU production container (validate on GPU host before use)
```

## Documentation

- [Product scope](docs/product/PRODUCT.md)
- [Architecture](docs/architecture/architecture.md)
- [Heterogeneous hardware planning](docs/architecture/heterogeneous_hardware.md)
- [Resource budgets and guardrails](docs/product/resource_budgets.md)
- [Benchmarks](docs/product/benchmarks.md)
- [Deployment and target validation](docs/product/deployment.md)
- [FAQ](docs/reference/faq.md)
- [Anti-patterns](docs/reference/anti_patterns.md)

## Versions and releases

Versions follow [Semantic Versioning](https://semver.org/) and release tags use
`vMAJOR.MINOR.PATCH`. The release workflow verifies that Python metadata, Rust
workspace metadata, the public `__version__`, the tag, and the changelog agree.
Pushing a `vMAJOR.MINOR.PATCH` tag builds manylinux wheels, creates the GitHub
Release (notes from `CHANGELOG.md`), and publishes to PyPI; see
[docs/RELEASING.md](docs/RELEASING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
