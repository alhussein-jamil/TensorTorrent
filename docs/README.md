# TensorTorrent documentation

Compile first, then open architecture docs only when needed.

[Install](getting-started/installation.md) · [Quickstart](getting-started/quickstart.md) · [Architecture](architecture/architecture.md) · [FAQ](reference/faq.md) · [README](../README.md)

## Start here

| Document | Use it for |
| --- | --- |
| [Installation](getting-started/installation.md) | wheels or source build |
| [Quickstart](getting-started/quickstart.md) | compile, inspect, save, reload |
| [Configuration](reference/configuration.md) | `CompileConfig` knobs |
| [FAQ](reference/faq.md) | common failure modes |

## Architecture

| Document | Scope |
| --- | --- |
| [Architecture](architecture/architecture.md) | portable vs specialize pipeline |
| [Planner](architecture/planner.md) | search, top-K, DES |
| [Runtime](architecture/runtime.md) | schedule, residency, cancel |
| [Heterogeneous hardware](architecture/heterogeneous_hardware.md) | resource graph |
| [Backends](architecture/backends.md) | built-in and plugins |

## Guides

| Document | Scope |
| --- | --- |
| [Large models](guides/large-models.md) | budgets, streaming, spill |
| [Training](guides/training.md) | opt-in autograd path |
| [Deployment](product/deployment.md) | serving and validation |
| [Resource budgets](product/resource_budgets.md) | cgroups and headroom |

## Project

| Document | Scope |
| --- | --- |
| [Product scope](product/PRODUCT.md) | in/out of scope, support levels |
| [Benchmarks](../benchmarks/README.md) | report → figures → raw evidence |
| [Benchmark methodology](product/benchmarks.md) | how results were measured |
| [Anti-patterns](reference/anti_patterns.md) | contributor invariants |
| [Contributing](../CONTRIBUTING.md) | local checks and PRs |
| [Releasing](RELEASING.md) | version and publish |
| [Figures](figures/README.md) | diagram assets |
