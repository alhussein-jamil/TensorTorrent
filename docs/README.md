<p align="center">
  <img src="figures/logo-icon.png" width="72" alt="TensorTorrent">
</p>

<h1 align="center">TensorTorrent documentation</h1>

<p align="center">
  Single-host heterogeneous compiler/runtime for PyTorch.<br>
  Start with a working compile, then dig into architecture only when you need it.
</p>

<p align="center">
  <a href="getting-started/installation.md">Install</a> ·
  <a href="getting-started/quickstart.md">Quickstart</a> ·
  <a href="architecture/architecture.md">Architecture</a> ·
  <a href="reference/faq.md">FAQ</a> ·
  <a href="../README.md">README</a>
</p>

---

## Start here

| Document | Use it for |
| --- | --- |
| [Installation](getting-started/installation.md) | installing wheels or building from source |
| [Quickstart](getting-started/quickstart.md) | compiling, inspecting, saving, and loading a model |
| [Configuration](reference/configuration.md) | `CompileConfig`, objectives, planner and memory knobs |
| [FAQ](reference/faq.md) | common behavior and failure modes |

## Architecture

| Document | Scope |
| --- | --- |
| [Architecture](architecture/architecture.md) | end-to-end compile/specialize/run pipeline |
| [Planner](architecture/planner.md) | native search, top-K finalists, DES selection |
| [Runtime](architecture/runtime.md) | schedule execution, residency, storage, cancellation |
| [Heterogeneous hardware](architecture/heterogeneous_hardware.md) | resource graph and transfer model |
| [Backends](architecture/backends.md) | built-in and plugin backend contracts |

## Guides

| Document | Scope |
| --- | --- |
| [Large models](guides/large-models.md) | VRAM/RAM limits, streaming, spill, prefetch |
| [Training](guides/training.md) | opt-in autograd path and its restrictions |
| [Deployment](product/deployment.md) | serving, containers, validation, runbook |
| [Resource budgets](product/resource_budgets.md) | cgroups, headroom, disk and CPU limits |

## Project information

| Document | Scope |
| --- | --- |
| [Product scope](product/PRODUCT.md) | supported problem boundary and non-goals |
| [Benchmarks](product/benchmarks.md) | methodology and published snapshot |
| [Anti-patterns](reference/anti_patterns.md) | invariants contributors should not break |
| [Contributing](../CONTRIBUTING.md) | local checks and pull requests |
| [Releasing](RELEASING.md) | version and publishing workflow |

---

<p align="center">
  <img src="figures/logo-banner.png" alt="TensorTorrent" width="420">
</p>
