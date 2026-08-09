<p align="center">
  <img src="docs/figures/logo-banner.png" alt="TensorTorrent" width="560">
</p>

<p align="center">
  <strong>Heterogeneous execution planning for PyTorch on a single machine.</strong><br>
  Profile → plan → simulate → compile → run across CPUs, GPUs, memory, and storage.
</p>

<p align="center">
  <a href="https://github.com/alhussein-jamil/TensorTorrent/actions/workflows/ci.yml"><img src="https://github.com/alhussein-jamil/TensorTorrent/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/tensortorrent/"><img src="https://img.shields.io/pypi/v/tensortorrent?color=8A5CF5" alt="PyPI"></a>
  <a href="https://pypi.org/project/tensortorrent/"><img src="https://img.shields.io/pypi/dm/tensortorrent?color=42D1F5" alt="PyPI downloads"></a>
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.13-42D1F5" alt="Python 3.10–3.13">
  <img src="https://img.shields.io/badge/Rust-native%20planner-DEA584" alt="Rust native planner">
  <img src="https://img.shields.io/badge/platform-Linux-0B0D14" alt="Linux">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0"></a>
  <a href="https://github.com/alhussein-jamil/TensorTorrent/stargazers"><img src="https://img.shields.io/github/stars/alhussein-jamil/TensorTorrent?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#when-to-use">When to use</a> ·
  <a href="docs/README.md">Docs</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

TensorTorrent profiles a host, searches placements in a native Rust planner, simulates the strongest schedules, then compiles and runs the winner — across unequal CPUs, accelerators, memory, and storage.

> [!NOTE]
> **Alpha.** CPU and virtual backends are covered by CI. Validate accelerators on the target host with `tensortorrent validate-hardware`.

<p align="center">
  <img src="docs/figures/pipeline.svg" alt="TensorTorrent compilation and execution pipeline" width="100%">
</p>

## Install

```bash
pip install torch
pip install tensortorrent
```

Linux · Python 3.10–3.13 · PyTorch 2.4+. Source builds and CUDA/ROCm/XPU notes: [Installation](docs/getting-started/installation.md).

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
```

Save/reload, objectives, and multi-module graphs: [Quickstart](docs/getting-started/quickstart.md).

## When to use

**Good fit:** model does not fit one GPU · unequal devices · transfer cost matters · RAM/VRAM budgets · parameter streaming or activation spill · reproducible plans instead of hand-written `.to(device)` maps.

**Not for:** multi-node clusters · exhaustive placement search · “use every detected GPU” · replacing PyTorch kernels · treating discovery as production validation.

Full boundary: [Product scope](docs/product/PRODUCT.md).

## Benchmark snapshot (MEASURED)

Host: RTX 3070 Ti Laptop 8 GiB, 61 GiB RAM, PyTorch 2.13, package **0.3.0**.
Exact commit + `git_dirty=false` in [published snapshot](benchmarks/published/2026-08-09/). Details: [Benchmarks](docs/product/benchmarks.md).

| Workload | Eager / baseline | TensorTorrent | Peak VRAM | Notes |
| --- | --- | --- | ---: | --- |
| DeepMLP 1.5× VRAM (12.35 GB) | GPU OOM (probe); tested Accelerate 899 ms; CPU 1092 ms | 1375 ms | 0.61 GB | Capacity / GPU compute story on this PCIe laptop — not a latency win vs CPU/Accelerate |
| Qwen3-8B bf16 **logits forward** seq16 (16.38 GB) | infeasible by param footprint; tested Accelerate OOM'd; CPU 3287 ms | 2854 ms | 1.33 GB | **Not** autoregressive generation; fixed-shape exportable forward only |
| MLP 512×8 (fits) | eager 0.23 ms | 0.97 ms | 17 MB | TT slower when model fits |
| MLP 2048×8 (fits) | eager 0.70 ms | 1.20 ms | 143 MB | overhead shrinks on heavier forwards |

2× GPU / ROCm / XPU / autoregressive generation: SUPPORTED BUT UNMEASURED on this machine.

## Docs

Start at the [documentation index](docs/README.md).

| | |
| --- | --- |
| Getting started | [Install](docs/getting-started/installation.md) · [Quickstart](docs/getting-started/quickstart.md) |
| Architecture | [Overview](docs/architecture/architecture.md) · [Planner](docs/architecture/planner.md) · [Runtime](docs/architecture/runtime.md) |
| Ops | [Large models](docs/guides/large-models.md) · [Deployment](docs/product/deployment.md) · [FAQ](docs/reference/faq.md) |

```bash
tensortorrent doctor
tensortorrent validate-hardware --output artifacts/validation_report.json
make check   # from a source checkout
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

<p align="center">
  <img src="docs/figures/logo-icon.png" width="48" alt="TensorTorrent icon"><br>
  <sub>Made for mixed machines — not just mixed kernels.</sub>
</p>
