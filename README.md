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
  <img src="https://img.shields.io/badge/python-3.10%E2%80%933.13-42D1F5" alt="Python 3.10–3.13">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0"></a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#when-to-use">When to use</a> ·
  <a href="docs/README.md">Docs</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

**TensorTorrent** is a capacity-oriented heterogeneous runtime for PyTorch. It profiles a host, searches placements in a native Rust planner, simulates finalists, then compiles and runs the winner across unequal CPUs, accelerators, memory tiers, and storage.

**Problem:** when a model approaches or exceeds accelerator memory, hand-written `.to(device)` maps and ad-hoc offload become fragile. TensorTorrent can trade PCIe/host transfer bandwidth for execution capacity by streaming state through the accelerator.

> [!NOTE]
> **Alpha.** CPU and virtual backends are covered by CI. Validate accelerators on the target host with `tensortorrent validate-hardware`.

## Headline capacity result (MEASURED)

Qwen3-8B BF16 contains **16.38 GB** of parameters. On an RTX 3070 Ti Laptop GPU with ~8 GiB VRAM, TensorTorrent executes a **fixed-shape logits forward** (`seq_len=16`) while keeping peak allocated GPU memory around **1.33 GB** — by streaming / Transfer–Evict, not by fitting the full model in VRAM.

This is **not** autoregressive generation. Native PyTorch is generally faster when the model fits comfortably in one GPU. Raw evidence (commit `2d7c450`, `git_dirty=false`, package **0.3.1**): [benchmarks/published/2026-08-09/](benchmarks/published/2026-08-09/) · [Benchmarks](docs/product/benchmarks.md).

| Workload | Eager / baseline | TensorTorrent | Peak VRAM | Notes |
| --- | --- | --- | ---: | --- |
| Qwen3-8B bf16 **logits forward** seq16 (16.38 GB) | infeasible by param footprint; tested Accelerate OOM'd; CPU 4861 ms | 2609 ms | 1.33 GB | cosine 0.9997, argmax 15/16; fixed-shape only |
| DeepMLP 1.5× VRAM (12.35 GB) | GPU OOM (probe); tested Accelerate 916 ms; CPU 734 ms | 1580 ms | 0.61 GB | capacity / GPU compute — not a latency win vs CPU/Accelerate here |
| MLP 512×8 (fits) | eager 0.23 ms | 1.09 ms | 17 MB | TT slower when model fits |
| MLP 2048×8 (fits) | eager 0.71 ms | 1.24 ms | 143 MB | overhead shrinks on heavier forwards |

2× GPU / ROCm / XPU / autoregressive generation: **SUPPORTED BUT UNMEASURED** on this machine.

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

**Good fit:** model does not fit one GPU · unequal devices · transfer cost matters · RAM/VRAM budgets · parameter streaming or activation spill · reproducible plans instead of hand-written device maps.

**Not for:** multi-node clusters · exhaustive placement search · “use every detected GPU” · replacing PyTorch kernels · treating discovery as production validation.

Full boundary: [Product scope](docs/product/PRODUCT.md).

## Docs

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
  <sub>TensorTorrent</sub>
</p>
