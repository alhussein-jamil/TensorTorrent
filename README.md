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
  <a href="benchmarks/evidence/">Benchmarks</a> ·
  <a href="#when-to-use">When to use</a> ·
  <a href="docs/README.md">Docs</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

When a model approaches or exceeds accelerator memory, hand-written `.to(device)` maps and ad-hoc offload get fragile. TensorTorrent profiles the host, searches placements in a native Rust planner, simulates finalists, then compiles and runs the winner — trading PCIe/host bandwidth for capacity when that trade makes sense.

> [!NOTE]
> **Alpha.** CPU and virtual backends are covered by CI. Validate accelerators on the target host with `tensortorrent validate-hardware`.

## Benchmarks

Primarily aimed at models that approach or exceed accelerator memory. Small models that fit one GPU are usually faster in native PyTorch — planning/runtime overhead shows up there. Under memory pressure or beyond-VRAM, TensorTorrent can be competitive with host-offload runtimes.

Numbers below: RTX 3070 Ti Laptop (~7.66 GiB VRAM). Full tables, figures, raw JSON: [benchmarks/evidence/](benchmarks/evidence/) · [methodology](docs/product/benchmarks.md).

### Qwen3-8B BF16 logits forward (`seq_len=16`, 16.38 GB params)

Fixed-shape forward only — **not** autoregressive generation.

| Approach | Median ms | Peak VRAM | Notes |
| --- | ---: | ---: | --- |
| GPU eager | — | — | infeasible (params > VRAM) |
| CPU eager | 4131 | 0 | ok |
| **TensorTorrent auto** | **1678** | **6.83 GB** | `transfer_evict` · cosine 0.9997 · argmax 15/16 |
| Accelerate (`device_map=auto`) | 1616 | 6.44 GB | tested config only |

### DeepMLP 1.5× VRAM (12.35 GB params)

| Approach | Median ms | Peak VRAM | Notes |
| --- | ---: | ---: | --- |
| GPU eager | — | — | OOM |
| CPU eager | 446 | 0.08 GB | ok |
| **TensorTorrent auto** | **444** | **0.00 GB** | chose CPU (`direct_export_free`) |
| Accelerate (`device_map=auto`) | 807 | 5.38 GB | tested config only |

### Fit-in-VRAM (native PyTorch wins)

| Workload | Eager ms | TensorTorrent ms | Peak VRAM |
| --- | ---: | ---: | ---: |
| MLP 512×8 | 0.23 | 0.28 | 17 MB |
| Transformer 256 | 0.26 | 0.34 | 20 MB |
| MLP 2048×8 | 0.70 | 0.75 | 146 MB |

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
