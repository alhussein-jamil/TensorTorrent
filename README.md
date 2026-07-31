# StreamCompiler

Single-machine multi-CPU / multi-GPU **inference** runtime for PyTorch.

Python compiles. Rust runs. One immutable `ExecutableArtifact` is the program.

```python
import torch
import torch.nn as nn
import streamcompiler as sc

model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10)).eval()
x = torch.randn(32, 256)

compiled = sc.compile(model, example_inputs=(x,))
torch.testing.assert_close(compiled(x), model(x))

compiled.save("artifact/")
reloaded = sc.load_compiled("artifact/")
```

## What this is

A heterogeneous inference stack: discover host topology (NUMA + GPUs), place
regions across CPU and accelerators, stream or spill when models outgrow device
memory, and serve concurrent requests from one compiled artifact.

See [docs/product/PRODUCT.md](docs/product/PRODUCT.md) for scope.

## Layout

```
python/streamcompiler/   # control plane + serve/
crates/sc-*/             # data plane (IR, runtime, memory, storage, backends, FFI)
tests/                   # unit, integration, e2e, hardware, property, simulation
docs/                    # product, architecture, reference
tools/                   # check, native_gate
bench/                   # runtime comparisons
examples/
```

## Install

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain. Native extension required (`streamcompiler._native`).

```bash
uv sync --extra dev
uv run maturin develop --release
uv run make pre-commit-install
uv run make native-gate
uv run pytest -q
```

Activate the env with `source .venv/bin/activate` if you prefer bare commands over `uv run`.

## Surface

| Area | Notes |
| --- | --- |
| CPU NUMA discovery + host buffers | `sc-backend-cpu` |
| CUDA / ROCm placement + execute | PyTorch device backends |
| Multi-device plans + collectives | NCCL / RCCL / Gloo / host-staged |
| Rust dispatcher + residency + storage | schedule, transfers, spill |
| Serving | `streamcompiler serve` / `streamcompiler-serve` |

## Docs

| Doc | Topic |
| --- | --- |
| [PRODUCT](docs/product/PRODUCT.md) | Scope |
| [architecture](docs/architecture/architecture.md) | Ownership boundaries |
| [backends](docs/architecture/backends.md) | Backend contracts |
| [deployment](docs/product/deployment.md) | Target-machine validation |
| [heterogeneous hardware](docs/architecture/heterogeneous_hardware.md) | Resource graph + planning |
| [faq](docs/reference/faq.md) | Common questions |

## License

Apache-2.0
