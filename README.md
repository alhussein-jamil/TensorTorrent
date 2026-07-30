# StreamCompiler

Single-machine heterogeneous **inference** runtime for PyTorch.

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

## Product scope

See [docs/PRODUCT.md](docs/PRODUCT.md). In: single host, many CPU/NUMA, one or many GPUs, streaming + spill, concurrent requests. Out: training-through-schedule, multi-node, untested accelerator claims.

## Layout

```
python/streamcompiler/   # control plane (api, frontend, partitioning, compilation, diagnostics)
rust/sc-*/               # data plane (ir, runtime, memory, storage, backends, python FFI)
server/                  # load / infer / health / readiness / metrics
tests/ benchmarks/ docs/
```

## Install

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain. Native extension required (`streamcompiler._native`).

```bash
uv sync --extra dev
uv run maturin develop --release
uv run make native-gate
uv run pytest -q
```

Activate the env with `source .venv/bin/activate` if you prefer bare commands over `uv run`.

## Status (honest)

| Surface | Label |
| --- | --- |
| CPU NUMA discovery + host buffers (`sc-backend-cpu`) | measured on this host |
| Virtual / mock accelerator path | simulated |
| Rust dispatcher + residency + storage | measured (CPU + virtual) |
| Single-GPU CUDA place / measure / execute (PyTorch path) | measured on NVIDIA when present |
| Real CUDA multi-GPU workers / NCCL collectives | **blocked** — not wired |
| Serving layer | experimental (in-process API; no HTTP yet) |
| Python compute callbacks on hot path | migration — AOT native region launch incomplete |

## Docs

| Doc | Topic |
| --- | --- |
| [PRODUCT](docs/PRODUCT.md) | Scope and readiness labels |
| [architecture](docs/architecture.md) | Ownership boundaries |
| [backends](docs/backends.md) | Backend contracts |
| [deployment](docs/deployment.md) | Target-machine validation |

## License

Apache-2.0
