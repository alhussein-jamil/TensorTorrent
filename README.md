# StreamCompiler

Streaming compiler and runtime for **PyTorch inference**.

Compile a module once. Run it under an explicit memory schedule: regions execute
where measured costs say they should, weights stream from disk when RAM is tight,
and the Rust data plane owns residency, transfers, and release.

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

`compiled` is a real `nn.Module` (`forward`, `state_dict`, `.eval()`, `.to()`, save/load).

<p align="center">
  <img src="docs/figures/pipeline.svg" alt="Compile and specialize pipeline" width="880" />
</p>

## What it does

- **Export & partition** — `torch.export`, region IR, packed weights
- **Specialize** — discover host resources, measure region costs, emit one
  immutable `ExecutableSchedule`
- **Execute** — Rust dispatcher (Prefetch / Load / Transfer / Compute / Evict /
  Release); Python runs Compute kernels and tensor materialization only
- **Stream** — keep peak resident weights under `ram_budget_bytes`
- **Simulate** — same schedule model for analytic makespan (labelled simulated)

Details: [docs/architecture.md](docs/architecture.md).

## Install

Native extension required (`streamcompiler._native`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
maturin develop --release
make native-gate   # public compile path smoke
pytest -q
```

## Configure

```python
from streamcompiler.config import CompileConfig

cfg = CompileConfig(
    ram_budget_bytes=64 << 20,   # stream weights under this ceiling
    prefetch_distance=1,
    use_torch_compile=False,    # optional Inductor; kept only if not slower
    activation_budget_bytes=None,
)
compiled = sc.compile(model, (x,), config=cfg)
```

Machine graph injection (tests / virtual accelerators):

```python
from streamcompiler.backends.mock_accel import make_mock_accel_graph
from streamcompiler.hardware.discovery import discover_resource_graph
from streamcompiler.ir.resource_graph import merge_graphs

machine = merge_graphs(discover_resource_graph(), make_mock_accel_graph())
compiled = sc.compile(model, (x,), machine=machine)
```

## Ops

```bash
streamcompiler doctor --full
streamcompiler profile --all-resources
streamcompiler validate-hardware
streamcompiler benchmark-topology
streamcompiler autotune model_artifact/
```

Validation statuses distinguish *detected* from *validated* — absence of a GPU
is `unsupported` / `skipped`, never silent success. See [docs/deployment.md](docs/deployment.md).

## Benchmarks

```bash
python benchmarks/run_baselines.py
python benchmarks/compare_runtimes.py   # eager vs native; streaming under budget
python benchmarks/run_streaming.py
```

Small models pay fixed dispatch overhead. Streaming trades latency for capacity.
Re-run on your machine before citing numbers.

## Limits

- Static shapes from example inputs (mismatch raises)
- Inference by default; `allow_training=True` is graph-module autograd, not schedule training
- CUDA / ROCm / multi-GPU concurrent execution needs real hardware validation
- Tensor / pipeline parallel is not emitted by the planner yet

## Docs

| Doc | Topic |
| --- | --- |
| [architecture](docs/architecture.md) | Pipeline, crates, schedule model |
| [heterogeneous hardware](docs/heterogeneous_hardware.md) | Resource graph + planning |
| [backends](docs/backends.md) | Execution / communication contracts |
| [deployment](docs/deployment.md) | Target-machine validation |
| [faq](docs/faq.md) | Common questions |
| [anti-patterns](docs/anti_patterns.md) | Explicitly rejected shortcuts |

## License

Apache-2.0
