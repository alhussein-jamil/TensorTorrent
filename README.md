# StreamCompiler

Single-machine multi-CPU / multi-GPU PyTorch runtime — inference-first, with
opt-in schedule training.

Python compiles. Rust runs. One immutable `ExecutableArtifact` is the program.

```python
import torch
import torch.nn as nn
import streamcompiler as sc

model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10)).eval()
x = torch.randn(32, 256)

compiled = sc.compile(model, example_inputs=(x,))
torch.testing.assert_close(compiled(x), model(x), check_device=False)

compiled.save("artifact/")
reloaded = sc.load_compiled("artifact/")
```

### Compose multiple modules

Compile a series as one exported graph and one executable schedule (rather than
chaining compiled artifacts and introducing hidden transfers):

```python
compiled = sc.compile_modules(
    [encoder, projector, decoder],
    example_inputs=(x,),
    names=["encoder", "projector", "decoder"],
)
```

For branches, joins, multiple inputs, structured arguments, or nested outputs,
declare a `ModuleGraph`. Nodes are topologically ordered; `NodeOutput.path`
selects tuple, list, or dictionary values. Graph outputs may themselves be
tuple/list/dict pytrees:

```python
graph = sc.ModuleGraph(
    [
        sc.ModuleNode("encode", encoder, (sc.GraphInput(0),)),
        sc.ModuleNode("left", left_head, (sc.NodeOutput("encode"),)),
        sc.ModuleNode("right", right_head, (sc.NodeOutput("encode"),)),
        sc.ModuleNode(
            "merge",
            merger,
            (sc.NodeOutput("left"), sc.NodeOutput("right")),
            {"mask": sc.GraphInput(1)},
        ),
    ],
)
compiled = sc.compile(graph, example_inputs=(x, mask))
```

Graph construction rejects empty graphs, duplicate names, unknown/forward
references, unsupported argument types, and invalid output references before export.

## What this is

A heterogeneous inference stack: discover host topology (NUMA + GPUs), place
regions across CPU and accelerators, stream or spill when models outgrow device
memory, and serve concurrent requests from one compiled artifact.

See [docs/product/PRODUCT.md](docs/product/PRODUCT.md) for scope.

### Training (opt-in)

Default `compile` is inference-only (`.train()` raises). Pass
`CompileConfig(allow_training=True)` for a normal PyTorch train loop — `.train()`
runs the heterogeneous schedule with autograd; `.eval()` switches back to the
fast inference schedule:

```python
compiled = sc.compile(
    model,
    example_inputs=(x,),
    config=sc.CompileConfig(allow_training=True),
)
opt = torch.optim.Adam(compiled.parameters())
compiled.train()
opt.zero_grad()
loss = compiled(x).sum()
loss.backward()
opt.step()
compiled.eval()  # inference schedule again, with updated weights

# same schedule path, thin loop:
# sc.fit(compiled, batches, optimizer=opt, loss_fn=loss_fn, epochs=1)
```

Incompatible with NVMe parameter streaming, activation spill budgets, and
`process_workers`.

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
make pre-commit-install
make native-gate
make test
# On each deployment target (may use most available VRAM / spill space):
make hardware-test
```

Activate the env with `source .venv/bin/activate` if you prefer bare commands over `uv run`.

## Surface

| Area | Notes |
| --- | --- |
| CPU NUMA discovery + host buffers | `sc-backend-cpu` |
| CUDA / ROCm / Intel XPU placement + execute | Capability-gated PyTorch device backends |
| Multi-device plans + collectives | NCCL / RCCL / oneCCL / Gloo / explicit host-staged fallbacks |
| Extensible backend registry | `streamcompiler.backends` entry points; plugin failures are isolated and reported |
| Rust dispatcher + residency + storage | schedule, transfers, spill |
| Atomic artifact bundles | required checksummed manifest, staged publication, explicit trusted-legacy opt-out |
| Serving | request-scoped cancellation via `streamcompiler serve` / `streamcompiler-serve` |

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
