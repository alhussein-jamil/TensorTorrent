# Quickstart

This guide covers the normal TensorTorrent lifecycle: compile, inspect, run, save, and reload.

## Compile a model

```python
import torch
import torch.nn as nn
import tensortorrent as tt

model = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512),
).eval()

x = torch.randn(16, 512)
compiled = tt.compile(model, example_inputs=(x,))
```

`example_inputs` fixes the shapes and dtypes used to capture and specialize the artifact. A call with incompatible inputs is rejected rather than silently executing a different graph.

## Run and check numerics

```python
expected = model(x)
actual = compiled(x)
torch.testing.assert_close(actual, expected, check_device=False)
```

Numerical validation during compilation is enabled by default (`validate_numerics=True`).

## Inspect the selected plan

```python
print(compiled.explain())
```

The explanation includes selected devices, placements, transfers, memory decisions, and planner/simulator notes exposed by the specialized plan.

For a timeline:

```python
compiled.visualize("run.html", measured=True)
```

Use `measured=True` after execution when you want observed timing where available. The default visualization uses simulation of the same executable schedule.

## Save and reload

```python
compiled.save("artifact/")
reloaded = tt.load_compiled("artifact/")
y = reloaded(x)
```

Artifacts are versioned and checksummed. `load_compiled()` reloads the exported program and specializes it for the current machine; pass `refresh_artifacts=True` when you want the newly measured specialization written back into the artifact directory.

## Choose an objective

```python
config = tt.CompileConfig(
    objective=tt.Objective.THROUGHPUT,
    target_inflight_requests=8,
)
compiled = tt.compile(model, example_inputs=(x,), config=config)
```

Objectives:

- `LATENCY` — minimize predicted request completion time.
- `THROUGHPUT` — favor the simulated steady-state bottleneck.
- `MEMORY` — prefer lower peak memory among feasible candidates.
- `BALANCED` — combine runtime and memory considerations.
- `WEIGHTED` — use `objective_weights` for explicit latency/memory/throughput weighting.

## Restrict devices

CPU only:

```python
config = tt.CompileConfig(allow_gpu=False)
compiled = tt.compile(model, example_inputs=(x,), config=config)
```

Mixed-vendor placement is allowed by default when compatible backends are available. Disable it explicitly when required:

```python
config = tt.CompileConfig(allow_mixed_vendor=False)
```

## Compile several modules as one graph

For a linear sequence:

```python
compiled = tt.compile_modules(
    [encoder, projector, decoder],
    example_inputs=(x,),
    names=["encoder", "projector", "decoder"],
)
```

This lets TensorTorrent optimize the whole sequence instead of introducing an opaque boundary between independently compiled modules.

For branches, joins, and structured inputs/outputs, use `ModuleGraph`, `ModuleNode`, `GraphInput`, and `NodeOutput`.

## Next steps

- [Architecture](../architecture/architecture.md)
- [Configuration](../reference/configuration.md)
- [Large models](../guides/large-models.md)
- [Hardware validation and deployment](../product/deployment.md)
