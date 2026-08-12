# Quickstart

Compile, inspect, run, save, reload.

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

`example_inputs` locks the shapes/dtypes for capture and specialization. Incompatible calls raise instead of running a different graph quietly.

## Run and check numerics

```python
expected = model(x)
actual = compiled(x)
torch.testing.assert_close(actual, expected, check_device=False)
```

Compile-time numerical checks are on by default (`validate_numerics=True`).

## Inspect the selected plan

```python
print(compiled.explain())
```

Devices, placements, transfers, memory choices, planner/sim notes.

Timeline:

```python
compiled.visualize("run.html", measured=True)
```

`measured=True` uses observed timing after a run when available; otherwise it's a simulation of the same schedule.

## Save and reload

```python
compiled.save("artifact/")
reloaded = tt.load_compiled("artifact/")
y = reloaded(x)
```

Artifacts are versioned and checksummed. `load_compiled()` reloads the export and specializes for the current machine. Pass `refresh_artifacts=True` to write the new specialization back into the artifact dir.

## Choose an objective

```python
config = tt.CompileConfig(
    objective=tt.Objective.THROUGHPUT,
    target_inflight_requests=8,
)
compiled = tt.compile(model, example_inputs=(x,), config=config)
```

- `LATENCY` — predicted request completion time
- `THROUGHPUT` — simulated steady-state bottleneck
- `MEMORY` — lower peak among feasible candidates
- `BALANCED` — mix of runtime and memory
- `WEIGHTED` — set `objective_weights` yourself

## Restrict devices

```python
config = tt.CompileConfig(allow_gpu=False)
compiled = tt.compile(model, example_inputs=(x,), config=config)
```

Mixed-vendor is allowed by default when backends exist. Turn it off with `allow_mixed_vendor=False`.

## Compile several modules as one graph

```python
compiled = tt.compile_modules(
    [encoder, projector, decoder],
    example_inputs=(x,),
    names=["encoder", "projector", "decoder"],
)
```

Whole sequence gets one plan instead of opaque boundaries between separately compiled modules. Branches/joins: `ModuleGraph`, `ModuleNode`, `GraphInput`, `NodeOutput`.

## Next steps

- [Architecture](../architecture/architecture.md)
- [Configuration](../reference/configuration.md)
- [Large models](../guides/large-models.md)
- [Deployment](../product/deployment.md)
