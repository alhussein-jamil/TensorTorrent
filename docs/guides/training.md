# Opt-in training

TensorTorrent is inference-first. Training is available only when explicitly enabled.

```python
import torch
import tensortorrent as tt

config = tt.CompileConfig(allow_training=True)
compiled = tt.compile(model, example_inputs=(x,), config=config)

optimizer = torch.optim.Adam(compiled.parameters())
compiled.train()
optimizer.zero_grad()
loss = compiled(x).sum()
loss.backward()
optimizer.step()

compiled.eval()
y = compiled(x)
```

## Execution model

With `allow_training=True`, `.train()` runs the executable **schedule** with autograd enabled (DirectPlan is not used under grad). `.eval()` returns to the inference path using the updated parameters. Persistent device-parameter hoist is inference-only; training keeps host parameters authoritative for the optimizer.

The training path preserves multi-region schedules; it is not implemented as a separate unrelated executor.

## Restrictions

Training is intentionally incompatible with features that would invalidate autograd tensor lifetime/identity assumptions:

- activation spill (`activation_budget_bytes`),
- process workers (`process_workers > 0`),
- NVMe parameter streaming on the training path.

The configuration validates incompatible combinations early.

## When to use it

Use the training mode when you need the same heterogeneous schedule abstraction with ordinary optimizer/backward semantics and resident parameters.

Do not interpret it as a distributed-training system or an out-of-core training engine. Those are outside the current scope.
