# FAQ

## Is TensorTorrent a PyTorch replacement?

No. PyTorch remains the model/frontend and, for torch-backed regions, the kernel execution environment. TensorTorrent adds capture, placement planning, schedule simulation, residency/data movement, and execution orchestration around that model.

## Does TensorTorrent use every GPU it finds?

No. The planner searches eligible device subsets and can exclude a device when its compute benefit does not offset transfer, contention, or memory cost.

Inspect the result with:

```python
print(compiled.explain())
```

## Does it exhaustively test every possible plan?

No. Native beam/local search shortlists a bounded set of strong placements. TensorTorrent then constructs and simulates schedule variants for those finalists. DES is the final selector, but it does not see every combinatorial plan.

## Why have both a planner and a simulator?

The planner must evaluate many partial states cheaply. The simulator can afford to be more detailed because it only evaluates finalists. This lets the final ranking account for contention, overlap, and residency without making every beam expansion a full simulation.

## Can DES select a plan that the analytical planner ranked second or third?

Yes. Analytical rank, finalist rank, and simulated rank are tracked separately. Detailed simulation can change the winner, including between two placements that use the same device subset.

## Do I need a GPU to use TensorTorrent?

No. CPU-only compilation/execution is supported. Portable artifacts are also independent of a particular accelerator host until specialization.

## Can I mix NVIDIA and AMD devices?

`allow_mixed_vendor=True` permits mixed-vendor eligibility when the required backends and transfer paths exist. Cross-vendor paths may fall back to host staging and can be expensive. Validate and benchmark the actual host before relying on such a plan.

## Why does `doctor` show a GPU but validation fails?

Discovery and validation are different. Discovery means the resource is visible. Validation checks whether the backend can execute the required path correctly on that host/runtime combination.

## Why is TensorTorrent slower on a tiny model?

The schedule/runtime layer has fixed overhead. For eligible resident static plans, the direct path avoids that overhead. TensorTorrent is primarily interesting when placement, memory hierarchy, or multi-resource execution matters.

See [Benchmarks](../product/benchmarks.md).

## Can inputs change shape after compilation?

The compiled artifact is specialized for the captured example-input shapes/dtypes. Incompatible calls raise `UnsupportedFeatureError`. Build separate artifacts for serving shapes that need different specialization.

## Where do output tensors live?

On the device selected by the final schedule. TensorTorrent does not automatically add an unscheduled copy back to CPU. Call `.cpu()` when the caller requires host residency.

## How do I force CPU-only planning?

```python
config = tt.CompileConfig(allow_gpu=False)
compiled = tt.compile(model, example_inputs=(x,), config=config)
```

## How do I force serial planning?

```python
config = tt.CompileConfig(planner_workers=1)
```

`planner_workers=0` is the normal automatic mode.

## Why can streaming be slower than CPU eager?

Streaming trades capacity for data movement. If a model fits host RAM but not VRAM, CPU eager may avoid repeatedly moving a large parameter set over PCIe. TensorTorrent's oversized-model benchmark intentionally shows this case rather than hiding it.

## Why does TensorTorrent refuse a tmpfs spill directory?

Because tmpfs consumes RAM. Spilling activations there does not move memory pressure to persistent storage. Use an NVMe/disk-backed path; override the check only for controlled tests.

## What does `Stalled` mean?

The runtime observed no progress for `stall_timeout_s` while waiting for work/resources. Typical causes are a lost completion, device/kernel hang, or pathologically slow I/O.

Do not increase the timeout until you understand whether the workload is genuinely making no progress.

## Can I train a compiled module?

Only with `CompileConfig(allow_training=True)`. The training path requires resident parameters and is incompatible with activation spill and process workers. It is not an out-of-core distributed training system.

See [Training](../guides/training.md).

## Can I compile several modules together?

Yes. `tt.compile_modules()` handles a linear sequence. `ModuleGraph` handles explicit branches, joins, multiple inputs, and structured outputs.

## How do I visualize execution?

```python
compiled.visualize("run.html", measured=True)
```

Without `measured=True`, visualization uses analytical/simulated timing for the same schedule.

## Does TensorTorrent support Windows?

No. Linux is the supported platform. WSL2 is not a supported production target, and `process_workers` has additional fork-related risks there.
