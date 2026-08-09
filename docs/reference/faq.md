# FAQ

Scope and non-goals: [Product scope](../product/PRODUCT.md). Planner/DES: [Planner](../architecture/planner.md).

## Does TensorTorrent use every GPU it finds?

No. Eligible subsets are searched; a device can be dropped when transfer, contention, or memory cost outweighs its compute. Inspect with `compiled.explain()`.

## Why does `doctor` show a GPU but validation fails?

Discovery ≠ validation. Discovery means visible; validation means the backend executed correctly on that host. Run `tensortorrent validate-hardware`.

## Do I need a GPU?

No. CPU-only works. Portable artifacts stay host-agnostic until specialization.

## Can I mix NVIDIA and AMD?

`allow_mixed_vendor=True` when backends and transfer paths exist. Cross-vendor often stages through host and can be expensive — validate and benchmark the real machine.

## Why is a tiny model slower?

Schedule/runtime has fixed overhead. Eligible resident plans can use the direct path. TensorTorrent pays off when placement, memory hierarchy, or multi-resource execution matter. See [Benchmarks](../product/benchmarks.md).

## Can inputs change shape after compile?

No — specialized for the captured example shapes/dtypes. Incompatible calls raise `UnsupportedFeatureError`. Build separate artifacts for other serving shapes.

## Where do outputs live?

On the device the schedule chose. No automatic copy to CPU — call `.cpu()` if needed.

## Force CPU-only / serial planning

```python
tt.CompileConfig(allow_gpu=False)
tt.CompileConfig(planner_workers=1)  # 0 = automatic
```

## Why can streaming be slower than CPU eager?

Streaming trades capacity for movement. If the model fits RAM but not VRAM, CPU eager may win by avoiding PCIe churn. Oversized-model benches show this on purpose.

## Why refuse a tmpfs spill directory?

tmpfs is RAM. Spill there does not relieve memory pressure. Use disk/NVMe; override only in controlled tests.

## What does `Stalled` mean?

No progress for `stall_timeout_s` while waiting. Typical causes: lost completion, device hang, pathological I/O. Do not raise the timeout until you know the work is truly stuck.

## Training?

Only with `CompileConfig(allow_training=True)`: resident parameters, no activation spill, no process workers. Not out-of-core or multi-node. See [Training](../guides/training.md).

## Several modules?

`tt.compile_modules()` for a linear sequence; `ModuleGraph` for branches/joins and structured I/O.

## Visualize

```python
compiled.visualize("run.html", measured=True)
```

Without `measured=True`, timing is analytical/simulated for the same schedule.

## Windows?

No. Linux only. WSL2 is not a supported production target.
