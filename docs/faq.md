# FAQ

## Why does `doctor` say CUDA is unsupported on my laptop?

No usable CUDA runtime on that host. Status is `unsupported_capability`, not a
successful GPU validation. Run `streamcompiler validate-hardware` on the target
machine.

## Will the planner always use every GPU?

No. A device is included only when it improves the selected objective after
sync/transfer cost. Reasons appear in `compiled.explain()`. Devices whose
working set exceeds allocatable memory are hard-excluded.

## Can I mix NVIDIA and AMD GPUs?

Not as one PyTorch process today. The resource graph can represent mixed-vendor
host-staged links; real mixed-vendor execution needs separate workers per backend
(planned, not shipped).

## Does portable compilation require GPUs?

No. Portable artifacts are hardware-independent. Specialization runs per host.

## Why is StreamCompiler slower than eager on a tiny model?

Fixed schedule dispatch (flatten, validate, instruction DAG). Small GEMMs are
dominated by that overhead; larger ones approach eager. See README benchmarks.
Capacity under a RAM budget is the main win today, not micro-latency.

## Can I call with a different batch size?

No. Example-input shapes/dtypes are fixed; mismatches raise
`UnsupportedFeatureError`. Compile per shape. Dynamic buckets are on the roadmap.

## Can I train through a compiled module?

Default path uses `torch.inference_mode`. With `allow_training=True`, forward uses
the partitioned live `graph_module` so `backward()` works — an autograd-compatible
fallback, **not** heterogeneous schedule training.

## How do I see a real execution timeline?

```python
compiled(x)
compiled.visualize("run.html", measured=True)
```

Default `visualize(path)` is analytic simulation (`simulated=True`) of the same
`ExecutableSchedule` instruction DAG.

## Is accelerator execution validated on CI / this VM?

No. Heterogeneous tests use deterministic `mock_accel` (virtual streams/events).
That validates schedule, residency, and overlap semantics — not CUDA/ROCm/multi-GPU
DMA. See [deployment.md](deployment.md) for production validation steps.

## What does cancel do?

`request_cancel()` stops dispatching new schedule instructions. In-flight
Compute/Transfer work drains, then `ExecutionCancelled` is raised. It is not a
hard kill of running kernels.
