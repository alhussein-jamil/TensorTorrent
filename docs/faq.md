# FAQ

## Why does `doctor` say CUDA is unsupported on my laptop?

Because this development host has no usable CUDA runtime. That is reported as
`unsupported_capability`, not as a successful GPU validation. Run
`streamcompiler validate-hardware` on the production machine.

## Will the planner always use every GPU?

No. A device is included only when it improves the selected objective after
accounting for synchronization and transfer cost. Exclusion reasons are printed
by `compiled_model.explain()`.

## Can I mix NVIDIA and AMD GPUs?

Not today. The resource graph and planner are built so that mixed-vendor plans are
representable, and missing cross-vendor peer-to-peer links are modelled as
host-staged transfers rather than failing the machine. But a single PyTorch install
generally cannot drive CUDA and ROCm devices at once, so real mixed-vendor execution
needs separate worker processes per backend. That is planned and not implemented.

## Does portable compilation require GPUs?

No. Portable artifacts are hardware-independent. Machine specialization happens
later on each deployment host.

## Why is StreamCompiler slower than eager PyTorch on my small model?

Because each call pays a fixed dispatch cost — flattening inputs, validating them
against the compiled shapes, walking the region dependency graph, and rebinding
region outputs — of roughly 25 microseconds on the development host. For a model
whose eager forward takes 50 microseconds that is a large fraction; for one taking a
millisecond it is noise. Reducing this overhead is the top open item in
[roadmap.md](roadmap.md). The value StreamCompiler adds today is capacity (running a
model whose weights do not fit in RAM) rather than raw latency.

## Can I call the compiled module with a different batch size?

No. Compilation specializes to the example inputs, and a different shape or dtype
raises `UnsupportedFeatureError`. Compile once per shape you need. Dynamic-shape
buckets are on the roadmap.

## Can I train through a compiled module?

Default inference uses `torch.inference_mode` and does not participate in autograd.
Set `CompileConfig.allow_training=True` to run the partitioned live `graph_module`
so `backward()` can populate gradients. That path is an **autograd-compatible
graph-module fallback** — it is not heterogeneous compiled training through the
instruction schedule.

## How do I see a real execution timeline?

After calling the compiled module at least once:

```python
compiled.visualize("run.html", measured=True)
```

That writes measured Chrome/HTML telemetry. The default `visualize(path)` path
is still analytic simulation and is labelled `simulated=True`. When an
`ExecutableSchedule` exists, simulation walks that same instruction DAG (same
instruction IDs as runtime), not a reconstructed placement plan.

## Is accelerator execution validated on this development VM?

No. This repository's CI/dev hosts are CPU-only. Heterogeneous tests use a
**deterministic virtual accelerator** (`mock_accel`) with async stream delays.
That validates scheduling, residency, and overlap semantics — not CUDA, ROCm,
multi-GPU, or real CPU–GPU DMA.

## What must be done on a real CUDA machine later?

1. Run `streamcompiler doctor` and `streamcompiler validate-hardware` with CUDA present.
2. Replace mock stream delays with CUDA streams / CUDA events / pinned host memory
   behind the existing `ExecutionStream` / `BackendEvent` interfaces.
3. Validate Transfer as `host→pinned→device` (and peer copies) with measured link
   models, not simulated DMA.
4. Re-run the hetero overlap and multi-copy residency tests against real GPUs;
   keep `simulated=True` only for analytic plan simulation.
5. Do not treat GPU-less CI green as CUDA validation.
