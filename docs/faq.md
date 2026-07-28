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

No. Regions execute under `torch.inference_mode`, so outputs are inference tensors
and cannot participate in autograd. Compile for inference only.
