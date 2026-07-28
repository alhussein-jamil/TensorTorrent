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

Yes when both backends are available. Direct P2P is queried per link. Missing
cross-vendor P2P uses host-staged transfers instead of failing the machine.

## Does portable compilation require GPUs?

No. Portable artifacts are hardware-independent. Machine specialization happens
later on each deployment host.
