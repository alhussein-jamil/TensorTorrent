# FAQ

**Why does `doctor` say CUDA is unsupported?**
No usable CUDA runtime. Status is `unsupported_capability`, not a pass. Validate
on the target machine with `streamcompiler validate-hardware`.

**Does the planner use every GPU?**
No. A device is included only when it improves the objective after transfer cost.
See `compiled.explain()`.

**Can I mix NVIDIA and AMD in one process?**
Not today. Mixed-vendor links may be host-staged; real execution needs separate
workers per backend.

**Do I need a GPU to compile?**
No. Portable artifacts are hardware-independent. Specialize per host.

**Why slower than eager on tiny models?**
Fixed schedule dispatch. Capacity under a RAM budget is the main win, not
micro-latency.

**Different batch size?**
No. Example shapes/dtypes are fixed. Mismatch raises `UnsupportedFeatureError`.

**Training?**
Default compile is inference-only (`.train()` raises). Pass
`CompileConfig(allow_training=True)` for a normal loop: `.train()` runs the live
`graph_module` (`backward` / `optimizer.step()`); `.eval()` returns to the
inference schedule with the updated weights. The opt-in is explicit because
training does not use the schedule. Incompatible with NVMe parameter streaming
and `process_workers`. Schedule-based training is not supported.

**Execution timeline?**

```python
compiled(x)
compiled.visualize("run.html", measured=True)
```

Default `visualize` is analytic simulation of the same schedule (`simulated=True`).

**Cancel?**
`request_cancel()` flips per-forward tokens. The dispatcher stops launching new
work at wave boundaries, then raises `ExecutionCancelled`. In-flight Compute in
the current wave still finishes.
