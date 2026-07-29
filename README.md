# StreamCompiler

Heterogeneous streaming compiler and runtime for **PyTorch inference**.

`streamcompiler.compile` captures a module with `torch.export`, partitions the graph
into regions, plans where each region runs from measured latencies, and returns a
`torch.nn.Module` that executes those regions and returns the same outputs as eager
PyTorch.

```python
import torch, torch.nn as nn
import streamcompiler as sc

model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10)).eval()
x = torch.randn(32, 256)

compiled = sc.compile(model, example_inputs=(x,), devices="auto")
# Optional: inject ResourceGraph / measurements for hetero tests
# compiled = sc.compile(model, (x,), machine=cpu_plus_mock, measurements=ms)

torch.testing.assert_close(compiled(x), model(x))  # passes
```

`compiled` is a real `nn.Module`: it implements `forward`, exposes `parameters()`,
`state_dict()`, `.eval()` and `.to()`, and can be saved with `compiled.save(dir)` and
reloaded with `sc.load_compiled(dir)`. Under disk streaming, module attributes are
empty placeholders so the RAM budget stays honest; `state_dict()` rematerializes the
real pack tensors one block at a time.

## Capability status

Nothing below is marked working unless a test in this repository exercises it on the
machine running the suite.

| Area | Status | Notes |
| --- | --- | --- |
| CPU execution of exported graphs | **implemented** | `CpuBackend`; `tests/e2e/test_compile_execute.py` |
| Eager numerical equivalence | **implemented** | linear, MLP, branching, multi-input, structured outputs, shared parameters, buffers |
| Dependency-aware region scheduling | **implemented** | independent regions overlap, chains never do; `tests/e2e/test_concurrency.py` |
| Measured planner costs | **implemented** | `BackendProfiler` / benchmarks; simulated probes stay `simulated=True` / `measured=False` |
| Measured concurrency decision | **implemented** | worker/intra-op splits timed; threads only when they beat sequential |
| Weight streaming from disk | **implemented** | RAM budget, `pread`, LRU, prefetch; timed I/O∩compute; `tests/e2e/test_weight_streaming.py` |
| Pack I/O without full-file RAM | **implemented** | Two-pass chunked write + atomic replace; manifest-only load |
| Measured pack `pread` bandwidth | **implemented** | Specialization samples payload `pread` into plan notes when streaming |
| Artifact save/reload | **implemented** | `torch.export.save` plus plan and config |
| Hardware discovery | **implemented** | CPU, NUMA, memory tiers, links; `streamcompiler doctor` |
| CUDA / ROCm / MPS / SYCL backends | **untested here** | Shared `torch_device` path; `BackendError` when device absent |
| NCCL / RCCL / oneCCL collectives | **untested here** | Selection exercised; only Gloo has run |
| Transfer / makespan simulator | **simulated** | Same `ExecutableSchedule` DAG as runtime; always `simulated=True` |
| Schedule residency / transfers | **implemented** | Immutable schedule + `ExecutionContext`; Load=disk→host; Transfer for device; `CopyStore` + `VirtualDeviceTensor` on mock |
| Schedule-driven activation spill | **implemented** | Evict/Load under `activation_budget_bytes`; `recompute` policy rejected |
| BackendProfiler | **implemented** | CPU measured; mock_accel simulated |
| `compiled.validate()` | **implemented** | Structure, specialized-machine resources, spill/reload edges |
| Optional TorchInductor regions | **implemented** | Keep Inductor only when ≤1.05× eager FX; else eager FX fallback |
| Measured execution telemetry | **implemented** | `visualize(..., measured=True)` after forward |
| Liveness buffer reuse | **implemented** | Non-overlapping activations share slots; single-worker allocator |
| Throughput objective | **implemented** | Minimizes makespan (regression-tested) |
| Device-specific profile cache keys | **implemented** | Device, fingerprint, shapes, dtype, kernel, threads |
| Online profile feedback → replan | **implemented** | Returns `{plan, deltas}`; swaps live executor |
| Persistent process worker pool | **implemented** | `process_workers>0` Linux-fork pool |
| Cancel in-flight run | **implemented** | Stops new dispatch; drains in-flight; then `ExecutionCancelled` |
| Quantized pack / stream load | **experimental (opt-in)** | `allow_quantized_storage` + `numerical_mode=quantized`; dequant on load |
| CPU + mock-accel schedule | **implemented (simulated accel)** | `make_mock_accel_graph(device_count=…)`; host-staged multi-mock |
| Host-staged allreduce | **experimental scaffolding** | Helper + Gloo; not schedule-driven via `compile()` |
| Training (`allow_training=True`) | **implemented (graph-module fallback)** | Autograd via live `graph_module`; not schedule training |
| CPU + real GPU concurrent execution | **untested here** | Needs accelerator hardware |
| Tensor / pipeline parallel via `compile()` | **experimental scaffolding** | Helpers exist; not emitted by planner yet |
| Dynamic shapes | **scaffolding / not supported** | Static-shape specialization |


## Measured performance

Real numbers from `python benchmarks/run_baselines.py` on the development host
(x86_64, 8 threads, torch 2.13.0+cpu, no GPU). `ratio` is StreamCompiler latency
divided by eager latency, so lower is better and values above 1.0 are overhead.
Re-run the benchmark on your machine before citing these numbers.

| Case | Regions | Eager | StreamCompiler | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- |
| linear (32x512) | 1 | 0.049 ms | 0.306 ms | 6.27x | 0 |
| mlp_256x4 (32) | 1 | 0.119 ms | 0.488 ms | 4.09x | 0 |
| mlp_1024x4 (64) | 1 | 1.022 ms | 1.663 ms | 1.63x | 0 |
| branching_512 (64) | 1 (fused) | 0.272 ms | 0.825 ms | 3.03x | 0 |
| branching_1024 (128) | 1 (fused) | 1.088 ms | 1.628 ms | 1.50x | 0 |
| branches8_1024 (64) | 1 (fused) | 2.336 ms | 3.319 ms | 1.42x | 0 |
| branches4_2048 (256) | 1 (fused) | 11.048 ms | 11.940 ms | 1.08x | 0 |

Small models pay fixed schedule-dispatch overhead (flatten, validate, instruction
DAG). Larger GEMMs approach eager (ratio → ~1). When concurrency measurement finds
no speedup — on a wide independent level, on the full region DAG, **or** versus a
fused single-region schedule — the compiler fuses branches into one region and
runs a single-region ``ExecutableSchedule``. Forced concurrency
(`max_concurrent_regions>1`) keeps branched regions for overlap.

Region concurrency is decided by measurement at compile time, and on this 8-thread
host it usually loses end-to-end: one PyTorch GEMM already saturates the cores, so
overlapping regions mostly contend, and multi-region dispatch can erase a local win.
The measurement times several worker/thread splits on the widest level, confirms on
the full topo schedule, then compares the winner against a fused single region
before enabling threads. Forced concurrency (`max_concurrent_regions`) is covered
by tests that assert independent regions really overlap and that dependent regions
never do.

Weight streaming trades latency for capacity, as expected. On this host a
~2 MB model under a ~0.5 MB RAM budget keeps peak resident parameters under the
budget and matches eager outputs exactly (re-reads under eviction inflate total
bytes read). Prefetch submits ahead of use; whether I/O wall-clock overlaps region
compute depends on compute duration versus page-cache `pread` latency — the runtime
reports both `io_overlapped_with_compute_s` and `exposed_io_s` from timed intervals
rather than assuming overlap from futures. See `python benchmarks/run_streaming.py`.

## Architecture

```
portable:  torch.export → regions → IR → packs
specialize: discover → measure → plan → ExecutableSchedule → backends
runtime:   ScheduleExecutor (Prefetch/Load/Transfer/events/Compute/Evict/Release)
           CopyStore residency · streaming or resident parameter store
```

Details: [docs/architecture.md](docs/architecture.md). Hardware model:
[docs/heterogeneous_hardware.md](docs/heterogeneous_hardware.md). Backends:
[docs/backends.md](docs/backends.md).

## Behaviour worth knowing

- Compilation specializes to example inputs; different shape/dtype raises
  `UnsupportedFeatureError`.
- Default inference uses `torch.inference_mode`. `allow_training=True` is an
  autograd-compatible `graph_module` fallback — not schedule training.
- The specialized `ExecutableSchedule` is the exclusive runtime program. Simulator
  and runtime share instruction IDs; the simulator invents no transfers.
- `request_cancel()` stops new instruction dispatch, drains in-flight work, then
  raises `ExecutionCancelled`.
- `process_workers>0` uses Linux `fork` (not mixed-vendor isolation).
- Tensor/pipeline parallel, host-staged allreduce, and dynamic-shape helpers remain
  scaffolding until schedule-driven — [docs/roadmap.md](docs/roadmap.md).
- Saved artifacts are trusted code bundles. Concurrent `forward` on one
  `CompiledModule` is rejected.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Deployment commands

```bash
streamcompiler doctor --full
streamcompiler profile --all-resources
streamcompiler validate-hardware
streamcompiler benchmark-topology
streamcompiler autotune model_artifact/
```

Validation reports use distinct statuses so a discovered capability is never
presented as a validated one:

- hardware detected
- backend available
- backend compiled
- basic execution validated
- concurrent execution validated
- numerical correctness validated
- performance characterized
- unsupported capability
- fallback selected
- skipped

On a machine without GPUs, `doctor` skips the GPU checks explicitly rather than
reporting success.

## Design rules

- Do not assume identical GPUs, CUDA-only stacks, one CPU socket, unified memory
  latency, or direct peer-to-peer links.
- Do not force every resource busy; include a device only when it improves the
  selected objective.
- Planner code queries `ExecutionBackend` contracts — no scattered vendor
  conditionals.
- Never treat missing accelerators on a development machine as proof that accelerator
  execution works.

See [docs/architecture.md](docs/architecture.md),
[docs/heterogeneous_hardware.md](docs/heterogeneous_hardware.md),
[docs/backends.md](docs/backends.md), [docs/deployment.md](docs/deployment.md),
[docs/roadmap.md](docs/roadmap.md), and [docs/anti_patterns.md](docs/anti_patterns.md).

## License

Apache-2.0
