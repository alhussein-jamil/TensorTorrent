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

torch.testing.assert_close(compiled(x), model(x))  # passes
```

`compiled` is a real `nn.Module`: it implements `forward`, exposes `parameters()`,
`state_dict()`, `.eval()` and `.to()`, and can be saved with `compiled.save(dir)` and
reloaded with `sc.load_compiled(dir)`.

## Capability status

Nothing below is marked working unless a test in this repository exercises it on the
machine running the suite.

| Area | Status | Notes |
| --- | --- | --- |
| CPU execution of exported graphs | **implemented** | `CpuBackend` compiles and runs every region; `tests/e2e/test_compile_execute.py` |
| Eager numerical equivalence | **implemented** | linear, MLP, branching, multi-input, structured outputs, shared parameters, buffers |
| Dependency-aware region scheduling | **implemented** | independent regions overlap, chains never do; `tests/e2e/test_concurrency.py` |
| Measured planner costs | **implemented** | region latencies are benchmarked per device; unmeasured regions are labelled `measured=False` |
| Measured concurrency decision | **implemented** | threads are used only when timing shows a speedup, otherwise the plan stays sequential |
| Weight streaming from disk | **implemented** | RAM budget, `pread` block loads, LRU eviction, prefetch, double buffering; `tests/e2e/test_weight_streaming.py` |
| Artifact save/reload | **implemented** | `torch.export.save` plus plan and config |
| Hardware discovery (CPU, NUMA, memory tiers, links) | **implemented** | `streamcompiler doctor` reports what it actually found |
| CUDA / ROCm / MPS / SYCL backends | **untested here** | they share the PyTorch device path in `backends/torch_device.py` and raise `BackendError` when the device is absent. No GPU was available to run them |
| NCCL / RCCL / oneCCL collectives | **untested here** | selection logic is exercised; only Gloo has run |
| Transfer and makespan simulator | **simulated** | analytic model used for planning, clearly reported as `simulated_makespan_s` |
| CPU + GPU concurrent execution | **planned** | planner data structures allow it; no implementation of cross-device dataflow yet |
| Multi-process mixed-vendor workers | **planned** | not started |
| Dynamic shapes, training, autograd | **not supported** | compilation is static-shape, inference only |

## Measured performance

Real numbers from `python benchmarks/run_baselines.py` on the development host
(x86_64, 8 threads, torch 2.13.0+cpu, no GPU). `ratio` is StreamCompiler latency
divided by eager latency, so lower is better and values above 1.0 are overhead.

| Case | Regions | Eager | StreamCompiler | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- |
| linear (32x512) | 1 | 0.048 ms | 0.073 ms | 1.51x | 0 |
| mlp_256x4 (32) | 1 | 0.106 ms | 0.141 ms | 1.32x | 0 |
| mlp_1024x4 (64) | 1 | 1.030 ms | 1.081 ms | 1.05x | 0 |
| branching_512 (64) | 4 | 0.278 ms | 0.349 ms | 1.26x | 0 |
| branching_1024 (128) | 4 | 1.095 ms | 1.198 ms | 1.09x | 0 |

StreamCompiler is currently **slower than eager** on CPU: roughly 25 microseconds of
fixed dispatch overhead per call, which is invisible for large models and dominant
for small ones. It does not yet beat eager on this host, and the benchmark is
reported as-is rather than tuned to look favourable.

Weight streaming trades latency for capacity, as expected: with a 0.5 MB RAM budget
against 76 MB of reads, the same model runs about 8x slower than the resident store
while producing identical outputs.

## Architecture

```
core compiler (hardware independent)
  torch.export capture -> region partitioning -> heterogeneous IR
  alias/liveness analysis, packed weights, cost model, planner, simulator

execution backends (capability-queried)
  CPU | CUDA | ROCm | MPS | Intel-SYCL

runtime
  region graph executor (dependency-aware, optional threads)
  parameter stores: resident | streaming with RAM budget and prefetch

communication backends
  NCCL | RCCL | oneCCL | Gloo | host-staged fallback
```

Compilation has two stages:

1. **Portable compilation** — export, normalize, partition into regions, lower to
   heterogeneous IR, alias/liveness, packed weights.
2. **Machine specialization** — discover hardware, measure regions on the available
   devices, plan placements, compile regions through the selected backends, measure
   whether concurrency helps, and cache a machine-specific artifact.

## Behaviour worth knowing

- Compilation is specialized to the example inputs. Calling with a different shape or
  dtype raises `UnsupportedFeatureError` instead of silently mis-executing.
- Regions run under `torch.inference_mode`, so outputs are inference tensors and
  cannot be used in an autograd graph.
- `torch.export`'s own input guard is removed during lowering because
  `RegionProgram.flatten_inputs` performs the equivalent shape and dtype check with a
  clearer error.

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
