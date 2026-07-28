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
reloaded with `sc.load_compiled(dir)`. Under disk streaming, module attributes are
empty placeholders so the RAM budget stays honest; `state_dict()` rematerializes the
real pack tensors one block at a time.

## Capability status

Nothing below is marked working unless a test in this repository exercises it on the
machine running the suite.

| Area | Status | Notes |
| --- | --- | --- |
| CPU execution of exported graphs | **implemented** | `CpuBackend` compiles and runs every region; `tests/e2e/test_compile_execute.py` |
| Eager numerical equivalence | **implemented** | linear, MLP, branching, multi-input, structured outputs, shared parameters, buffers |
| Dependency-aware region scheduling | **implemented** | independent regions overlap, chains never do; `tests/e2e/test_concurrency.py` |
| Measured planner costs | **implemented** | region latencies are benchmarked per device; unmeasured regions are labelled `measured=False` |
| Measured concurrency decision | **implemented** | several worker/intra-op-thread splits are timed; threads are used only when one beats the sequential schedule |
| Weight streaming from disk | **implemented** | RAM budget, `pread` block loads, LRU eviction, prefetch after pin, double buffering; timed I/O∩compute overlap in stats; `tests/e2e/test_weight_streaming.py` |
| Pack I/O without full-file RAM | **implemented** | Manifest load and pack write never assemble the whole file; `tests/unit/test_pack_format.py` |
| Measured pack `pread` bandwidth | **implemented** | Specialization samples payload `pread` and records MiB/s in plan notes when streaming |
| Artifact save/reload | **implemented** | `torch.export.save` plus plan and config |
| Hardware discovery (CPU, NUMA, memory tiers, links) | **implemented** | `streamcompiler doctor` reports what it actually found |
| CUDA / ROCm / MPS / SYCL backends | **untested here** | they share the PyTorch device path in `backends/torch_device.py` and raise `BackendError` when the device is absent. No GPU was available to run them |
| NCCL / RCCL / oneCCL collectives | **untested here** | selection logic is exercised; only Gloo has run |
| Transfer and makespan simulator | **simulated** | analytic critical-path model with tensor lifetimes, transfers, destination residency, release, prefetch hints, contention; overlapping shared-memory state stacks in peak; always labelled `simulated=True` |
| Explicit residency / transfer schedule | **implemented** | `ExecutableSchedule` drives Transfer / RecordEvent / WaitEvent / Compute / Release; mock CPU+accel path in `tests/unit/test_hetero_execution_path.py` |
| Optional TorchInductor regions | **implemented** | `CompileConfig.use_torch_compile=True` wraps regions with `torch.compile`; keeps Inductor only when measured ≤1.05× eager FX, else explicit eager fallback |
| Tensor residency | **implemented** | Schedule path: `CopyStore` keyed by `(logical_tensor_id, resource_id)`; Load=disk→RAM, Transfer=RAM→dest |
| Measured execution telemetry | **implemented** | `compiled.visualize(path, measured=True)` after a forward; Chrome JSON / HTML; distinct from simulated plan traces |
| Liveness buffer reuse plan | **implemented** | non-overlapping activations share slots; overlapping stay distinct |
| Throughput objective | **implemented** | minimizes makespan (regression-tested); no inverted score |
| Device-specific profile cache keys | **implemented** | device, fingerprint, shapes, dtype, kernel, threads |
| Host-staged allreduce | **implemented** | real CPU tensor sum; vendor collectives raise until wired |
| Training / autograd (`allow_training=True`) | **implemented (graph-module fallback)** | partitioned live `graph_module` for autograd; **not** heterogeneous schedule training |
| Online profile feedback → replan | **implemented** | `apply_profile_feedback()` re-specializes and swaps the live executor |
| Persistent process worker pool | **implemented** | nonblocking submit; `process_workers>0` Linux-fork pool via `GraphExecutor` / schedule path |
| Quantized pack / stream load | **experimental (opt-in)** | `allow_quantized_storage` + `numerical_mode=quantized` writes `int8_affine`; streaming dequant; not a quantized kernel path |
| CPU + mock-accel concurrent schedule | **implemented (simulated accel)** | instruction-DAG `ScheduleExecutor`; multi-copy residency; mock async streams |
| CPU + real GPU concurrent execution | **untested here** | needs accelerator hardware |
| Tensor / pipeline parallel via `compile()` | **experimental scaffolding** | helpers exist; not emitted/executed through `compile()` schedule yet |
| Dynamic shapes | **scaffolding / not supported** | compilation is static-shape until schedule emits dynamic programs |

## Measured performance

Real numbers from `python benchmarks/run_baselines.py` on the development host
(x86_64, 8 threads, torch 2.13.0+cpu, no GPU). `ratio` is StreamCompiler latency
divided by eager latency, so lower is better and values above 1.0 are overhead.

| Case | Regions | Eager | StreamCompiler | Ratio | Max abs error |
| --- | --- | --- | --- | --- | --- |
| linear (32x512) | 1 | 0.048 ms | 0.056 ms | 1.18x | 0 |
| mlp_256x4 (32) | 1 | 0.108 ms | 0.116 ms | 1.07x | 0 |
| mlp_1024x4 (64) | 1 | 1.028 ms | 1.042 ms | 1.01x | 0 |
| branching_512 (64) | 1 (fused) | 0.466 ms | 0.290 ms | 0.62x | 0 |
| branching_1024 (128) | 1 (fused) | 1.101 ms | 1.115 ms | 1.01x | 0 |
| branches8_1024 (64) | 1 (fused) | 2.332 ms | 2.365 ms | 1.01x | 0 |
| branches4_2048 (256) | 1 (fused) | 11.134 ms | 11.152 ms | 1.00x | 0 |

When concurrency measurement finds no speedup — on a wide independent level, on
the full region DAG, **or** versus a fused single-region schedule — the compiler
fuses branches into one region and runs a single-region ``ExecutableSchedule``. Forced
concurrency (`max_concurrent_regions>1`) keeps branched regions for overlap.
Large models are at or near eager parity on this host.

Region concurrency is decided by measurement at compile time, and on this 8-thread
host it usually loses end-to-end: one PyTorch GEMM already saturates the cores, so
overlapping regions mostly contend, and multi-region dispatch can erase a local win.
The measurement times several worker/thread splits on the widest level, confirms on
the full topo schedule, then compares the winner against a fused single region
before enabling threads. Forced concurrency (`max_concurrent_regions`) is covered
by tests that assert independent regions really overlap and that dependent regions
never do.

Weight streaming trades latency for capacity, as expected. On this host a
2.1 MB model under a 0.5 MB RAM budget reads about 6.4 MB from the pack
(with eviction re-reads), keeps peak resident parameters under the budget, and
matches eager outputs exactly. Prefetch submits ahead of use; whether I/O
wall-clock overlaps region compute depends on compute duration versus page-cache
`pread` latency — the runtime reports both `io_overlapped_with_compute_s` and
`exposed_io_s` from timed intervals rather than assuming overlap from futures.
See `python benchmarks/run_streaming.py`.

## Architecture

```
core compiler (hardware independent)
  torch.export capture -> region partitioning -> heterogeneous IR
  alias/liveness analysis, packed weights, cost model, planner, simulator

execution backends (capability-queried)
  CPU | CUDA | ROCm | MPS | Intel-SYCL

runtime
  ExecutableSchedule instruction-DAG executor (exclusive)
  parameter stores: resident | streaming with RAM budget and prefetch
  CopyStore multi-copy residency keyed by (tensor, resource)
communication backends
  NCCL | RCCL | oneCCL | Gloo | host-staged fallback
```

Compilation has two stages:

1. **Portable compilation** — export, partition into regions, lower to
   heterogeneous IR, alias/liveness, packed weights.
2. **Machine specialization** — discover hardware, measure regions on the available
   devices, plan placements, compile regions through the selected backends, measure
   whether concurrency helps, and cache a machine-specific artifact.

## Behaviour worth knowing

- Compilation is specialized to the example inputs. Calling with a different shape or
  dtype raises `UnsupportedFeatureError` instead of silently mis-executing.
- Default inference path runs under `torch.inference_mode`. With
  `CompileConfig.allow_training=True`, forward uses the partitioned live
  `graph_module` so `backward()` can populate grads — an
  **autograd-compatible graph-module fallback**, not heterogeneous schedule training.
- The specialized `ExecutableSchedule` is the exclusive runtime program: an
  instruction dependency DAG (`Prefetch`/`Load`/`Transfer`/`RecordEvent`/
  `WaitEvent`/`Compute`/`Evict`/`Release`). Runtime does not convert it back into
  a region-prelude scheduler.
- `process_workers>0` uses a Linux `fork` pool for concurrent CPU regions.
  Fork CoW and CUDA-after-fork limitations apply; this is not mixed-vendor process
  isolation.
- Tensor-parallel, pipeline microbatch, and dynamic-shape helpers remain
  scaffolding until emitted and executed through `compile()`'s schedule; see
  `docs/roadmap.md`.
- `torch.export`'s own input guard is removed during lowering because
  `RegionProgram.flatten_inputs` performs the equivalent shape and dtype check with a
  clearer error.
- Saved artifacts (`exported.pt2`) are trusted code bundles. Only load directories you
  produced. Concurrent `forward` on one `CompiledModule` is rejected; serialize or
  compile per thread.

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
