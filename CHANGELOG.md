# Changelog

## Unreleased

- Robustify: post-spill consumers always depend on an `activation_reload` Load;
  spill Evict waits for already-emitted consumers; losing spill races unlink
  orphan tempfiles; simulator keeps disk copies after reload (matches runtime).
- Immutable `ExecutableSchedule` / `PlanInstruction` (`FrozenAttrs`) with
  per-call `ExecutionContext` for futures, events, `CopyStore`, spills, and
  telemetry — schedule serialize before/after execution is unchanged.
- Activation spill/reload moved into the schedule: planner emits
  `Evict(kind=activation_spill)` and `Load(kind=activation_reload)` under
  `activation_budget_bytes`; runtime no longer transparent-spills in Compute.
- Load is disk→host only; device residency requires explicit Transfer
  (including parameter host→device for mock/CUDA-class placements).
- `BackendProfiler` (CPU measured + virtual accel simulated),
  `compiled.validate()`, stronger Inductor cache keys / `fullgraph` attempt,
  sim/runtime memory-agreement tests, schedule immutability tests.
- Simulator host Load peaks no longer alias into device VRAM when the machine
  graph lacks a host memory resource.

- Roadmap cleared: Milestone 2/3 features shipped with CPU-tested paths —
  shape buckets, host-staged tensor parallel, pipeline microbatch, intra-op
  split, quantized storage, measured contention + online profile feedback,
  process workers, Gloo/host multi-peer allreduce, storage fast-path hooks,
  async event helpers, real `TorchDeviceTransfer` when devices exist, and
  activation recompute overflow policy.
- Milestone 1 remaining list cleared on CPU hosts: schedule-driven Compute/Release
  execution, live `ActivationAllocator` reuse, activation disk spill/reload under
  `activation_budget_bytes`, `use_torch_compile=True` by default (keep only when
  measured ≥ eager), cancel API, and a dispatch-overhead ceiling test.
- `validate_schedule_resources` rejects Compute instructions whose resource is
  not in the discovered machine graph; specialization fails closed on mismatch.
- `GraphExecutor.request_cancel` aborts multi-region runs at the next region
  boundary with `ExecutionCancelled` and releases partial activations; the
  fast path honors a pre-call cancel only. Also exposed as
  `CompiledModule.request_cancel` / `streamcompiler.ExecutionCancelled`.
- Fast-path micro-dispatch: skip region-result coerce for bare tensors, cache
  resident parameter-store stats on the bound path, and avoid per-call
  `begin_execution` / transfer-event clear when the single-region fast path runs
  (measured ~0.7–0.9 µs less overhead on `Linear(8,4)` vs prior fast path).
- `validate_schedule` rejects malformed `ExecutableSchedule`s (duplicate
  instruction ids, dangling dependencies, dependency cycles,
  release-before-use, compute-before-transfer-completion); enforced when the
  planner builds a schedule and when `GraphExecutor` is constructed from one.
- `TensorDirectory` now joins a second concurrent request for the same
  tensor+destination transfer to the in-flight one instead of duplicating the
  copy or disk read.
- `ActivationAllocator`: liveness-derived buffer-reuse slots backed by one
  real physical byte buffer; two non-overlapping activations placed in the
  same slot now provably share memory (`data_ptr()` equality), with
  allocation/reuse/release telemetry per slot.
- Shared `ExecutableSchedule` (Compute / Transfer / Prefetch / Load / Release /
  WaitEvent / RecordEvent) built at specialization; simulator can replay it via
  `simulate_schedule`; GraphExecutor runs Transfer ops before consumers and
  records WaitEvent markers as host sync telemetry.
- Optional region `torch.compile` (Inductor) with explicit eager FX fallback and
  fingerprint-keyed compile cache (`CompileConfig.use_torch_compile`). Inductor
  is kept only when measured faster than eager FX on specialization examples.
- `TensorDirectory` tracks residency states; explicit `host_memcpy` / `disk_pread`
  transfer backends; device transfers remain simulated until hardware-validated.
- Liveness recomputed from producer–consumer edges; buffer-reuse plans for
  non-overlapping activations; alias analysis covers views and rejects mutable
  shared weights.
- Measured execution Chrome/HTML telemetry via
  `CompiledModule.visualize(..., measured=True)` including residency and I/O
  intervals.
- Planner resource decisions cite millisecond critical-path deltas.
- CUDA kernel ids renamed from fictional `cuda_inductor_*` to `cuda_fx_*`.
- Fingerprint mismatch clears the process-local ``torch.compile`` region cache.
- Mixed-vendor plans annotate ``host_staged_tax_prior=1.15x`` as unmeasured.
- Plan HTML/Chrome visualize labels analytic simulation explicitly.
- Host-staged ``allreduce`` sums real CPU tensors; NCCL/RCCL/oneCCL/Gloo raise
  ``UnsupportedFeatureError`` instead of returning fake ``status`` dictionaries.
- ``Objective.MEMORY`` minimizes a per-device peak working-set estimate
  (largest region on each device, summed across devices; latency tie-break).
- Fixed inverted `Objective.THROUGHPUT` scoring so the planner selects the
  lower-makespan plan; regression tests cover score order and device choice.
- Region measurement cache keys include device id, fingerprint, shapes, dtype,
  kernel id, and CPU thread configuration (no cross-device reuse).
- Repository URLs in `pyproject.toml` point at
  `github.com/alhussein-jamil/streamcompiler`.
- GPU presence is reported as `hardware_detected` / unvalidated, not
  `concurrent_execution_validated`.
- Discrete-event simulator tracks tensor lifetimes, transfers, destination
  residency, release events, eviction pressure, prefetch hints, and contention;
  results always carry `simulated=True`. Parameter/state bytes stay live for the
  region interval so overlapping peers that share a memory pool stack in peak.
  Specialization profiles expose `transfer_landed_events` beside transfer counts.
- Explicit residency/transfer schedule (`runtime/residency.py`) prepares
  future CPU–GPU plans without claiming validation.
- CI builds wheel/sdist, clean-installs the wheel, runs CLI smoke, and gates
  against an undeclared native tree.
- Model packs no longer load or assemble the full file in RAM when reading
  manifests or writing packs.
- Streaming runtime records timed ``pread`` ∩ compute overlap, exposes I/O
  stalls, and prefetches only after the live region is pinned.
- Specialization measures pack ``pread`` bandwidth when disk streaming is used.
- ``allow_nvme_streaming=False`` now rejects over-budget compiles instead of
  being ignored.
- ``CompiledModule.state_dict()`` rematerializes real weights under streaming
  (module attributes stay empty placeholders to enforce the RAM budget).
- Concurrency enablement is confirmed on the full region DAG, not only the
  widest independent level, so local microbench wins cannot slow the whole graph.
- Per-call streaming I/O overlap windows reset each ``forward``; region partition
  budgets scale with ``prefetch_distance``.
- ``pack_state_dict`` lays out from tensor nbytes and serializes each payload
  once at write time; streaming ``save()``/``load_compiled`` share ``model.pack``.
- After concurrency measurement, a fused single-region candidate is timed and
  preferred when it beats multi-region execution on the same inputs.

## 0.1.0

Initial heterogeneous streaming compiler milestone:

- Resource graph IR with independent compute, memory, and transfer resources
- ExecutionBackend contract for CPU, CUDA, ROCm, MPS, and SYCL
- Communication backends including host-staged mixed-vendor fallback
- Two-stage portable compile + machine specialization
- Maximal planner with subset search and inclusion/exclusion reasons
- Discrete-event simulator and chrome-trace visualization
- Packed model storage format
- Hardware validation suite and CLI:
  `doctor`, `profile`, `validate-hardware`, `benchmark-topology`, `autotune`
- CPU-path correctness tests; accelerator paths reported honestly when absent
