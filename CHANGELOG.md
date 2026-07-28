# Changelog

## Unreleased

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
