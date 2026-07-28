# Changelog

## Unreleased

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
