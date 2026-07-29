# StreamCompiler engineering update

Date: 2026-07-29

This archive contains a focused implementation pass over the remaining CPU-side and simulated-heterogeneous architecture gaps. The public API and existing execution path were preserved.

## Implemented changes

### Storage-aware physical allocation accounting

- Physical allocation identity now follows PyTorch backing storage rather than Python tensor object identity.
- Tensor views with different shapes or storage offsets share one physical allocation.
- View metadata (`storage_offset`, shape, stride) remains attached to each resident copy.
- Distinct physical copies on different resources count independently.
- `AllocationTable` now exposes allocation capacity, resource ownership, and live bytes per resource.
- Replacing the same residency key with the same allocation no longer increments reference counts incorrectly.

### Backend-neutral specialization profiling

- Normal specialization now considers every allowed, profile-capable resource supplied in the machine graph.
- Explicit virtual/mock accelerators can be profiled on a CPU-only VM while remaining labelled simulated.
- Unavailable real backends remain excluded unless a valid profiler is provided.
- Device-specific measurement keys remain tied to concrete device, graph signature, dtype, shape, implementation, and thread configuration.

### Exact schedule tensor sizes

- Specialized schedules now propagate per-tensor byte metadata through parameter prefetch/load, activation and parameter transfers, compute, spill, eviction, and release operations.
- New schedule validation rejects ambiguous multi-tensor instructions without exact per-tensor sizes.
- Prior-only CLI planning without an exported program remains supported but is explicitly labelled estimated.

### Schedule-level asynchronous liveness

- Added an executable-schedule liveness pass.
- It derives producer sets, consumer sets, maximal asynchronous consumers, and safe release dependencies from the instruction DAG.
- Release instructions preserve explicit safety edges and add the final asynchronous consumer frontier.

### Simulator physical-allocation model

- The simulator now tracks resident copies through simulated physical allocations and reference counts.
- Releases remove exact copies and free capacity only after the final reference.
- Peak activation memory counts distinct physical allocations across resources rather than one logical tensor globally.

### Ordered virtual streams

- Direct virtual/mock streams preserve ordered stream semantics by default.
- CPU resources may still use an explicitly unordered concurrent worker pool through `DeviceStreams`.
- This keeps deterministic accelerator-style ordering while preserving legal CPU branch concurrency.

### Incremental huge-tensor packing

- Added public `ChunkedTensorSource` support.
- A loader can provide a fresh iterator of byte chunks for a single huge tensor.
- Chunks are written incrementally and never concatenated into one payload.
- Existing two-pass metadata calculation, atomic replacement, bounded manifest reading, and block-level loading are preserved.
- Quantization of chunk sources is rejected until a streaming quantizer exists rather than silently materializing the tensor.

### Documentation

- Updated README capability descriptions and behavioral invariants.
- Added an Unreleased changelog section.
- Added `docs/local_cpu_validation.json` with a real local CPU correctness and latency smoke measurement.

## New regression coverage

Added tests for:

- Different views sharing one backing allocation.
- Distinct host and virtual-device copies counting separately.
- Per-resource physical memory accounting.
- Virtual-device profiling through normal specialization.
- Exact schedule-size validation.
- Schedule-level final asynchronous consumer derivation.
- Incremental `ChunkedTensorSource` pack writing and round-trip loading.

## Validation performed

### Passed

- `python -m compileall -q src tests`
- Focused final regression suite: **37 passed**
- Earlier implementation batches covered all unit, integration, simulation, hardware, and end-to-end files in smaller groups; every completed batch passed.
- Test collection: **306 non-property tests collected**.
- Public import smoke test for `ChunkedTensorSource`.
- Changed-file whitespace/tab scan.
- Real CPU compile/execute correctness smoke benchmark.

### CPU smoke benchmark

Model: `Linear(128,256) -> GELU -> Linear(256,128)`, batch 32.

- PyTorch: `2.10.0+cpu`
- Compile time: approximately `0.878 s`
- Maximum absolute output error: `0.0`
- Eager median: approximately `0.207 ms`
- StreamCompiler median: approximately `0.811 ms`
- Small-model overhead: approximately `3.92x`

This overhead is expected from Python schedule dispatch on a small model. No performance claim is inferred from this smoke test.

### Environment limitations

The sandbox has no network access and lacks these optional development dependencies:

- `ruff`
- `mypy`
- `hypothesis`
- `pytest-timeout`
- `build`
- `hatchling`

Consequently:

- Property tests requiring Hypothesis were not executed.
- Ruff and MyPy were not executed.
- Wheel/sdist construction could not be executed.
- The full non-property suite in one pytest process did not terminate within the sandbox timeout; the suites were therefore validated in smaller batches, which also avoids cross-test process-pool interference.

No missing tool was bypassed by weakening repository configuration or tests.

## Remaining hardware-dependent work

These capabilities are still simulated or unvalidated because the development VM has no GPU:

- Real CUDA/ROCm streams and completion events.
- Pinned-memory staging.
- RAM-to-VRAM weight streaming.
- CPU/GPU and DMA/GPU overlap.
- Real multi-GPU and peer-to-peer transfers.
- VRAM eviction under measured device pressure.
- Mixed-vendor worker isolation.

The new allocation, schedule-size, profiling, liveness, and virtual-stream foundations are designed to make those validations possible without introducing fake success paths.
