## Unreleased

- Hybrid Rust data plane: workspace under `crates/`, maturin/`streamcompiler._native`.
- Public `ScheduleExecutor.run` dispatches via Rust (GIL released); Python keeps
  tensor instruction bodies. Fallback only with `STREAMCOMPILER_ALLOW_PYTHON_RUNTIME=1`.
- Rust schedule model, validation, residency/allocations, DES simulator, virtual
  backend, pack validation, profiler DB, Criterion schedule-overhead benches.
- See `MIGRATION_REPORT.md` and `docs/architecture.md`.

- Integrated backend-neutral region profiling into normal specialization for every profile-capable supplied resource.
- Made physical allocation identity storage-aware so tensor views share backing memory while separate resource copies count independently.
- Added exact per-tensor schedule-size validation and physical-allocation simulation.
- Added schedule-level asynchronous liveness metadata for Release instructions.
- Added `ChunkedTensorSource` for incrementally writing an individual huge tensor.
- Added regression coverage for virtual-device profiling, view aliases, multi-resource memory, exact sizes, and async liveness.

# Changelog

## Unreleased

- Schedule residency: storage-identity allocs, `VirtualDeviceTensor` for mock
  devices, Release waits on Transfer Record/Wait edges, activation peaks tracked
  without requiring an activation budget, sim/runtime peak agreement.
- Profiler cache preserves `measured`/`simulated`; planner may use simulated
  latencies without claiming measured hardware.
- Pack writes are chunked and atomic (`*.tmp` + `os.replace`).
- Planner hard-filters devices whose working set exceeds allocatable memory;
  `make_mock_accel_graph(device_count=…)` supports unequal multi-mock topologies.
- Removed legacy `TensorDirectory`, `execute_transfer_instruction`, and
  `placements_from_schedule`; schedule path is `CopyStore` only.
- `activation_overflow_policy="recompute"` rejected until implemented (spill only).
- `replan_with_profile_feedback()` / `apply_profile_feedback()` return
  `{plan, deltas}`.
- `request_cancel()` stops new schedule dispatch, drains in-flight ops, then
  raises `ExecutionCancelled`.
- Docs deduped around the schedule-driven runtime
  (`architecture` / `roadmap` / `faq` / `heterogeneous_hardware` / `backends`).

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
