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
