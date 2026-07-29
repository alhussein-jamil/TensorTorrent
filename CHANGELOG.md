# Changelog

## Unreleased

- Native Rust data plane is required for `compile()` / forward.
- Batched Compute, parameter Load, handle-release, and copy-sync callbacks
  (single GIL cross per wave of ready ops).
- Python `CopyStore` is a passive `handle → Tensor` value bag; Rust owns
  versions, residency, leases, allocations, aliases, transfers, and lifetime.
- Virtual-device buffers free on final allocation release (and on context drop);
  simulated device memory stays bounded across long mock forwards.
- Streaming byte cache / prefetch / inflight owned by `NativeStreamingStore`;
  Python keeps only the decoded-tensor RAM budget.
- Host cost calibration (`calibrate_host_priors`) feeds planner priors,
  VirtualBackend topology, and simulator bridge tax; prediction error reported.
- Bench-only legacy Python DAG (`testing.legacy_runtime`) for comparisons;
  production `run()` never enters it.
- Streaming under `ram_budget_bytes` via `NativeStreamingStore`.
- Public mock accelerators use native `VirtualBackend` buffers/streams/events.
- Activation spill files owned in Rust; Python converts tensor↔bytes only.
- Cancel, concurrent same-module forwards, save/load, TorchInductor keep-or-fallback.

## 0.1.0

Initial release: resource-graph IR, backend contracts, schedule-driven runtime,
weight streaming packs, discrete-event simulator, CLI (`doctor`, `profile`,
`validate-hardware`).
