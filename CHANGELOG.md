# Changelog

## Unreleased

- Native Rust data plane is required for `compile()` / forward.
- Batched Compute, parameter Load, handle-release, and copy-sync callbacks
  (single GIL cross per wave).
- Python `CopyStore` is a passive tensor value bag; Rust owns residency.
- Streaming under `ram_budget_bytes` via `NativeStreamingStore`.
- Public mock accelerators use native `VirtualBackend` buffers/streams/events.
- Activation spill files owned in Rust; Python converts tensor↔bytes only.
- Cancel, concurrent same-module forwards, save/load, TorchInductor keep-or-fallback.

## 0.1.0

Initial release: resource-graph IR, backend contracts, schedule-driven runtime,
weight streaming packs, discrete-event simulator, CLI (`doctor`, `profile`,
`validate-hardware`).
