# Hybrid Rust migration report

Date: 2026-07-29 (remaining residuals closed)

## Architecture

```
Python PyTorch compiler frontend
  (torch.export / FX / Inductor / shape guards / PyTrees / user API)
        |
NativeCompiledArtifact  (immutable schedule, created once)
        |
NativeExecutionContext  (per forward: residency, events, allocs, resources,
                         storage spills, cancel)
        |
Rust dispatcher           Transfer / Record / Wait / Release / Evict / Load
                          + native activation spill files
        |
Python ONLY for           Compute region waves
                          + tensor↔bytes (dematerialize/materialize)
                          + streaming decoded-tensor cache pin budget
```

Public path: `compile(...)` / `compiled(x)`. Native extension required.

## Closed this pass

- Native activation spill format (`SCSPILL1`) in `streamcompiler-storage`
- Rust `ExecutionStorageState` on `NativeExecutionContext`
- Spill/reload via dematerialize/materialize callbacks (not instruction_handler)
- Release falls back to disk copy after spill
- Streaming prefetch: native byte pread/inflight; Python worker only tensorizes
- Resident path: `non_compute_python_callbacks == 0`
- Persistent residency (no fake prematerialized Load events)

## Ownership boundary

| Concern | Owner |
|---------|--------|
| Export / FX / Inductor / guards / PyTrees | Python |
| Region Compute execution | Python (batched GIL) |
| Tensor ↔ contiguous bytes | Python (dematerialize/materialize) |
| Schedule, residency, events, allocs, leases | Rust |
| Pack pread / byte cache / prefetch queue | Rust (`NativeStreamingStore`) |
| Activation spill files | Rust |
| Transfer / Record / Wait / Release | Rust |
| Simulator / telemetry intervals | Rust |
| Profile DB + feedback | Rust |

`CopyStore` = Python tensor values only.

## Measured (CPU-only)

| Metric | Value |
|--------|-------|
| Non-compute Python callbacks (resident) | **0** |
| Compute GIL callbacks / forward | 1 |
| Hot-path schedule convert | 0 |
| pytest (non-property) | 336 passed |
| Eager median (MLP 256, batch 32) | ~0.048 ms |
| Native median | ~0.617 ms |
| `make native-gate` | PASS |

JSON: `benchmarks/results/native_forward_*.json`.

## Remaining (cannot close on this VM)

- Real CUDA / ROCm / multi-GPU / pinned DMA / P2P — **untested / simulated only**
- Tiny-model fixed dispatch tax (~0.6 ms); large GEMMs near eager
- Streaming decoded-tensor pin map still Python (bytes cache is native)
- Full typed context table split (logical/view/ready-queue structs) incremental polish

## Exact real-GPU validation work

1. CUDA/ROCm region execute + numerical parity
2. Real stream/event overlap vs mock delays
3. Pinned host staging + DMA
4. Multi-GPU peer transfer vs host-staged fallback
5. VRAM eviction under measured budgets
