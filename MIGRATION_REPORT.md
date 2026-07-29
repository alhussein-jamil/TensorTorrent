# Hybrid Rust migration report

Date: 2026-07-29 (updated)

## Architecture

Python control plane (`torch.export` / FX / Inductor) → persistent
`NativeCompiledArtifact` → one `NativeExecutionContext` per forward → Rust
data plane → Python only for real PyTorch region compute.

Public path unchanged: `compile(...)` / `compiled(x)`.

## Stages

| Stage | Status |
|-------|--------|
| Native schedule model | Done |
| Persistent artifact | Done |
| Region-only Python callbacks (resident CPU + Transfer metadata) | Done |
| Batched Compute region invoker (one GIL cross per ready wave) | Done |
| One `NativeExecutionContext` (shared residency/events/allocs/resources) | Done |
| Same-module concurrent forwards | Done (`InFlightGate`) |
| Opaque handles + Rust residency metadata | Done (region path) |
| Explicit stream / copy-engine / link occupancy | Done (`ResourceState`) |
| Strict Transfer / WaitEvent (no invent source/event) | Done |
| Static runtime path selection (no mid-forward restart) | Done |
| `SimulationOutcome` (Valid / InfeasibleMemory / …) | Done |
| Rust simulator default | Done (`STREAMCOMPILER_PYTHON_SIM=1` forces oracle) |
| Native pack Load/Prefetch | Done (`NativeStreamingStore` + `NativePackReader`) |
| Native profiler + `apply_profile_feedback` | Done |
| Async virtual backend | Done (**simulated**) |
| CI `make native-gate` | Done (import + public-path proof) |
| `cargo test/clippy` includes `streamcompiler-python` | Done |
| Dual-runtime removal | Dev-only: `STREAMCOMPILER_DEV_PYTHON_RUNTIME=1` |
| CopyStore | Value bag; Rust owns valid/lease/release on native-residency path |

## Measured (CPU-only)

| Metric | Value |
|--------|-------|
| Hot-path schedule convert | 0 |
| GIL callbacks / forward (MLP 256) | 1 Compute |
| Same-module concurrent forwards | 8 threads OK |
| `make native-gate` | PASS |
| cargo workspace (+ python crate) | green |
| Eager median (batch 32, Linear256 MLP) | ~0.042 ms |
| Native SC median | ~1.09 ms |
| Legacy Python SC median | ~1.14 ms |
| Tiny Linear dispatch overhead vs eager | ~740 µs (ceiling 1000 µs) |

JSON: `benchmarks/results/native_forward_*.json`.

## Simulated / unvalidated

Virtual accelerators = **simulated**. No CUDA/ROCm/multi-GPU validation on this VM.

## Residual

- Streaming schedules still use instruction-callback path (not region data plane)
- Python still tensorizes + enforces decoded RAM budget; Rust owns byte cache/prefetch
- Python CopyStore still holds tensor values
- Event-derived final liveness incomplete
- Activation spill still Python
- Native still ~26× eager on tiny MLP (dispatch tax); barely beats legacy Python
- Pack SHA-256 still verified in Python (native CRC field unused for SCPACK1)
- WorkerPool spawn still `expect` (typed error deferred)
