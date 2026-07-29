# Hybrid Rust migration report

Date: 2026-07-29 (residuals closed)

## Architecture

Python control plane (`torch.export` / FX / Inductor) → persistent
`NativeCompiledArtifact` → one `NativeExecutionContext` per forward → Rust
data plane → Python only for real PyTorch region compute (plus narrow I/O
bodies for streaming pins and activation spill/reload).

Public path unchanged: `compile(...)` / `compiled(x)`. Native extension required.

## Stages

| Stage | Status |
|-------|--------|
| Native schedule model | Done |
| Persistent artifact | Done |
| Region-only Python callbacks (resident CPU + Transfer metadata) | Done |
| Batched Compute region invoker (one GIL cross per ready wave) | Done |
| One `NativeExecutionContext` (shared residency/events/allocs/resources) | Done |
| Same-module concurrent forwards | Done (`InFlightGate`) |
| Opaque handles + Rust residency metadata | Done |
| Explicit stream / copy-engine / link occupancy | Done (`ResourceState`) |
| Strict Transfer / WaitEvent (no invent source/event) | Done |
| Static runtime path selection (no mid-forward restart) | Done |
| `SimulationOutcome` (Valid / InfeasibleMemory / …) | Done |
| Rust simulator default | Done (`STREAMCOMPILER_PYTHON_SIM=1` forces oracle) |
| Native pack Load/Prefetch | Done (`NativeStreamingStore` + `NativePackReader`) |
| Hybrid streaming/spill I/O handler on region path | Done |
| Pack CRC32 dual-write + native verify | Done |
| WorkerPool spawn typed errors (`try_new`) | Done |
| Python production DAG removed | Done (`run_schedule_native` only) |
| Transfer dest leases until Release | Done |
| Activation spill fail-closed without Python body | Done |
| Native profiler + `apply_profile_feedback` | Done |
| Async virtual backend | Done (**simulated**) |
| CI `make native-gate` | Done |

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
| Tiny Linear dispatch overhead vs eager | ~740 µs (ceiling 1000 µs) |

JSON: `benchmarks/results/native_forward_*.json`.

## Simulated / unvalidated

Virtual accelerators = **simulated**. No CUDA/ROCm/multi-GPU validation on this VM.

## Closed residuals (this pass)

- Streaming schedules on hybrid region + I/O handler (pooled overlap when width > 1)
- Python DAG / `STREAMCOMPILER_DEV_PYTHON_RUNTIME` removed from production path
- WorkerPool `expect` → `try_new`
- Pack CRC32 written and verified natively
- CopyStore demoted to value bag (docs + comments); Rust owns lease/release
- Event-derived liveness: planner `apply_schedule_liveness` + Transfer dest leases
- Activation spill remains Python `torch.save`/`load` by design; bare Rust Evict of
  `kind=activation_spill` fails closed

## Known limits (not blockers)

- Native still ~26× eager on tiny MLP (dispatch/setup tax); beats nothing magical vs prior Python SC
- Activation spill file bytes stay Python-owned (tensor serialize)
- Virtual accelerators remain simulated
