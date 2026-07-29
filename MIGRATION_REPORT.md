# Hybrid Rust migration report

Date: 2026-07-29

## Architecture before → after

**Before:** Pure-Python research runtime. Python owned schedule validation,
discrete-event simulation, residency/`CopyStore`, `ScheduleExecutor` DAG loop,
storage packs, and profiling.

**After:** Hybrid control/data plane.

- Python: `torch.export`, FX/regions, planner, public API, tensor instruction bodies.
- Rust: typed schedule model, validation, serde, residency/allocations, simulator,
  event-driven dispatcher (GIL released), virtual backend, storage validation,
  profiler DB, PyO3 bindings (`streamcompiler._native`).

Public path: `compile()` → `ScheduleExecutor.run` → Rust `execute_schedule` with
per-instruction Python handlers for Load/Transfer/Compute/Evict/Release.

## Migration stages completed

| Stage | Status |
|-------|--------|
| 1 Native schedule model | Done |
| 2 Rust simulator | Done (opt-in via `STREAMCOMPILER_NATIVE_SIM=1`; Python remains planner oracle) |
| 3 Residency/memory | Done (Rust authoritative types + tests; Python CopyStore still used for torch values) |
| 4 Event-driven runtime | Done (public path) |
| 5 Python compute bridges | Partial (instruction callback; not yet batched opaque handle tables) |
| 6 Storage runtime | Partial (manifest/pread/cache in Rust; Python packs still primary I/O) |
| 7 Profiling | Partial (Rust ProfileDatabase; Python BackendProfiler still used) |
| Switch public path | Done for scheduling |
| Remove obsolete Python DAG loop as default | Done (fallback only with `STREAMCOMPILER_ALLOW_PYTHON_RUNTIME=1`) |
| Delete Python instruction bodies | Not yet (still required for torch tensors) |

## Crates added

`streamcompiler-core`, `memory`, `simulator`, `runtime`, `backend-api`,
`virtual-backend`, `storage`, `profiler`, `python` under `crates/`.

## Public API changes

None required. `streamcompiler.compile(...)` / `compiled(x)` unchanged.

New env flags:

- `STREAMCOMPILER_ALLOW_PYTHON_RUNTIME=1` — allow Python DAG loop if native missing
- `STREAMCOMPILER_NATIVE_SIM=1` — use Rust discrete-event simulator from Python

## FFI / unsafe inventory

- PyO3 extension module (safe bindings; no manual unsafe in Python crate source)
- `streamcompiler-backend-api` C ABI stubs (`#[no_mangle] unsafe extern "C"`) —
  always return `SC_ERR` until a vendor links a real backend

## Tests / commands run

```
cargo fmt
cargo clippy --workspace --all-targets --exclude streamcompiler-python -- -D warnings
cargo test --workspace --exclude streamcompiler-python
cargo test -p streamcompiler-core --test proptest_schedule
cargo bench -p streamcompiler-runtime --bench schedule_overhead -- --quick
maturin develop
pytest tests/e2e/test_compile_execute.py tests/unit/test_native_runtime.py \
       tests/unit/test_schedule_driven_runtime.py tests/simulation/test_discrete_event.py
```

## Numerical correctness

CPU MLP / linear / branch e2e: `max_abs_err = 0.0` vs eager on this machine.

## Latency / scheduler overhead (this CPU-only VM)

| Metric | Value |
|--------|-------|
| Criterion empty schedule (pure Rust) | ~96 ns |
| Criterion linear 64 dry-run (pure Rust) | ~60.6 µs (**~0.95 µs/op**) |
| Python→native object convert + dry-run | dominated by PyO3 conversion (~14 µs/op) |
| Python→native JSON dry-run 256 ops | ~10.5 µs/op (parse + schedule) |
| Public compile MLP median | ~6.7 ms vs eager ~0.21 ms (~32×) — Python region/tensor handlers dominate |
| Numerical | `max_abs_err = 0.0` |

Native in-process scheduling meets the spirit of the <20 µs/op goal on Criterion
empty/linear dry-run. End-to-end inference still limited by Stage-5 Python Compute
callbacks (not scheduler bookkeeping).


## Still simulated / experimental / needs GPU

- Virtual accelerator backend: **simulated**
- CUDA / ROCm / multi-GPU: **untested** (no GPU on this VM)
- Rust simulator peak-memory parity with Python oracle: **experimental** (opt-in)
- Native pack streaming replacing Python pread path: **partial**

## First CUDA machine validation procedure

1. Install CUDA toolkit + matching `torch` wheel.
2. `maturin develop && pytest -q -m gpu`
3. `streamcompiler doctor --full`
4. Compile a tiny MLP with `devices="cuda:0"`; assert numerical close to eager.
5. Confirm residency rejects host/device alias mistakes.
6. Record measured (not simulated) transfer/compute profiles.
7. Do not claim ROCm/multi-GPU until those devices are present and tested.
