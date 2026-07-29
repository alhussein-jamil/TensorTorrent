# Native Rust migration report

Date: 2026-07-29 (completion audit)

## Public API (unchanged)

```python
compiled = streamcompiler.compile(model, example_inputs=inputs)
result = compiled(*inputs)
```

## Architecture

```
Python  torch.export / FX / Inductor / PyTrees / public API
        Compute regions + tensor↔bytes materialize only
            |
NativeCompiledArtifact
            |
NativeExecutionContext   (sole runtime authority per forward)
  residency versions/aliases/leases/allocs
  events · streams · copy engines · links · io queues
  StreamingStore · spills · cancel · VirtualBackend map
            |
Rust dispatcher  Prefetch Load Transfer Record Wait Release Evict Spill
            |
Python  batched Compute waves (inline width≤1) / pooled overlap (width>1)
```

## Criteria checklist

| # | Item | Status |
|---|------|--------|
| 1 | One Rust execution context | **Done** — Rust owns residency/leases/allocs/events/resources. Native `CopyStore.value_bag_only=True` (no Python version/stale/AllocationTable authority). |
| 2 | `non_compute_python_callbacks=0` | **Done** — public `instruction_handler=None`. Spill/param = materialization, not instruction callbacks. |
| 3 | Views / aliases | **Done** — storage_id, nbytes, offset, shape, strides, dtype; shared storage → shared alloc. |
| 4 | Stream metadata operational | **Done** — serialize on `stream_id` / engine, not whole device; real capacity wait on engines/queues; link contention via `bytes_in_flight`. |
| 5 | Virtual backend public | **Done** — mock Compute/Transfer use Rust `VirtualBackend` (buffers/streams/pending events/capacity). Labelled `simulated`. `DeviceStreams` unused on native path (legacy/oracle only). |
| 6 | Storage in Rust | **Done** — `StreamingStore` owns prefetch/inflight/byte cache. Native path skips Python prefetch worker. Decoded tensors keep `bytearray` backing (no `.clone()`). |
| 7 | Strict simulator | **Done** — unknown Wait / missing transfer → error; OOM → infeasible. Default Rust sim; Python oracle only if `STREAMCOMPILER_PYTHON_SIM`. |
| 8 | No fake / duplicate exec | **Done** — persistent params via acquire+mirror (not `_exec_load`); no mid-forward restart; Python post-drop only clears value bag after Rust release. |
| 9 | Batch ready Computes | **Done** — region-only path always wave-batches in one GIL call (incl. width>1). Hybrid pooled path wave-batches ready Computes while Prefetch I/O overlaps. |
| 10 | Prune | **Done for production** — public instruction-handler path removed. Legacy DAG lives under `testing.legacy_runtime` (bench/oracle only, never auto). Planner residency + `_exec_*` helpers remain for oracle/tests. |

## Also required

| Item | Status |
|------|--------|
| Counter tests (not only equality) | **Done** — `native_gate`, `test_native_zero_callbacks`, `assert_zero_non_compute_callbacks` |
| Fresh wheel + all tests | **Done** — `maturin build` + `pip install --force-reinstall` wheel; **342 passed** |
| Benchmarks eager + old + native | **Done** — `benchmarks/compare_runtimes.py` |
| Concurrent same-module forwards | **Done** — per-forward cancel + tests |
| No forward restart after begin | **Done** |

## Measured (fresh wheel, CPU-only)

`native-gate`: `non_compute=0` compute=2

Three-way median (Sequential Linear 256, batch 32) — see `benchmarks/results/native_forward_*.json`:

- Eager PyTorch
- Legacy Python DAG (`testing.legacy_runtime`, oracle only)
- Native data plane (`non_compute_python_callbacks=0`)

## Simulated vs real

No GPU on this machine. Mock / hetero accelerator results are **simulated** via `VirtualBackend`.

## Gate commands (all green)

```
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
maturin build --release
pip install --force-reinstall target/wheels/*.whl
pytest -m 'not property'
ruff check .
mypy src
```
