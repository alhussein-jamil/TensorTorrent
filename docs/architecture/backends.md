# Backends

Backends expose capabilities to the compiler/runtime. Planner logic is expected to reason about capabilities and resources rather than scatter vendor-name conditionals through the search implementation.

## Built-in execution backends

| Backend | Backend ID / path | Execution model |
| --- | --- | --- |
| CPU | `cpu` | host execution, NUMA-aware resource reporting |
| NVIDIA CUDA | `cuda` | torch-backed CUDA execution and measurement |
| AMD ROCm | `rocm` | torch/HIP-backed execution and measurement |
| Intel XPU | `xpu` | capability-gated `torch.xpu` execution |
| Virtual | `mock_accel` / Rust virtual backend | deterministic simulated accelerator for tests |

CUDA, ROCm, and XPU support depend on the installed PyTorch build and target host. A device appearing in discovery is not itself a production-support claim.

## Backend responsibilities

An execution backend is responsible for the capabilities required by its path, including combinations of:

- device discovery and metadata,
- memory/capacity reporting,
- kernel candidate enumeration,
- region measurement,
- region compilation,
- execution,
- copy/transfer behavior,
- health checks.

The core planner consumes backend candidates and resource-graph information rather than embedding CUDA-specific placement policy.

## Region implementation selection

For torch-backed regions, specialization can use eager FX and `torch.compile`, with optional deeper competitive profiling depending on `profile_level`. Under competitive/full profiling, available implementations are measured on example inputs and the faster viable implementation is retained.

`use_torch_compile=True` does not mean a compiled implementation is kept unconditionally.

## Communication

TensorTorrent can select collective/communication paths exposed by the environment, including NCCL, RCCL, oneCCL, Gloo, and host-staged fallbacks where applicable.

Host-staged communication is the portability fallback; it should not be read as equivalent in performance to a native peer/collective path.

## Plugin backends

Third-party packages can register backends through the `tensortorrent.backends` entry-point group:

```toml
[project.entry-points."tensortorrent.backends"]
my_accelerator = "my_package.backend:create_backend"
```

The entry point may expose an `ExecutionBackend` instance, subclass, or zero-argument factory according to the backend registry contract.

Plugin failures are isolated so a broken optional backend does not prevent unrelated built-in backends from starting. Set:

```bash
export TENSORTORRENT_DISABLE_BACKEND_PLUGINS=1
```

for hermetic environments that should not load external backend plugins.

## Hardware support policy

A backend should be described as supported on a production host only after target-machine validation has exercised basic execution and numerical parity.

Use:

```bash
tensortorrent doctor --full
tensortorrent validate-hardware --output artifacts/validation_report.json
```

Real hardware tests live under `tests/hardware/` and intentionally do not run as part of the architecture-neutral CI gate.
