# Deployment on heterogeneous production machines

Development hosts may be CPU-only. Production specialization and validation must
run on the target workstation or server.

## Recommended sequence

```bash
# 1. Inventory + backend readiness
streamcompiler doctor --full --json artifacts/doctor.json

# 2. Discover topology and profile resources
streamcompiler profile --all-resources --output artifacts/profile
streamcompiler benchmark-topology --output artifacts/topology.json

# 3. Full hardware validation suite
streamcompiler validate-hardware --stress --output artifacts/validation_report.json

# 4. Specialize a portable model artifact for this machine
streamcompiler autotune model_artifact/ --profile
```

## Interpreting validation statuses

| Status | Meaning |
|--------|---------|
| `hardware_detected` | Resource present in the discovered graph |
| `backend_available` | Runtime libraries usable |
| `backend_compiled` | Capability/dtype enumerated for a device |
| `basic_execution_validated` | Smoke execution path succeeded |
| `concurrent_execution_validated` | Multi-device concurrency path exercised |
| `numerical_correctness_validated` | Compared against eager PyTorch |
| `performance_characterized` | Measured latency/bandwidth sample stored |
| `unsupported_capability` | Not present on this machine (honest negative) |
| `fallback_selected` | e.g. host-staged collectives instead of NCCL/RCCL |
| `failed` | Hard failure requiring attention |
| `skipped` | Not applicable without required hardware |

Absence of GPUs on a development machine yields `unsupported_capability` /
`skipped` for accelerator checks. That is **not** production GPU validation.

## When to respecialize

Regenerate machine-specific artifacts when any fingerprint input changes:

- hardware inventory or topology
- drivers / firmware
- PyTorch or backend runtime versions
- resource limits (cgroup, visible devices)
- profile cache invalidation
