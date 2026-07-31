# Deployment

Specialize and validate on the **target** machine — not only the laptop that
built the portable artifact.

```bash
streamcompiler doctor --full --json artifacts/doctor.json
streamcompiler profile --all-resources --output artifacts/profile
streamcompiler benchmark-topology --output artifacts/topology.json
streamcompiler validate-hardware --stress --output artifacts/validation_report.json
streamcompiler autotune model_artifact/ --profile
```

## Status meanings

| Status | Meaning |
| --- | --- |
| `hardware_detected` | Present in the resource graph |
| `backend_available` | Runtime libraries usable |
| `backend_compiled` | Capability / dtype enumerated |
| `basic_execution_validated` | Smoke path succeeded |
| `concurrent_execution_validated` | Measured overlapping execution |
| `numerical_correctness_validated` | Matches eager PyTorch |
| `performance_characterized` | Latency / bandwidth sample stored |
| `unsupported_capability` | Capability absent or unusable |
| `fallback_selected` | e.g. host-staged collectives |
| `failed` / `skipped` | Hard fail / not applicable |

GPU absence on a development host is `unsupported` / `skipped`. Validate GPUs on
the target machine.

Respecialize when hardware, drivers, PyTorch/backends, or resource limits change.
