# Changelog

## 0.1.0

Initial heterogeneous streaming compiler milestone:

- Resource graph IR with independent compute, memory, and transfer resources
- ExecutionBackend contract for CPU, CUDA, ROCm, MPS, and SYCL
- Communication backends including host-staged mixed-vendor fallback
- Two-stage portable compile + machine specialization
- Maximal planner with subset search and inclusion/exclusion reasons
- Discrete-event simulator and chrome-trace visualization
- Packed model storage format
- Hardware validation suite and CLI:
  `doctor`, `profile`, `validate-hardware`, `benchmark-topology`, `autotune`
- CPU-path correctness tests; accelerator paths reported honestly when absent
