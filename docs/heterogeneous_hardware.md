# Heterogeneous hardware model

## Resource graph

Every production machine is discovered as independent resources:

- **Compute:** CPU sockets, NUMA pools, discrete/integrated GPUs, other
  accelerators, copy engines
- **Memory:** NUMA RAM, pinned host, unified shared, per-device VRAM, disk/NVMe
- **Links:** CPU-local, NUMA, PCIe, NVLink / Infinity Fabric / CXL when exposed,
  shared-memory, host-staged fallbacks, storage links

The planner never assumes identical GPUs, symmetric bandwidth, one CPU socket,
uniform memory latency, or CUDA-only stacks.

Backend contracts and status live in [backends.md](backends.md). Capabilities are
queried; mixed-vendor plans may use host-staged transfers instead of declaring
the machine unsupported.

## Maximal planning

The planner searches device subsets and keeps a device only when it improves the
selected objective. Working-set bytes that exceed a device's allocatable memory
are hard-filtered (not soft-penalized only).

Exercised on CPU hosts and on CPU + `mock_accel` (including multi-device
host-staged graphs). Real multi-GPU concurrent execution remains unvalidated
until run on accelerator hardware — see [roadmap.md](roadmap.md).

## Two-stage compilation

1. **Portable** — export, regions, IR, packs (hardware-independent)
2. **Specialize** — discover, measure, plan, compile backends, validate memory,
   measure concurrency, cache fingerprint-gated artifacts

Regenerate when fingerprint inputs change (hardware, drivers, PyTorch/backends,
resource limits). Deployment sequence: [deployment.md](deployment.md).
