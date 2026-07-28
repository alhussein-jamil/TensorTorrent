# Execution and communication backends

Backend-specific code is isolated behind stable interfaces. The planner queries
capabilities; it does not branch on vendor names.

## ExecutionBackend contract

Implemented by:

- `CpuBackend` (`cpu`) — always available when PyTorch is present
- `CudaBackend` (`cuda`) — NVIDIA devices when `torch.cuda` is usable
- `RocmBackend` (`rocm`) — AMD devices when ROCm/HIP runtime is present
- `MpsBackend` (`mps`) — Apple Metal Performance Shaders
- `SyclBackend` (`sycl`) — Intel XPU / SYCL when `torch.xpu` or `dpctl` works

Each backend must implement discovery, op/dtype queries, kernel enumeration,
benchmarking, compile, execute, and transfer capability reporting.

## Communication backends

- NCCL — CUDA collectives when available
- RCCL — ROCm collectives when available
- oneCCL — Intel collectives when available
- Gloo — host/CPU collectives
- host-staged — always-available fallback for mixed-vendor or missing P2P

`select_communication_backend(devices)` chooses the first capable backend and
otherwise returns host-staged. Missing direct interconnects never make the
whole machine unsupported.
