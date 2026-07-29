# Execution and communication backends

Backend-specific code is isolated behind stable interfaces. The planner queries
capabilities; it does not branch on vendor names.

## ExecutionBackend contract

Implemented by:

- `CpuBackend` (`cpu`) — **implemented and tested**; always available when PyTorch is
  present, and the only backend the test suite has executed end-to-end on real silicon
- `MockAccelBackend` (`mock_accel`) — **implemented (virtual)**; host-backed accelerator
  for schedule/residency/overlap tests; never auto-discovered (`available()` is False);
  inject via `make_mock_accel_graph(...)` + `compile(..., machine=...)`
- `CudaBackend` (`cuda`) — **untested**; NVIDIA devices when `torch.cuda` is usable
- `RocmBackend` (`rocm`) — **untested**; AMD devices when the ROCm/HIP runtime is present
- `MpsBackend` (`mps`) — **untested**; Apple Metal Performance Shaders
- `SyclBackend` (`sycl`) — **untested**; Intel XPU / SYCL when `torch.xpu` or `dpctl` works
- `OpenClVulkanBackend` (`opencl`, `vulkan`) — **planned**; raises
  `UnsupportedFeatureError` rather than pretending to compile

Each backend must implement discovery, op/dtype queries, kernel enumeration,
benchmarking, compile, execute, and transfer capability reporting. Compilation and
execution for every PyTorch-backed device share `backends/torch_device.py`, so the
untested backends run the same code as CPU with a different `torch.device`; what is
unverified is the device, not the compiler logic. Requesting an absent device raises
`BackendError` instead of silently falling back.

Region realization defaults to eager FX subgraphs. With
`CompileConfig.use_torch_compile=True`, `torch_device.compile_region_for_torch_device`
wraps the module in `torch.compile` (Inductor by default), measures it against
eager FX on the specialization examples, and keeps Inductor only when it is not
slower; otherwise the real eager FX callable remains the executable.

## Communication backends

Only Gloo has actually run here; the rest are selected by capability query but have
not been executed on this hardware.

- NCCL — **untested**; CUDA collectives when available
- RCCL — **untested**; ROCm collectives when available
- oneCCL — **untested**; Intel collectives when available
- Gloo — **implemented and tested**; host/CPU collectives
- host-staged — always-available fallback for mixed-vendor or missing P2P

`select_communication_backend(devices)` chooses the first capable backend and
otherwise returns host-staged. Missing direct interconnects never make the
whole machine unsupported.
