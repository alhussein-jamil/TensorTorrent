# Heterogeneous hardware model

## Resource graph

Every production machine is discovered as independent resources:

**Compute:** CPU sockets, NUMA worker pools, discrete GPUs, integrated GPUs, other accelerators, copy engines.

**Memory:** NUMA RAM regions, pinned host pools, unified shared memory, per-device VRAM, disk cache, NVMe devices.

**Links:** CPU-local, NUMA interconnects, PCIe, NVLink / Infinity Fabric / CXL when exposed, shared-memory paths, host-staged fallbacks, storage links.

The planner never assumes:

- identical GPU vendors, speeds, VRAM, dtypes, kernels, or P2P connectivity
- symmetric transfer bandwidth
- one CPU socket or uniform memory latency
- CUDA as the only accelerator backend

## Backend contract

```python
class ExecutionBackend:
    def discover_devices(self) -> ResourceGraph: ...
    def supported_ops(self, device: ComputeResource) -> tuple[str, ...]: ...
    def supported_dtypes(self, device: ComputeResource) -> tuple[str, ...]: ...
    def enumerate_kernels(self, region, device: ComputeResource) -> list[KernelCandidate]: ...
    def benchmark(self, candidate: KernelCandidate) -> BenchmarkResult: ...
    def compile(self, region: RegionSource, candidate: KernelCandidate) -> CompiledRegion: ...
    def execute(self, executable: CompiledRegion, inputs: Sequence[Any]) -> tuple[Any, ...]: ...
    def transfer_capabilities(self, source, destination) -> TransferCapability: ...
```

`compile` must return a `CompiledRegion` holding a real callable; the dataclass
rejects anything that is not callable, so a backend cannot report success with a
status dictionary. `execute` returns the region's real output tensors.

Capabilities are **queried**, not inferred from vendor names. Mixed-vendor plans may coexist; when direct communication is unavailable the planner models **host-staged** transfers instead of declaring the machine unsupported.

## Maximal planning

The planner searches subsets:

- CPU only — **exercised on this host**
- each GPU independently — **untested**, no GPU available
- all GPUs — **untested**
- all GPUs + selected CPU cores — **planned**, cross-device dataflow is not implemented
- pipeline / tensor partitions across unequal devices — **simulated**, sizing is
  covered by unit tests against the analytic transfer model only
- independent branches and separate storage pipelines — **exercised on this host** for
  CPU branches

A device is included only when it reduces critical-path latency, increases throughput, enables a larger model, or improves another selected objective. Plans report why each resource was selected or excluded.

## Two-stage compilation

Portable artifacts are hardware-independent. On each deployment machine StreamCompiler:

1. Discovers hardware and topology
2. Loads valid profiles
3. Benchmarks each region on the available devices
4. Builds backend executables
5. Searches a global plan
6. Validates memory feasibility
7. Measures promising plans
8. Caches the specialized artifact

Artifacts are regenerated when fingerprint inputs change (hardware, topology, drivers, runtime/backend versions, resource limits).

## Validation

`streamcompiler validate-hardware` / `doctor --full` must be run on production machines. Missing GPUs on a development host are reported as `unsupported_capability` / `skipped`, never as successful accelerator validation.
