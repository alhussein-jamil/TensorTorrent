# StreamCompiler

Heterogeneous streaming compiler and runtime for **PyTorch inference**.

StreamCompiler treats a production machine as a **resource graph**: independent compute devices, memory tiers, and transfer links. It discovers that graph at deployment time and specializes an execution plan for the actual hardware — including mixed-vendor GPUs, multi-socket NUMA CPUs, and host-staged fallbacks when peer-to-peer links are missing.

The current development host may be CPU-only. That is an environment limitation, not an architectural assumption. GPU and mixed-vendor paths are validated on production machines via the hardware suite.

## Architecture

```
core compiler
  hardware-independent graph analysis
  heterogeneous IR
  cost model
  planner
  simulator
  storage planning
  execution-plan format

execution backends (capability-queried)
  CPU | CUDA | ROCm | MPS | Intel/SYCL | …

communication backends
  NCCL | RCCL | oneCCL | Gloo | host-staged fallback
```

Compilation has two stages:

1. **Portable compilation** — export, normalize, lower to heterogeneous IR, alias/liveness, packed weights, candidate partitions.
2. **Machine specialization** — discover hardware/topology, load profiles, benchmark missing candidates, search a global plan, validate memory, measure, cache a machine-specific artifact.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Deployment commands

```bash
streamcompiler doctor --full
streamcompiler profile --all-resources
streamcompiler validate-hardware
streamcompiler benchmark-topology
streamcompiler autotune model_artifact/
```

Validation reports distinguish:

- hardware detected
- backend available
- backend compiled
- basic execution validated
- concurrent execution validated
- numerical correctness validated
- performance characterized
- unsupported capability
- fallback selected

## Design rules

- Do not assume identical GPUs, CUDA-only stacks, one CPU socket, unified memory latency, or direct P2P.
- Do not force every resource busy; include a device only when it improves the selected objective.
- Planner code queries `ExecutionBackend` contracts — no scattered vendor conditionals.
- Never treat missing accelerators on a development machine as proof that accelerator execution works.

See [docs/architecture.md](docs/architecture.md), [docs/heterogeneous_hardware.md](docs/heterogeneous_hardware.md),
[docs/backends.md](docs/backends.md), [docs/deployment.md](docs/deployment.md),
[docs/roadmap.md](docs/roadmap.md), and [docs/anti_patterns.md](docs/anti_patterns.md).

## License

Apache-2.0
