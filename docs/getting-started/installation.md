# Installation

TensorTorrent ships as a Python package with a native Rust extension. Install PyTorch first so you control the backend build used by your environment.

## Requirements

- Linux
- Python 3.10–3.13
- PyTorch 2.4 or newer
- A supported PyTorch accelerator build if you intend to use CUDA, ROCm, or Intel XPU

Windows is not a supported target. WSL2 may be useful for CPU-only development, but it is not a production target.

## PyPI

CPU example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install tensortorrent
```

For CUDA or ROCm, install the matching PyTorch build from the PyTorch installation instructions, then install TensorTorrent normally:

```bash
python -m pip install tensortorrent
```

TensorTorrent uses the already-installed PyTorch runtime for torch-backed accelerator execution.

## Verify the installation

```bash
tensortorrent doctor
```

`doctor` reports discovered resources, backend readiness, resource budgets, and relevant diagnostics. Discovery is not a production validation result.

On a deployment host, run:

```bash
tensortorrent validate-hardware --output artifacts/validation_report.json
```

Use `--stress` for a bounded soak or `--overnight` for the extended validation path.

## Build from source

The development workflow uses `uv`, Maturin, and Rust.

```bash
git clone https://github.com/alhussein-jamil/TensorTorrent.git
cd TensorTorrent
make sync
make doctor
```

`make sync` installs development dependencies and builds the native extension with the `release-quick` Cargo profile. Published wheels use the full release profile.

Manual equivalent:

```bash
uv sync --extra dev
uv run maturin develop --profile release-quick
```

## Validate a source checkout

```bash
make check
make native-gate
```

Real accelerator tests are opt-in because they can consume substantial VRAM and spill space:

```bash
make hardware-test
```

## Common installation failures

### `tensortorrent._native` cannot be imported

The native extension was not built for the active Python environment. For a source checkout, run:

```bash
uv run maturin develop --profile release-quick
```

### CUDA/ROCm/XPU is not discovered

Check the PyTorch build first. TensorTorrent cannot provide a GPU runtime that PyTorch itself does not expose.

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

Then inspect TensorTorrent's view:

```bash
tensortorrent doctor --full
```

### Container sees less RAM or fewer CPUs than the host

Expected. TensorTorrent resolves effective cgroup and affinity limits rather than sizing from host totals. See [Resource budgets](../product/resource_budgets.md).
