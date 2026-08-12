# Installation

Python package + native Rust extension. Install PyTorch first so you pick the backend build.

## Requirements

- Linux
- Python 3.10–3.13
- PyTorch 2.4+
- Matching PyTorch accelerator build if you want CUDA, ROCm, or Intel XPU

Windows not supported. WSL2 is fine for CPU-only hacking; not a production target.

## PyPI

CPU:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install tensortorrent
```

CUDA/ROCm: install the matching PyTorch wheel first, then:

```bash
python -m pip install tensortorrent
```

TensorTorrent uses whatever PyTorch you already have for torch-backed accelerator work.

## Verify

```bash
tensortorrent doctor
```

Reports discovery, backend readiness, budgets. Discovery alone is not production validation.

On a real host:

```bash
tensortorrent validate-hardware --output artifacts/validation_report.json
```

`--stress` for a soak, `--overnight` for the long path.

## Build from source

Needs `uv`, Maturin, Rust. Toolchain pinned in `rust-toolchain.toml` (**1.85.1** + `rustfmt` / `clippy`).

```bash
git clone https://github.com/alhussein-jamil/TensorTorrent.git
cd TensorTorrent
make sync
make doctor
```

`make sync` installs deps and builds the native extension with `release-quick`. Published wheels use full release.

Manual:

```bash
uv sync --extra dev
uv run maturin develop --profile release-quick
```

## Validate a source checkout

```bash
make check
make native-gate
```

Accelerator tests are opt-in (VRAM / spill can get large):

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
