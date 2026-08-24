#!/usr/bin/env python3
"""Prepare a TensorTorrent source checkout on any supported host.

Stdlib only until ``uv`` / Rust exist. Detects the host, refuses unsupported
targets, installs missing toolchain pieces, then syncs and builds the native
extension.

    python3 tools/bootstrap.py
    python3 tools/bootstrap.py --check-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tt_platform import load as _load_platform  # noqa: E402

_platform = _load_platform()
detect = _platform.detect
torch_index_url = _platform.torch_index_url

SUPPORTED_PYTHON = ((3, 10), (3, 13))
DEFAULT_PYTHON = "3.13"
RUSTUP_URL = "https://sh.rustup.rs"
UV_URL = "https://astral.sh/uv/install.sh"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _prepend_path(directory: Path) -> None:
    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")


def ensure_uv() -> str:
    found = _which("uv")
    if found:
        return found
    installer = _which("curl")
    if installer is None:
        raise SystemExit("uv not found and curl missing — install uv from https://docs.astral.sh/uv/")
    run(["sh", "-c", f"curl -LsSf {UV_URL} | sh"])
    _prepend_path(Path.home() / ".local" / "bin")
    found = _which("uv")
    if found is None:
        raise SystemExit("uv install finished but uv is not on PATH")
    return found


def ensure_rust() -> None:
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        _prepend_path(cargo_bin)
    if _which("cargo") and _which("rustc"):
        return
    if _which("curl") is None:
        raise SystemExit("Rust toolchain missing and curl missing — install rustup from https://rustup.rs/")
    run(["sh", "-c", f"curl --proto '=https' --tlsv1.2 -sSf {RUSTUP_URL} | sh -s -- -y --default-toolchain none"])
    _prepend_path(cargo_bin)
    if not (_which("cargo") and _which("rustc")):
        raise SystemExit("rustup install finished but cargo/rustc are not on PATH")


def _python_spec(requested: str) -> str:
    parts = requested.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError as exc:
        raise SystemExit(f"invalid --python {requested!r}") from exc
    lo, hi = SUPPORTED_PYTHON
    if (major, minor) < lo or (major, minor) > hi:
        raise SystemExit(f"Python {requested} outside supported {lo[0]}.{lo[1]}–{hi[0]}.{hi[1]}")
    return requested


def sync_and_build(uv: str, python: str, *, extra: str) -> None:
    run([uv, "python", "install", python])
    env = os.environ.copy()
    index = torch_index_url(flavor="cpu")
    sync = [uv, "sync", "--python", python, "--extra", extra]
    if index:
        sync.extend(["--no-install-package", "torch"])
    run(sync, env=env)
    if index:
        run([uv, "pip", "install", "--index-url", index, "torch"], env=env)
        run(
            [uv, "sync", "--python", python, "--extra", extra, "--reinstall-package", "tensortorrent"],
            env=env,
        )
    run([uv, "run", "maturin", "develop", "--profile", "release-quick"], env=env)


def check_only() -> int:
    host = detect()
    print(f"platform: {host.label} ({host.support_level})")
    for note in host.notes:
        print(f"  {note}")
    print(f"uv:    {_which('uv') or 'missing'}")
    print(f"cargo: {_which('cargo') or 'missing'}")
    print(f"rustc: {_which('rustc') or 'missing'}")
    if not host.supported:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up a TensorTorrent source checkout")
    parser.add_argument("--check-only", action="store_true", help="Print host/toolchain facts and exit")
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="CPython minor to provision (default: 3.13)")
    parser.add_argument("--extra", default="dev", help="uv extra to sync (default: dev)")
    args = parser.parse_args()

    if args.check_only:
        return check_only()

    host = detect()
    print(f"platform: {host.label} ({host.support_level})")
    for note in host.notes:
        print(f"  {note}")
    if not host.supported:
        return 2

    python = _python_spec(args.python)
    uv = ensure_uv()
    ensure_rust()
    sync_and_build(uv, python, extra=args.extra)
    run([uv, "run", "tensortorrent", "doctor"])
    print("bootstrap_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
