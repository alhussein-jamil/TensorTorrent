"""Host platform detection, capabilities, and native library-path wiring."""

from __future__ import annotations

import os
from typing import Any

from tensortorrent.platform import (
    apply_native_library_path,
    detect,
    detect_os,
    emit_native_library_exports,
    native_library_path_vars,
    normalize_arch,
    supports_process_workers,
    torch_index_url,
)


def test_detect_os_aliases() -> None:
    assert detect_os("linux") == "linux"
    assert detect_os("linux2") == "linux"
    assert detect_os("darwin") == "macos"
    assert detect_os("win32") == "windows"
    assert detect_os("aix") == "unknown"


def test_normalize_arch_aliases() -> None:
    assert normalize_arch("x86_64") == "x86_64"
    assert normalize_arch("amd64") == "x86_64"
    assert normalize_arch("arm64") == "aarch64"
    assert normalize_arch("aarch64") == "aarch64"


def test_linux_is_production() -> None:
    host = detect(sys_platform="linux", machine="x86_64", python="3.13.2")
    assert host.supported
    assert host.support_level == "production"
    assert host.label == "linux/x86_64 py3.13.2"
    assert host.notes == ()


def test_macos_is_development() -> None:
    host = detect(sys_platform="darwin", machine="arm64", python="3.13.2")
    assert host.os == "macos"
    assert host.arch == "aarch64"
    assert host.support_level == "development"
    assert host.supported
    assert any("MPS" in note for note in host.notes)


def test_windows_is_unsupported() -> None:
    host = detect(sys_platform="win32", machine="AMD64", python="3.12.0")
    assert host.os == "windows"
    assert host.arch == "x86_64"
    assert not host.supported
    assert host.support_level == "unsupported"


def test_process_workers_linux_only() -> None:
    assert supports_process_workers(sys_platform="linux")
    assert not supports_process_workers(sys_platform="darwin")
    assert not supports_process_workers(sys_platform="win32")


def test_live_detect_matches_runtime() -> None:
    host = detect()
    assert host.os == detect_os()
    assert host.arch == normalize_arch()
    if host.os == "macos":
        assert host.supported
        assert not supports_process_workers()
    elif host.os == "linux":
        assert host.support_level == "production"
        assert supports_process_workers()


def test_library_path_vars_are_host_specific() -> None:
    assert native_library_path_vars(sys_platform="linux") == ("LD_LIBRARY_PATH",)
    assert native_library_path_vars(sys_platform="darwin") == (
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
    )
    assert native_library_path_vars(sys_platform="win32") == ()


def test_apply_native_library_path_linux_only_ld(monkeypatch: Any) -> None:
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/old")
    apply_native_library_path("/lib", sys_platform="linux")
    assert os.environ["LD_LIBRARY_PATH"].startswith("/lib")
    assert "DYLD_LIBRARY_PATH" not in os.environ


def test_apply_native_library_path_macos_only_dyld(monkeypatch: Any) -> None:
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    apply_native_library_path("/lib", sys_platform="darwin")
    assert os.environ["DYLD_LIBRARY_PATH"].startswith("/lib")
    assert "LD_LIBRARY_PATH" not in os.environ


def test_apply_native_library_path_skips_empty(monkeypatch: Any) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/keep")
    apply_native_library_path("", sys_platform="linux")
    assert os.environ["LD_LIBRARY_PATH"] == "/keep"


def test_emit_native_library_exports_linux() -> None:
    text = emit_native_library_exports("/opt/lib", sys_platform="linux")
    assert 'export LD_LIBRARY_PATH="/opt/lib' in text
    assert "DYLD" not in text


def test_apply_native_library_path_is_idempotent(monkeypatch: Any) -> None:
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    apply_native_library_path("/lib", sys_platform="linux")
    apply_native_library_path("/lib", sys_platform="linux")
    assert os.environ["LD_LIBRARY_PATH"] == "/lib"


def test_torch_cpu_index_is_generic() -> None:
    assert torch_index_url(flavor="cpu") == "https://download.pytorch.org/whl/cpu"
    assert torch_index_url(flavor="cuda") is None
