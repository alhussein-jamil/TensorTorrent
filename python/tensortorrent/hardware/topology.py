"""Host CPU topology: sockets, NUMA nodes, cores, vector ISAs.

Linux uses lscpu + sysfs. macOS uses sysctl. Other hosts report a single domain.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

from tensortorrent.hardware.fingerprint import DEFAULT_SYSTEM_PROBE_TIMEOUT_S
from tensortorrent.platform import detect_os, normalize_arch


@dataclass
class TopologyHint:
    sockets: int
    numa_nodes: list[int]
    cores_per_socket: int | None
    threads_per_core: int | None


def read_host_topology() -> TopologyHint:
    os_name = detect_os()
    if os_name == "linux":
        return _linux_topology()
    if os_name == "macos":
        return _macos_topology()
    return TopologyHint(sockets=1, numa_nodes=[0], cores_per_socket=None, threads_per_core=None)


def _sysfs_numa_nodes() -> list[int]:
    base = "/sys/devices/system/node"
    if not os.path.isdir(base):
        return [0]
    nodes = []
    for name in sorted(os.listdir(base)):
        m = re.fullmatch(r"node(\d+)", name)
        if m:
            nodes.append(int(m.group(1)))
    return nodes or [0]


def cpu_vector_isas() -> tuple[str, ...]:
    os_name = detect_os()
    if os_name == "linux":
        return _linux_cpu_isas()
    if os_name == "macos":
        return _macos_cpu_isas()
    return ()


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=DEFAULT_SYSTEM_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return ""


def _linux_topology() -> TopologyHint:
    sockets = 1
    cores_per_socket = None
    threads_per_core = None
    out = _run(["lscpu"])
    for line in out.splitlines():
        if line.startswith("Socket(s):"):
            sockets = max(1, int(line.split(":")[1].strip()))
        elif line.startswith("Core(s) per socket:"):
            cores_per_socket = int(line.split(":")[1].strip())
        elif line.startswith("Thread(s) per core:"):
            threads_per_core = int(line.split(":")[1].strip())
    return TopologyHint(
        sockets=sockets,
        numa_nodes=_sysfs_numa_nodes(),
        cores_per_socket=cores_per_socket,
        threads_per_core=threads_per_core,
    )


def _sysctl_int(name: str) -> int | None:
    text = _run(["sysctl", "-n", name]).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _macos_topology() -> TopologyHint:
    packages = max(1, _sysctl_int("hw.packages") or 1)
    physical = _sysctl_int("hw.physicalcpu")
    logical = _sysctl_int("hw.logicalcpu")
    cores = (physical // packages) if physical else None
    threads = (logical // physical) if physical and logical else None
    return TopologyHint(
        sockets=packages,
        numa_nodes=[0],
        cores_per_socket=cores,
        threads_per_core=threads,
    )


_ISA_NAMES = (
    "avx",
    "avx2",
    "avx512f",
    "avx512_bf16",
    "avx512_fp16",
    "amx_bf16",
    "amx_int8",
    "neon",
    "sve",
)


def _linux_cpu_isas() -> tuple[str, ...]:
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("flags") or line.startswith("Features"):
                    flags.update(line.split(":", 1)[1].split())
    except OSError:
        return ()
    return tuple(sorted(name for name in _ISA_NAMES if name in flags))


def _macos_cpu_isas() -> tuple[str, ...]:
    if normalize_arch() == "aarch64":
        return ("neon",)
    found: list[str] = []
    mapping = (
        ("hw.optional.avx1_0", "avx"),
        ("hw.optional.avx2_0", "avx2"),
        ("hw.optional.avx512f", "avx512f"),
    )
    for key, name in mapping:
        if _sysctl_int(key):
            found.append(name)
    return tuple(found)
