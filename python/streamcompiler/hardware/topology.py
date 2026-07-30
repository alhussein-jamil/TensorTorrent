"""Improve CPU socket / NUMA discovery beyond a single homogenized pool."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass


@dataclass
class TopologyHint:
    sockets: int
    numa_nodes: list[int]
    cores_per_socket: int | None
    threads_per_core: int | None


def read_lscpu_topology() -> TopologyHint:
    sockets = 1
    cores_per_socket = None
    threads_per_core = None
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError):
        out = ""
    for line in out.splitlines():
        if line.startswith("Socket(s):"):
            sockets = max(1, int(line.split(":")[1].strip()))
        elif line.startswith("Core(s) per socket:"):
            cores_per_socket = int(line.split(":")[1].strip())
        elif line.startswith("Thread(s) per core:"):
            threads_per_core = int(line.split(":")[1].strip())
    numa = _sysfs_numa_nodes()
    return TopologyHint(
        sockets=sockets,
        numa_nodes=numa,
        cores_per_socket=cores_per_socket,
        threads_per_core=threads_per_core,
    )


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
