"""Host topology and ISA probes are OS-dispatched, not Linux-only."""

from __future__ import annotations

import pytest

from tensortorrent.hardware.topology import cpu_vector_isas, read_host_topology
from tensortorrent.platform import detect_os, normalize_arch


def test_host_topology_is_nonempty() -> None:
    topo = read_host_topology()
    assert topo.sockets >= 1
    assert topo.numa_nodes == [0] or all(n >= 0 for n in topo.numa_nodes)
    if detect_os() == "macos":
        assert topo.numa_nodes == [0]
        assert topo.cores_per_socket is None or topo.cores_per_socket >= 1


@pytest.mark.skipif(detect_os() != "macos" or normalize_arch() != "aarch64", reason="Apple Silicon only")
def test_macos_arm_reports_neon() -> None:
    assert "neon" in cpu_vector_isas()
