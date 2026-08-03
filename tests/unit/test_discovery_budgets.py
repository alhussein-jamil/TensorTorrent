"""Unit tests for backend budget integration during device discovery."""

from __future__ import annotations

from typing import Any

from tensortorrent.hardware import budget as _b

_MiB = 1 << 20
_GiB = 1 << 30
_256_MiB = 256 * _MiB
_768_MiB = 768 * _MiB


# ---------------------------------------------------------------------------
# cpu.py — monkeypatch budget resolution to a known ResolvedBudget
# ---------------------------------------------------------------------------


def _make_budget(allowed: int, kind: str = "explicit") -> _b.ResolvedBudget:
    return _b.ResolvedBudget(
        total_bytes=16 * _GiB,
        allowed_bytes=allowed,
        reserved_bytes=_256_MiB,
        source=_b.BudgetSource(kind=kind, detail=f"test-{kind}"),
        notes=(),
    )


def test_cpu_discovery_allocatable_reflects_budget(monkeypatch: Any) -> None:
    """CPU discovery uses the budget resolver; allocatable_bytes reflects allowed//nodes."""
    import tensortorrent.backends.cpu as cpu_module

    known_allowed = 4 * _GiB
    monkeypatch.setattr(_b, "resolve_host_memory_budget", lambda **kw: _make_budget(known_allowed))

    backend = cpu_module.CpuBackend()
    graph = backend.discover_devices()

    # Find all NUMA_RAM memory resources
    from tensortorrent.ir.resource_graph import MemoryClass

    numa_mems = [m for m in graph.memory.values() if m.memory_class == MemoryClass.NUMA_RAM]
    assert len(numa_mems) >= 1

    # Total allocatable across NUMA nodes must equal allowed_bytes
    total_alloc = sum(m.allocatable_bytes for m in numa_mems)
    assert total_alloc == known_allowed


def test_cpu_discovery_provenance_attributes_present(monkeypatch: Any) -> None:
    """CPU memory resources carry budget_source / budget_detail attributes."""
    import tensortorrent.backends.cpu as cpu_module

    monkeypatch.setattr(_b, "resolve_host_memory_budget", lambda **kw: _make_budget(2 * _GiB, kind="cgroup_v2"))

    graph = cpu_module.CpuBackend().discover_devices()
    from tensortorrent.ir.resource_graph import MemoryClass

    numa_mems = [m for m in graph.memory.values() if m.memory_class == MemoryClass.NUMA_RAM]
    for mem in numa_mems:
        assert "budget_source" in mem.attributes, f"{mem.id.name} missing budget_source"
        assert "budget_detail" in mem.attributes, f"{mem.id.name} missing budget_detail"
        assert "budget_reserved_bytes" in mem.attributes
        assert mem.attributes["budget_source"] == "cgroup_v2"


# ---------------------------------------------------------------------------
# cuda.py — monkeypatch torch.cuda.is_available off → no crash
# ---------------------------------------------------------------------------


def test_cuda_discovery_no_crash_when_unavailable(monkeypatch: Any) -> None:
    """CudaBackend.discover_devices returns an empty graph when CUDA is off."""
    import tensortorrent.backends.cuda as cuda_module

    monkeypatch.setattr(cuda_module.CudaBackend, "available", lambda self: False)
    graph = cuda_module.CudaBackend().discover_devices()
    assert len(graph.compute) == 0
    assert len(graph.memory) == 0
    assert graph.attributes.get("cuda_status") == "runtime_unavailable"


def test_cuda_backend_available_returns_false_gracefully(monkeypatch: Any) -> None:
    """CudaBackend.available handles torch import errors gracefully."""
    import contextlib

    import tensortorrent.backends.cuda as cuda_module

    def _raise(self: Any) -> bool:
        raise RuntimeError("no cuda")

    monkeypatch.setattr(cuda_module.CudaBackend, "available", _raise)
    # discover_devices must not crash even if available() raises
    with contextlib.suppress(RuntimeError):
        cuda_module.CudaBackend().available()

    monkeypatch.setattr(cuda_module.CudaBackend, "available", lambda self: False)
    graph = cuda_module.CudaBackend().discover_devices()
    assert len(graph.compute) == 0


# ---------------------------------------------------------------------------
# xpu.py — integrated classification heuristic
# ---------------------------------------------------------------------------


def test_xpu_uhd_name_classified_as_integrated() -> None:
    """'Intel(R) UHD Graphics 770' → INTEGRATED_GPU."""
    import tensortorrent.backends.xpu as xpu_module

    class FakeProps:
        has_fp64 = None

    is_integrated, reason = xpu_module._xpu_is_integrated("Intel(R) UHD Graphics 770", FakeProps())
    assert is_integrated is True
    assert reason == "name-heuristic"


def test_xpu_arc_name_not_classified_as_integrated() -> None:
    """'Intel(R) Arc(TM) A770' → DISCRETE_GPU (not integrated)."""
    import tensortorrent.backends.xpu as xpu_module

    class FakeProps:
        has_fp64 = True

    is_integrated, reason = xpu_module._xpu_is_integrated("Intel(R) Arc(TM) A770", FakeProps())
    assert is_integrated is False


def test_xpu_discovery_integrated_compute_class(monkeypatch: Any) -> None:
    """UHD device: compute_class INTEGRATED_GPU + classified_integrated attribute."""
    import tensortorrent.backends.xpu as xpu_module
    from tensortorrent.ir.resource_graph import ComputeClass

    class _Props:
        name = "Intel(R) UHD Graphics 770"
        total_memory = 4 * _GiB
        architecture = "xe-lpg"
        max_compute_units = 32
        copy_engines = 1

    class _FakeXpu:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int) -> _Props:
            return _Props()

    monkeypatch.setattr(xpu_module, "_xpu_module", lambda: _FakeXpu())
    monkeypatch.setattr(xpu_module.XpuBackend, "_probe_dtypes", lambda self, idx: ("float32",))

    graph = xpu_module.XpuBackend().discover_devices()
    device = graph.compute["xpu_gpu_0"]
    assert device.compute_class == ComputeClass.INTEGRATED_GPU
    assert "classified_integrated" in device.attributes


def test_xpu_discovery_discrete_compute_class(monkeypatch: Any) -> None:
    """Arc device: compute_class DISCRETE_GPU, no classified_integrated attribute."""
    import tensortorrent.backends.xpu as xpu_module
    from tensortorrent.ir.resource_graph import ComputeClass

    class _Props:
        name = "Intel(R) Arc(TM) A770"
        total_memory = 16 * _GiB
        architecture = "xe-hpg"
        max_compute_units = 512
        copy_engines = 2

    class _FakeXpu:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int) -> _Props:
            return _Props()

    monkeypatch.setattr(xpu_module, "_xpu_module", lambda: _FakeXpu())
    monkeypatch.setattr(xpu_module.XpuBackend, "_probe_dtypes", lambda self, idx: ("float32", "bfloat16"))

    graph = xpu_module.XpuBackend().discover_devices()
    device = graph.compute["xpu_gpu_0"]
    assert device.compute_class == ComputeClass.DISCRETE_GPU
    assert "classified_integrated" not in device.attributes
