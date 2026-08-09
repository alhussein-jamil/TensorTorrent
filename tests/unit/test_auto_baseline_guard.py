from types import SimpleNamespace

from tensortorrent.compile.entry import _plan_uses_only_cpu


def _specialized(*devices: str) -> SimpleNamespace:
    return SimpleNamespace(plan=SimpleNamespace(devices_used=devices))


def test_cpu_only_plan_is_recognized_as_baseline() -> None:
    assert _plan_uses_only_cpu(_specialized("cpu_numa_0"))
    assert _plan_uses_only_cpu(_specialized("cpu_0", "cpu_numa_1"))


def test_accelerator_or_mixed_plan_is_not_cpu_baseline() -> None:
    assert not _plan_uses_only_cpu(_specialized("cuda_gpu_0"))
    assert not _plan_uses_only_cpu(_specialized("cpu_numa_0", "cuda_gpu_0"))
    assert not _plan_uses_only_cpu(_specialized())
