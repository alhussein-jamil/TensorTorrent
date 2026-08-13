from types import SimpleNamespace

import torch

from tensortorrent.config import CompileConfig
from tensortorrent.runtime import provisioning
from tensortorrent.runtime.pinning import (
    pin_for_dma,
    pinned_host_allocatable_bytes,
    should_pin_parameter_store,
)
from tensortorrent.runtime.worker_policy import intraop_threads, worker_count


def test_pin_for_dma_noop_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tensor = torch.randn(8, 8)
    assert pin_for_dma(tensor) is tensor


def _specialized(decision: dict[str, object] | None = None) -> SimpleNamespace:
    validation = {} if decision is None else {"concurrency": decision}
    return SimpleNamespace(validation=validation)


def test_worker_policy_preserves_forced_serial_behavior() -> None:
    config = CompileConfig(allow_concurrent_regions=False, max_concurrent_regions=8)
    specialized = _specialized({"enabled": True, "workers": 4, "intraop_threads": 3})

    assert worker_count(specialized, config) == 1
    assert intraop_threads(specialized, config) == 0


def test_worker_policy_uses_measured_concurrency() -> None:
    config = CompileConfig(allow_concurrent_regions=True, max_concurrent_regions=0)
    specialized = _specialized({"enabled": True, "workers": 3, "intraop_threads": 2})

    assert worker_count(specialized, config) == 3
    assert intraop_threads(specialized, config) == 2


def test_resident_parameter_pinning_respects_discovered_pool(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    machine = SimpleNamespace(
        memory={
            "pinned_host": SimpleNamespace(
                memory_class="pinned_host",
                allocatable_bytes=1024,
                capacity_bytes=2048,
            )
        }
    )
    schedule = SimpleNamespace(
        instructions=[
            SimpleNamespace(
                opcode="Transfer",
                attributes={"kind": "parameter_host_to_device"},
                destination="cuda:0",
                resource="cuda:0",
            )
        ]
    )

    assert pinned_host_allocatable_bytes(machine) == 1024
    assert should_pin_parameter_store(schedule, state_bytes=512, machine=machine)
    assert not should_pin_parameter_store(schedule, state_bytes=2048, machine=machine)
    assert should_pin_parameter_store(
        schedule,
        state_bytes=2048,
        machine=machine,
        streaming=True,
    )


def test_provisioning_facade_keeps_existing_import_surface() -> None:
    assert provisioning.worker_count is worker_count
    assert provisioning.intraop_threads is intraop_threads
    assert provisioning.should_pin_parameter_store is should_pin_parameter_store
