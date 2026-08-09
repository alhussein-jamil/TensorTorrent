from tensortorrent.compile.fit import (
    accelerator_region_state_budget_bytes,
    exceeds_accelerator_region_budget,
    region_state_budget,
    should_hoist_resident_parameters,
)
from tensortorrent.config import CompileConfig


def test_accelerator_budget_is_single_source_for_residency() -> None:
    config = CompileConfig(vram_budget_bytes=1_000, ram_budget_bytes=None)

    assert accelerator_region_state_budget_bytes(config, None) == 700
    assert should_hoist_resident_parameters(config, state_bytes=700)
    assert not should_hoist_resident_parameters(config, state_bytes=701)
    assert not exceeds_accelerator_region_budget(config, None, parameter_bytes=700)
    assert exceeds_accelerator_region_budget(config, None, parameter_bytes=701)


def test_accelerator_only_check_does_not_inherit_ram_streaming_budget() -> None:
    config = CompileConfig(vram_budget_bytes=1_000, ram_budget_bytes=100, prefetch_distance=1)

    assert accelerator_region_state_budget_bytes(config, None) == 700
    assert region_state_budget(config, None, parameter_bytes=800) == 50
    assert not exceeds_accelerator_region_budget(config, None, parameter_bytes=500)
