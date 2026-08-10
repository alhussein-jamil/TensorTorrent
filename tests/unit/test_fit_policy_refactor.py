from tensortorrent.compile.fit import (
    accelerator_hoist_budget_bytes,
    accelerator_region_state_budget_bytes,
    exceeds_accelerator_region_budget,
    region_state_budget,
    select_persistent_parameter_ids,
    should_hoist_resident_parameters,
)
from tensortorrent.config import CompileConfig


def test_accelerator_budget_is_single_source_for_residency() -> None:
    config = CompileConfig(vram_budget_bytes=1_000, ram_budget_bytes=None)

    assert accelerator_region_state_budget_bytes(config, None) == 700
    # Region partition budget stays 0.70×; hoist uses usable − safety.
    hoist_limit = int(accelerator_hoist_budget_bytes(config, None) or 0)
    assert hoist_limit == 875  # 1000 − min(safety floor, capacity/8)
    assert should_hoist_resident_parameters(config, state_bytes=hoist_limit)
    assert not should_hoist_resident_parameters(config, state_bytes=hoist_limit + 1)
    assert not exceeds_accelerator_region_budget(config, None, parameter_bytes=700)
    assert exceeds_accelerator_region_budget(config, None, parameter_bytes=701)


def test_accelerator_only_check_does_not_inherit_ram_streaming_budget() -> None:
    config = CompileConfig(vram_budget_bytes=1_000, ram_budget_bytes=100, prefetch_distance=1)

    assert accelerator_region_state_budget_bytes(config, None) == 700
    assert region_state_budget(config, None, parameter_bytes=800) == 50
    assert not exceeds_accelerator_region_budget(config, None, parameter_bytes=500)


def test_partial_selection_leaves_stream_headroom_for_transfer_groups() -> None:
    sizes = {f"w{i}": 100 for i in range(10)}
    # Two region transfers of 500 each; budget 800 → naive fill takes 800, but
    # must leave enough free so a streamed region's remainder still fits.
    groups = [tuple(f"w{i}" for i in range(5)), tuple(f"w{i}" for i in range(5, 10))]
    selected = select_persistent_parameter_ids(sizes, budget_bytes=800, transfer_groups=groups)
    resident = sum(sizes[n] for n in selected)
    room = 800 - resident
    assert room >= 0
    for group in groups:
        rem = sum(sizes[n] for n in group if n not in selected)
        assert rem <= room
    # Must not keep the naive full 800B fill.
    assert resident < 800
    assert selected
