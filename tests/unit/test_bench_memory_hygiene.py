"""Regression tests for benchmark host-RAM hygiene."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from benchmarks.harness import TimedRun, to_plain, write_suite_json
from benchmarks.memory_hygiene import (
    SMOKE_CROSSOVER_MULTIPLES,
    SMOKE_PUBLIC_SUITES,
    abort_if_host_tight,
    crossover_multiples,
    deepmlp_weight_file,
    public_suite_names,
)


def test_smoke_public_suites_exclude_heavy() -> None:
    names = set(public_suite_names(smoke=True))
    assert names == set(SMOKE_PUBLIC_SUITES)
    assert "deepmlp" not in names
    assert "crossover" not in names
    assert "transformer" not in names
    full = set(public_suite_names(smoke=False))
    assert {"deepmlp", "crossover", "transformer"} <= full


def test_smoke_crossover_multiples_stay_tiny() -> None:
    assert crossover_multiples(smoke=True, full=True) == SMOKE_CROSSOVER_MULTIPLES
    assert max(SMOKE_CROSSOVER_MULTIPLES) < 0.5


def test_abort_if_host_tight_when_avail_low() -> None:
    with mock.patch("benchmarks.memory_hygiene.host_available_bytes", return_value=1 * (1024**3)):
        run = abort_if_host_tight(4 * (1024**3), label="unit")
    assert run is not None
    assert run.ok is False
    assert "skip unit" in run.note


def test_abort_if_host_tight_passes_when_avail_ok() -> None:
    with mock.patch("benchmarks.memory_hygiene.host_available_bytes", return_value=40 * (1024**3)):
        run = abort_if_host_tight(2 * (1024**3), label="unit")
    assert run is None


def test_deepmlp_weight_file_cleans_up() -> None:
    with deepmlp_weight_file(32, 2) as (path, pbytes):
        assert pbytes > 0
        assert Path(path).is_file()
        saved = path
    assert not Path(saved).exists()


def test_to_plain_and_write_suite_json(tmp_path: Path) -> None:
    payload = {"approaches": {"tensortorrent": TimedRun(ok=True, median_ms=1.5)}}
    write_suite_json(tmp_path, payload, "a.json", "b.json")
    assert (tmp_path / "a.json").is_file()
    assert (tmp_path / "b.json").is_file()
    plain = to_plain(payload)
    assert plain["approaches"]["tensortorrent"]["median_ms"] == 1.5
