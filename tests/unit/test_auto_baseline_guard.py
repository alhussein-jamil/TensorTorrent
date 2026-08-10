import math

from tensortorrent.compile.bakeoff import prefer_cpu_baseline as _prefer_cpu_baseline


def test_cpu_baseline_wins_when_measured_faster() -> None:
    assert _prefer_cpu_baseline(cpu_s=0.7, streamed_s=1.5)


def test_cpu_baseline_gets_small_hysteresis_for_near_ties() -> None:
    assert _prefer_cpu_baseline(cpu_s=1.01, streamed_s=1.0)
    assert not _prefer_cpu_baseline(cpu_s=1.03, streamed_s=1.0)


def test_surviving_candidate_wins_when_other_measurement_fails() -> None:
    assert _prefer_cpu_baseline(cpu_s=0.7, streamed_s=math.inf)
    assert not _prefer_cpu_baseline(cpu_s=math.inf, streamed_s=0.7)
    assert not _prefer_cpu_baseline(cpu_s=math.inf, streamed_s=math.inf)
