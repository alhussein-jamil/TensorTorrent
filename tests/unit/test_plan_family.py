"""Tests for dynamic-shape plan family selection."""

from __future__ import annotations

import pytest

from streamcompiler.planner.plan_family import PlanFamily, select_bucket


def test_select_bucket_prefers_tightest_match() -> None:
    b = select_bucket(1, 128)
    assert b is not None
    assert b.name == "decode_b1_s512"


def test_plan_family_fallback() -> None:
    family = PlanFamily(fingerprint="x", plans={"decode_b1_s512": "p1"}, fallback="decode_b1_s512")
    assert family.choose(1, 100) == "p1"
    # Unseen large shape uses fallback rather than silently applying an invalid plan key.
    assert family.choose(16, 100000) == "p1"


def test_plan_family_errors_without_fallback() -> None:
    family = PlanFamily(fingerprint="x", plans={})
    with pytest.raises(KeyError):
        family.choose(1, 128)
