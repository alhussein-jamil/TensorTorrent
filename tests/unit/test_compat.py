"""Tests for torch / Python compatibility helpers."""

from __future__ import annotations

import pytest

from tensortorrent._compat import (
    TORCH_MIN,
    parse_torch_version,
    require_torch,
    torch_compat_line,
    torch_meets_minimum,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.4.0", (2, 4)),
        ("2.4", (2, 4)),
        ("2.13.0+cpu", (2, 13)),
        ("2.4.0+cu124", (2, 4)),
        ("2.5.0.dev20240101", (2, 5)),
    ],
)
def test_parse_torch_version(version: str, expected: tuple[int, int]) -> None:
    assert parse_torch_version(version) == expected


@pytest.mark.parametrize(
    ("version", "ok"),
    [
        ("2.4.0", True),
        ("2.4.0+cpu", True),
        ("2.13.0", True),
        ("2.3.1", False),
        ("1.13.0", False),
    ],
)
def test_torch_meets_minimum(version: str, ok: bool) -> None:
    assert torch_meets_minimum(version) is ok


def test_parse_torch_version_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="unparsable"):
        parse_torch_version("not.a.version")


def test_require_torch_and_compat_line() -> None:
    torch = require_torch()
    assert torch_meets_minimum(str(torch.__version__))
    line = torch_compat_line()
    assert str(torch.__version__) in line
    assert f">={TORCH_MIN[0]}.{TORCH_MIN[1]}" in line
