"""Static direct-path parameter cache behavior."""

from __future__ import annotations

import torch

from tensortorrent.runtime.direct_path import DirectParameter


def test_direct_parameter_refreshes_after_source_mutation() -> None:
    source = torch.tensor([1.0, 2.0])
    parameter = DirectParameter(
        source=source,
        value=source.clone(),
        torch_device=torch.device("cpu"),
        source_version=source._version,
    )
    cached = parameter.resolve()
    with torch.no_grad():
        source.add_(3.0)
    refreshed = parameter.resolve()
    assert refreshed is not cached
    torch.testing.assert_close(refreshed, source)


def test_direct_parameter_reuses_unchanged_copy() -> None:
    source = torch.tensor([1.0])
    placed = source.clone()
    parameter = DirectParameter(
        source=source,
        value=placed,
        torch_device=torch.device("cpu"),
        source_version=source._version,
    )
    assert parameter.resolve() is placed
