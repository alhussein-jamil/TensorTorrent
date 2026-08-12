"""exported.pt2 must materialize onto map_location (default CPU), not archive device."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch


def test_load_exported_program_uses_force_device(tmp_path: Path):
    from tensortorrent.frontend.export import load_exported_program

    sentinel = object()
    path = tmp_path / "exported.pt2"
    path.write_bytes(b"x")

    with (
        patch("torch.export.load", return_value=sentinel) as load_mock,
        patch("tensortorrent.frontend.export._force_pt2_load_device") as force_mock,
    ):
        force_mock.return_value.__enter__ = lambda self: None
        force_mock.return_value.__exit__ = lambda *a: None
        out = load_exported_program(path, map_location="cpu")

    assert out is sentinel
    force_mock.assert_called_once_with("cpu")
    load_mock.assert_called_once_with(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_load_exported_program_cpu_despite_cuda_capture(tmp_path: Path):
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(2, 2, device="cuda"))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.w

    model = Tiny().eval()
    x = torch.randn(2, 2, device="cuda")
    ep = torch.export.export(model, (x,))
    path = tmp_path / "exported.pt2"
    torch.export.save(ep, path)

    from tensortorrent.frontend.export import load_exported_program

    loaded = load_exported_program(path, map_location="cpu")
    assert loaded.state_dict
    for tensor in loaded.state_dict.values():
        if isinstance(tensor, torch.Tensor):
            assert tensor.device.type == "cpu", tensor.device
