"""Opt-in training UX: backward, optimizers, train/eval, incompatibility guards."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.config import CompileConfig
from streamcompiler.errors import UnsupportedFeatureError


def _train_config(**extra: object) -> CompileConfig:
    return CompileConfig(
        allow_training=True,
        use_torch_compile=False,
        measure_regions=False,
        **extra,  # type: ignore[arg-type]
    )


def test_optimizer_step_updates_weights() -> None:
    model = nn.Linear(4, 2)
    x = torch.randn(8, 4)
    compiled = sc.compile(model, (torch.randn(8, 4),), config=_train_config())
    try:
        assert compiled.training is True
        before = {name: p.detach().clone() for name, p in compiled.named_parameters()}
        opt = torch.optim.SGD(compiled.parameters(), lr=0.5)
        out_before = compiled(x).detach().clone()
        opt.zero_grad()
        loss = compiled(x).sum()
        loss.backward()
        opt.step()
        assert any(not torch.equal(before[n], p.detach()) for n, p in compiled.named_parameters())
        out_after = compiled(x)
        assert not torch.allclose(out_before, out_after)
    finally:
        compiled.close()


def test_eval_after_train_uses_updated_weights_on_schedule() -> None:
    model = nn.Linear(4, 2)
    x = torch.randn(8, 4)
    compiled = sc.compile(model, (torch.randn(8, 4),), config=_train_config())
    try:
        compiled.train()
        opt = torch.optim.SGD(compiled.parameters(), lr=1.0)
        opt.zero_grad()
        compiled(x).sum().backward()
        opt.step()
        train_out = compiled(x).detach().clone()

        compiled.eval()
        assert compiled.training is False
        assert "inference schedule" in compiled.explain()
        eval_out = compiled(x)
        assert eval_out.requires_grad is False
        torch.testing.assert_close(eval_out, train_out, atol=1e-5, rtol=1e-5)
    finally:
        compiled.close()


def test_train_without_allow_training_raises() -> None:
    compiled = sc.compile(
        nn.Linear(4, 2).eval(),
        (torch.randn(2, 4),),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        assert compiled.config.allow_training is False
        assert compiled.training is False
        with pytest.raises(UnsupportedFeatureError, match="allow_training=True"):
            compiled.train()
        compiled.eval()
        assert compiled.training is False
    finally:
        compiled.close()


def test_process_workers_incompatible_with_training() -> None:
    with pytest.raises(UnsupportedFeatureError, match="process_workers"):
        CompileConfig(allow_training=True, process_workers=2)


def test_streaming_incompatible_with_training() -> None:
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8))
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    with pytest.raises(UnsupportedFeatureError, match="parameter streaming"):
        sc.compile(
            model,
            (torch.randn(2, 32),),
            config=_train_config(ram_budget_bytes=max(1, total // 4), allow_nvme_streaming=True),
        )


def test_default_compile_stays_inference() -> None:
    compiled = sc.compile(
        nn.Linear(4, 2).eval(),
        (torch.randn(2, 4),),
        config=CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        x = torch.randn(2, 4, requires_grad=True)
        out = compiled(x)
        assert out.requires_grad is False
        assert "inference schedule" in compiled.explain()
    finally:
        compiled.close()


def test_training_explain_notes_mode() -> None:
    compiled = sc.compile(nn.Linear(4, 2), (torch.randn(2, 4),), config=_train_config())
    try:
        assert "live graph_module" in compiled.explain()
        compiled.eval()
        assert "inference schedule" in compiled.explain()
        compiled.train()
        assert "live graph_module" in compiled.explain()
    finally:
        compiled.close()
