"""Opt-in training UX: schedule autograd, optimizers, train/eval, guards."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import tensortorrent as tt
from tensortorrent.config import CompileConfig
from tensortorrent.errors import UnsupportedFeatureError


def _train_config(**extra: object) -> CompileConfig:
    return CompileConfig(
        allow_training=True,
        use_torch_compile=False,
        measure_regions=False,
        **extra,  # type: ignore[arg-type]
    )


class _Chain(nn.Module):
    """Three linears so ``max_region_nodes=1`` yields multiple schedule regions."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(self.b(self.a(x)))


def test_optimizer_step_updates_weights() -> None:
    model = nn.Linear(4, 2)
    x = torch.randn(8, 4)
    compiled = tt.compile(model, (torch.randn(8, 4),), config=_train_config())
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


def test_train_forward_uses_schedule_with_grad() -> None:
    compiled = tt.compile(_Chain(), (torch.randn(2, 4),), config=_train_config(max_region_nodes=1))
    try:
        assert len(compiled._program.regions) > 1
        assert "schedule with autograd" in compiled.explain()
        calls = {"n": 0}
        original = compiled.executor.run

        def _spy(flat_inputs, *, cancel_token=None, enable_grad=False):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            assert enable_grad is True
            return original(flat_inputs, cancel_token=cancel_token, enable_grad=enable_grad)

        compiled.executor.run = _spy  # type: ignore[method-assign]
        x = torch.randn(2, 4, requires_grad=True)
        out = compiled(x)
        assert calls["n"] == 1
        assert out.requires_grad
        out.sum().backward()
        assert x.grad is not None
        assert any(p.grad is not None for p in compiled.parameters())
    finally:
        compiled.close()


def test_multi_region_schedule_train_backward() -> None:
    compiled = tt.compile(_Chain(), (torch.randn(2, 4),), config=_train_config(max_region_nodes=1))
    try:
        assert len(compiled._program.regions) >= 3
        compute_ops = [i for i in compiled.executor.schedule.instructions if i.opcode.value == "Compute"]
        assert len(compute_ops) >= 3
        x = torch.randn(2, 4, requires_grad=True)
        opt = torch.optim.SGD(compiled.parameters(), lr=0.1)
        opt.zero_grad()
        loss = compiled(x).sum()
        loss.backward()
        opt.step()
        assert x.grad is not None
        assert all(p.grad is not None for p in compiled.parameters())
    finally:
        compiled.close()


def test_eval_forward_disables_schedule_grad() -> None:
    compiled = tt.compile(nn.Linear(4, 2), (torch.randn(2, 4),), config=_train_config())
    try:
        compiled.eval()
        seen: list[bool] = []
        original = compiled.executor.run

        def _spy(flat_inputs, *, cancel_token=None, enable_grad=False):  # type: ignore[no-untyped-def]
            seen.append(bool(enable_grad))
            return original(flat_inputs, cancel_token=cancel_token, enable_grad=enable_grad)

        compiled.executor.run = _spy  # type: ignore[method-assign]
        out = compiled(torch.randn(2, 4))
        assert seen == [False]
        assert out.requires_grad is False
    finally:
        compiled.close()


def test_eval_after_train_uses_updated_weights_on_schedule() -> None:
    model = nn.Linear(4, 2)
    x = torch.randn(8, 4)
    compiled = tt.compile(model, (torch.randn(8, 4),), config=_train_config())
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
    compiled = tt.compile(
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
        tt.compile(
            model,
            (torch.randn(2, 32),),
            config=_train_config(ram_budget_bytes=max(1, total // 4), allow_nvme_streaming=True),
        )


def test_activation_budget_incompatible_with_training() -> None:
    with pytest.raises(UnsupportedFeatureError, match="activation_budget"):
        CompileConfig(allow_training=True, activation_budget_bytes=1024)


def test_fit_runs_schedule_train_loop() -> None:
    compiled = tt.compile(nn.Linear(4, 2), (torch.randn(4, 4),), config=_train_config())
    try:
        before = {n: p.detach().clone() for n, p in compiled.named_parameters()}
        opt = torch.optim.SGD(compiled.parameters(), lr=0.5)
        batches = [(torch.randn(4, 4), torch.randn(4, 2)) for _ in range(3)]
        history = tt.fit(compiled, batches, optimizer=opt, loss_fn=nn.MSELoss(), epochs=2)
        assert len(history) == 2
        assert all(isinstance(v, float) for v in history)
        assert any(not torch.equal(before[n], p.detach()) for n, p in compiled.named_parameters())
    finally:
        compiled.close()


def test_schedule_train_with_torch_compile_regions() -> None:
    compiled = tt.compile(
        _Chain(),
        (torch.randn(2, 4),),
        config=CompileConfig(
            allow_training=True,
            use_torch_compile=True,
            measure_regions=False,
            max_region_nodes=1,
        ),
    )
    try:
        assert len(compiled._program.regions) > 1
        x = torch.randn(2, 4, requires_grad=True)
        out = compiled(x)
        assert out.requires_grad
        out.sum().backward()
        assert x.grad is not None
        assert any(p.grad is not None for p in compiled.parameters())
    finally:
        compiled.close()


def test_grad_device_move_preserves_autograd() -> None:
    from tensortorrent.runtime.grad_transfer import move_for_training

    x = torch.randn(3, 4, requires_grad=True)
    y = move_for_training(x, torch.device("cpu"))
    assert y.requires_grad
    y.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_fit_materializes_iterator_across_epochs() -> None:
    compiled = tt.compile(nn.Linear(4, 2), (torch.randn(4, 4),), config=_train_config())
    try:
        opt = torch.optim.SGD(compiled.parameters(), lr=0.1)

        def _gen():
            yield (torch.randn(4, 4), torch.randn(4, 2))
            yield (torch.randn(4, 4), torch.randn(4, 2))

        history = tt.fit(compiled, _gen(), optimizer=opt, loss_fn=nn.MSELoss(), epochs=2)
        assert len(history) == 2
        assert history[1] != 0.0 or history[0] != 0.0
    finally:
        compiled.close()


def test_fit_supports_multiple_inputs_with_target() -> None:
    class TwoInput(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 2)

        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return self.projection(left + right)

    example = (torch.randn(3, 4), torch.randn(3, 4))
    compiled = tt.compile(TwoInput(), example, config=_train_config())
    try:
        optimizer = torch.optim.SGD(compiled.parameters(), lr=0.1)
        history = tt.fit(
            compiled,
            [((torch.randn(3, 4), torch.randn(3, 4)), torch.randn(3, 2))],
            optimizer=optimizer,
            loss_fn=nn.MSELoss(),
        )
        assert len(history) == 1
    finally:
        compiled.close()


def test_fit_rejects_non_positive_epochs() -> None:
    compiled = tt.compile(nn.Linear(4, 2), (torch.randn(2, 4),), config=_train_config())
    try:
        with pytest.raises(ValueError, match="epochs"):
            tt.fit(
                compiled,
                [(torch.randn(2, 4),)],
                optimizer=torch.optim.SGD(compiled.parameters(), lr=0.1),
                loss_fn=lambda pred: pred.sum(),
                epochs=0,
            )
    finally:
        compiled.close()


@pytest.mark.parametrize("epochs", (True, 1.5))
def test_fit_rejects_non_integer_epochs(epochs: object) -> None:
    compiled = tt.compile(nn.Linear(4, 2), (torch.randn(2, 4),), config=_train_config())
    try:
        with pytest.raises(TypeError, match="epochs must be an integer"):
            tt.fit(
                compiled,
                [(torch.randn(2, 4),)],
                optimizer=torch.optim.SGD(compiled.parameters(), lr=0.1),
                loss_fn=lambda pred: pred.sum(),
                epochs=epochs,  # type: ignore[arg-type]
            )
    finally:
        compiled.close()


def test_fit_requires_allow_training() -> None:
    compiled = tt.compile(
        nn.Linear(4, 2).eval(),
        (torch.randn(2, 4),),
        config=CompileConfig(use_torch_compile=False, measure_regions=False, prefer_direct_path=False),
    )
    try:
        with pytest.raises(UnsupportedFeatureError, match="allow_training"):
            tt.fit(
                compiled,
                [(torch.randn(2, 4),)],
                optimizer=torch.optim.SGD(compiled.parameters(), lr=0.1),
                loss_fn=lambda pred: pred.sum(),
            )
    finally:
        compiled.close()


def test_fit_rejects_empty_batches() -> None:
    compiled = tt.compile(nn.Linear(4, 2), (torch.randn(2, 4),), config=_train_config())
    try:
        with pytest.raises(ValueError, match="zero batches"):
            tt.fit(
                compiled,
                [],
                optimizer=torch.optim.SGD(compiled.parameters(), lr=0.1),
                loss_fn=lambda pred: pred.sum(),
            )
    finally:
        compiled.close()


def test_fit_rejects_non_scalar_loss() -> None:
    compiled = tt.compile(nn.Linear(4, 2), (torch.randn(2, 4),), config=_train_config())
    try:
        with pytest.raises(ValueError, match="scalar"):
            tt.fit(
                compiled,
                [torch.randn(2, 4)],
                optimizer=torch.optim.SGD(compiled.parameters(), lr=0.1),
                loss_fn=lambda pred: pred,  # shape (2, 2)
            )
    finally:
        compiled.close()


def test_fit_rejects_closed_module() -> None:
    compiled = tt.compile(nn.Linear(4, 2), (torch.randn(2, 4),), config=_train_config())
    opt = torch.optim.SGD(compiled.parameters(), lr=0.1)
    compiled.close()
    with pytest.raises(Exception, match="closed"):
        tt.fit(compiled, [torch.randn(2, 4)], optimizer=opt, loss_fn=lambda pred: pred.sum())


def test_forward_rejects_enable_grad_kwarg() -> None:
    compiled = tt.compile(nn.Linear(4, 2), (torch.randn(2, 4),), config=_train_config())
    try:
        with pytest.raises(TypeError, match="enable_grad"):
            compiled(torch.randn(2, 4), enable_grad=True)  # type: ignore[call-arg]
    finally:
        compiled.close()


def test_train_does_not_feed_profile_feedback() -> None:
    compiled = tt.compile(
        nn.Linear(4, 2),
        (torch.randn(2, 4),),
        config=_train_config(online_profile_feedback=True),
    )
    try:
        calls = {"n": 0}
        original = compiled._profile_feedback.observe_report

        def _spy(report):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return original(report)

        compiled._profile_feedback.observe_report = _spy  # type: ignore[method-assign]
        compiled.train()
        compiled(torch.randn(2, 4)).sum().backward()
        assert calls["n"] == 0
        compiled.eval()
        compiled(torch.randn(2, 4))
        assert calls["n"] == 1
    finally:
        compiled.close()


def test_enable_grad_lives_on_execution_context() -> None:
    """Train flag is per-run context so Rust worker-thread callbacks see it."""
    from tensortorrent.runtime.execution_context import ExecutionContext

    train_ctx = ExecutionContext(enable_grad=True)
    infer_ctx = ExecutionContext(enable_grad=False)
    assert train_ctx.enable_grad is True
    assert infer_ctx.enable_grad is False

    compiled = tt.compile(
        nn.Linear(4, 2),
        (torch.randn(2, 4),),
        config=_train_config(prefer_direct_path=False, allow_gpu=False),
    )
    try:
        seen: list[bool] = []
        se = compiled.executor._schedule_executor
        assert se is not None
        original = se._exec_compute

        def _spy(inst, ctx, submitted):  # type: ignore[no-untyped-def]
            seen.append(bool(ctx.enable_grad))
            return original(inst, ctx, submitted)

        se._exec_compute = _spy  # type: ignore[method-assign]
        compiled.train()
        compiled(torch.randn(2, 4)).sum().backward()
        compiled.eval()
        compiled(torch.randn(2, 4))
        assert True in seen
        assert False in seen
    finally:
        compiled.close()


def test_default_compile_stays_inference() -> None:
    compiled = tt.compile(
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
    compiled = tt.compile(nn.Linear(4, 2), (torch.randn(2, 4),), config=_train_config())
    try:
        assert "schedule with autograd" in compiled.explain()
        compiled.eval()
        assert "inference schedule" in compiled.explain()
        compiled.train()
        assert "schedule with autograd" in compiled.explain()
    finally:
        compiled.close()


def test_hetero_mock_schedule_train_backward() -> None:
    """CPU + mock-accel Transfers must keep autograd through the schedule."""
    from tests.support.helpers import cpu_config, cpu_host_graph

    from tensortorrent.backends.mock_accel import make_mock_accel_graph
    from tensortorrent.compile.measure import MeasurementSet, RegionMeasurement
    from tensortorrent.config import Objective
    from tensortorrent.ir.graph import OpCode
    from tensortorrent.ir.resource_graph import merge_graphs

    class Parallel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Linear(8, 8)
            self.right = nn.Linear(8, 8)
            self.mix = nn.Linear(16, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mix(torch.cat([torch.relu(self.left(x)), torch.relu(self.right(x))], dim=-1))

    machine = merge_graphs(cpu_host_graph(), make_mock_accel_graph(delay_hint_s=0.05))
    cpu = next(n for n, c in machine.compute.items() if c.backend_id == "cpu")
    accel = "mock_accel_0"
    cfg = cpu_config(
        allow_training=True,
        allow_concurrent_regions=True,
        max_concurrent_regions=2,
        max_region_nodes=8,
        objective=Objective.LATENCY,
        use_torch_compile=False,
        measure_regions=False,
    )
    x0 = torch.randn(2, 8)
    probe = tt.compile(Parallel(), (x0,), config=cfg, machine=machine)
    try:
        region_ids = [r.region_id for r in probe._program.regions]
        assert len(region_ids) >= 2
        ms = MeasurementSet()
        for i, rid in enumerate(region_ids):
            if i % 2 == 0:
                ms.add(RegionMeasurement(rid, cpu, "cpu", 0.001, True))
                ms.add(RegionMeasurement(rid, accel, "mock_accel", 1.0, True))
            else:
                ms.add(RegionMeasurement(rid, cpu, "cpu", 1.0, True))
                ms.add(RegionMeasurement(rid, accel, "mock_accel", 0.001, True))
    finally:
        probe.close()

    compiled = tt.compile(Parallel(), (x0,), config=cfg, machine=machine, measurements=ms)
    try:
        devices = {p.device for p in compiled.specialized.plan.placements}
        assert cpu in devices and accel in devices
        assert any(i.opcode == OpCode.TRANSFER for i in compiled.executor.schedule.instructions)
        x = torch.randn(2, 8, requires_grad=True)
        out = compiled(x)
        assert out.requires_grad
        out.sum().backward()
        assert x.grad is not None
        assert any(p.grad is not None for p in compiled.parameters())

        opt = torch.optim.SGD(compiled.parameters(), lr=0.05)
        opt.zero_grad()
        compiled(x.detach()).sum().backward()
        opt.step()
        train_out = compiled(x.detach()).detach().clone()
        compiled.eval()
        eval_out = compiled(x.detach())
        assert eval_out.requires_grad is False
        torch.testing.assert_close(eval_out, train_out, atol=1e-4, rtol=1e-4)
    finally:
        compiled.close()
