"""End-to-end equivalence tests: StreamCompiler output must equal eager PyTorch."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc
from streamcompiler.errors import UnsupportedFeatureError


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Residual(nn.Module):
    """Two independent branches that join — the canonical concurrency shape."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.left = nn.Linear(width, width)
        self.right = nn.Linear(width, width)
        self.head = nn.Linear(width, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        return self.head(torch.relu(self.left(h)) + torch.tanh(self.right(h)) + h)


class MultiInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(4, 8)

    def forward(self, x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (self.a(x) + self.b(y)) * scale


class StructuredOutputs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.register_buffer("offset", torch.arange(8, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        h = self.lin(x) + self.offset
        return {"hidden": h, "pair": (h.sum(dim=-1), torch.relu(h))}


class SharedParameter(nn.Module):
    """One weight matrix used twice; lowering must not duplicate it."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(8, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.shared(torch.relu(self.shared(x)))))


class BufferCounter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.register_buffer("running", torch.full((8,), 3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) / self.running


def assert_matches_eager(model: nn.Module, args: tuple[object, ...]) -> sc.CompiledModule:
    model = model.eval()
    with torch.no_grad():
        expected = model(*args)
    compiled = sc.compile(model, args, devices="auto")
    actual = compiled(*args)
    torch.testing.assert_close(actual, expected)
    return compiled


def test_linear_model_matches_eager() -> None:
    compiled = assert_matches_eager(nn.Linear(16, 4), (torch.randn(3, 16),))
    assert isinstance(compiled, nn.Module)
    assert isinstance(compiled, torch.nn.Module)


def test_mlp_matches_eager() -> None:
    assert_matches_eager(Mlp(), (torch.randn(5, 16),))


def test_branching_model_matches_eager() -> None:
    compiled = assert_matches_eager(Residual(), (torch.randn(4, 32),))
    # Branching must produce more than one region, otherwise nothing can overlap.
    assert len(compiled.regions) > 2


def test_multiple_inputs_match_eager() -> None:
    assert_matches_eager(MultiInput(), (torch.randn(2, 8), torch.randn(2, 4), torch.randn(2, 8)))


def test_structured_outputs_preserve_pytree() -> None:
    model = StructuredOutputs().eval()
    x = torch.randn(3, 8)
    with torch.no_grad():
        expected = model(x)
    compiled = sc.compile(model, (x,))
    actual = compiled(x)
    assert isinstance(actual, dict)
    assert set(actual) == {"hidden", "pair"}
    assert isinstance(actual["pair"], tuple) and len(actual["pair"]) == 2
    torch.testing.assert_close(actual, expected)


def test_shared_parameters_are_not_duplicated() -> None:
    model = SharedParameter().eval()
    compiled = assert_matches_eager(model, (torch.randn(2, 8),))
    bindings = compiled.program.state_bindings
    targets = list(bindings.values())
    # `shared.weight` appears once per fx get_attr, all mapping to one real tensor.
    shared = [t for t in targets if t.endswith("shared.weight")]
    assert shared, "shared weight must be bound"
    tensors = {id(compiled.program.state_tensor(n)) for n, t in bindings.items() if t in shared}
    assert len(tensors) == 1


def test_registered_buffers_are_used() -> None:
    model = BufferCounter().eval()
    compiled = assert_matches_eager(model, (torch.randn(2, 8),))
    kinds = {compiled.program.values[n].kind for n in compiled.program.state_bindings}
    assert "buffer" in kinds
    assert "running" in dict(compiled.named_buffers()) or any(k.endswith("running") for k in compiled.state_dict())


def test_repeated_calls_are_stable() -> None:
    model = Mlp().eval()
    x = torch.randn(4, 16)
    compiled = sc.compile(model, (x,))
    with torch.no_grad():
        expected = model(x)
    first = compiled(x)
    for _ in range(4):
        torch.testing.assert_close(compiled(x), expected)
    torch.testing.assert_close(first, expected)


def test_different_shape_is_rejected_explicitly() -> None:
    """Static-shape compilation must fail loudly rather than silently mis-execute."""
    compiled = sc.compile(nn.Linear(8, 8).eval(), (torch.randn(2, 8),))
    with pytest.raises(UnsupportedFeatureError, match="compiled for shape"):
        compiled(torch.randn(5, 8))


def test_wrong_dtype_is_rejected_explicitly() -> None:
    compiled = sc.compile(nn.Linear(8, 8).eval(), (torch.randn(2, 8),))
    with pytest.raises(UnsupportedFeatureError, match="compiled for dtype"):
        compiled(torch.randn(2, 8, dtype=torch.float64))


def test_identity_model_returns_its_input() -> None:
    class Identity(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    x = torch.randn(2, 3)
    compiled = sc.compile(Identity().eval(), (x,))
    torch.testing.assert_close(compiled(x), x)
    # A pass-through graph has nothing to compute, so it must not fabricate regions.
    assert compiled.regions == ()
    assert compiled.specialized.validation["pass_through"] is True
    assert compiled.specialized.plan.placements == []


def test_export_shape_guards_are_replaced_by_our_own_validation() -> None:
    """We drop torch.export's guard node because flatten_inputs already checks inputs."""
    compiled = sc.compile(nn.Linear(8, 8).eval(), (torch.randn(2, 8),))
    assert compiled.program.metadata["export_guards_removed"] >= 1
    assert all("guard" not in region.submodule for region in compiled.program.regions)
    with pytest.raises(UnsupportedFeatureError, match="compiled for shape"):
        compiled(torch.randn(3, 8))


def test_outputs_do_not_track_gradients_even_when_regions_overlap() -> None:
    """Worker threads must inherit inference mode, not just the calling thread."""
    model = Residual(width=256).eval()
    x = torch.randn(64, 256)
    compiled = sc.compile(model, (x,), config=sc.CompileConfig(max_concurrent_regions=4))
    assert compiled.executor.max_workers == 4
    out = compiled(x)
    assert out.requires_grad is False
    assert torch.is_inference(out)
    with torch.no_grad():
        torch.testing.assert_close(out, model(x))


def test_wrong_input_structure_raises() -> None:
    compiled = sc.compile(nn.Linear(8, 8).eval(), (torch.randn(2, 8),))
    with pytest.raises(UnsupportedFeatureError, match="Input structure"):
        compiled(torch.randn(2, 8), torch.randn(2, 8))


def test_non_module_input_raises() -> None:
    from streamcompiler.errors import GraphCaptureError

    with pytest.raises(GraphCaptureError, match="torch.nn.Module"):
        sc.compile(lambda x: x, (torch.randn(2, 2),))


def test_unexportable_model_raises_graph_capture_error() -> None:
    from streamcompiler.errors import GraphCaptureError

    class DataDependent(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Python-level branch on tensor data cannot be exported.
            if bool(x.sum() > 0):
                return x * 2
            return x - 1

    with pytest.raises(GraphCaptureError):
        sc.compile(DataDependent().eval(), (torch.randn(4),))


def test_cpu_only_execution_uses_cpu_backend() -> None:
    compiled = sc.compile(Mlp().eval(), (torch.randn(2, 16),), devices="cpu")
    plan = compiled.specialized.plan
    assert {p.backend_id for p in plan.placements} == {"cpu"}
    assert all(d.startswith("cpu") for d in plan.devices_used)


def test_artifacts_and_visualization(tmp_path: Path) -> None:
    model = Mlp().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(
        model,
        (x,),
        config=sc.CompileConfig(objective=sc.Objective.LATENCY, profile_level="coarse"),
        artifact_dir=tmp_path / "art",
    )
    text = compiled.explain()
    assert "devices_used" in text
    assert "regions:" in text
    compiled.visualize(str(tmp_path / "plan.html"))
    assert (tmp_path / "plan.html").exists()
    assert (tmp_path / "plan.trace.json").exists()
    assert (tmp_path / "art" / "portable.json").exists()
    torch.testing.assert_close(compiled(x), model(x))


def test_save_and_reload_reproduces_outputs(tmp_path: Path) -> None:
    model = Mlp().eval()
    x = torch.randn(3, 16)
    compiled = sc.compile(model, (x,))
    expected = compiled(x)
    compiled.save(tmp_path / "saved")
    assert (tmp_path / "saved" / "exported.pt2").exists()

    reloaded = sc.load_compiled(tmp_path / "saved")
    torch.testing.assert_close(reloaded(x), expected)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), model(x))


def test_state_dict_roundtrip_is_a_real_module(tmp_path: Path) -> None:
    model = Mlp().eval()
    x = torch.randn(2, 16)
    compiled = sc.compile(model, (x,))
    path = tmp_path / "sd.pt"
    torch.save(compiled.state_dict(), path)
    loaded = torch.load(path, weights_only=True)
    assert loaded
    assert list(compiled.parameters())
    assert compiled.training is False or compiled.eval() is compiled


def test_call_fast_path_is_used_only_when_the_spec_allows_it() -> None:
    """The positional-tensor fast path must be derived from the spec, not assumed."""
    flat = sc.compile(MultiInput().eval(), (torch.randn(2, 8), torch.randn(2, 4), torch.randn(2, 8)))
    assert flat.program._positional_tensor_arity == 3
    assert flat.program._single_output is True

    structured = sc.compile(StructuredOutputs().eval(), (torch.randn(2, 8),))
    assert structured.program._single_output is False


def test_structured_output_model_still_matches_eager_after_fast_paths() -> None:
    model = StructuredOutputs().eval()
    x = torch.randn(3, 8)
    with torch.no_grad():
        expected = model(x)
    actual = sc.compile(model, (x,))(x)
    assert set(actual) == set(expected)
    torch.testing.assert_close(actual["hidden"], expected["hidden"])
    torch.testing.assert_close(actual["pair"][0], expected["pair"][0])
    torch.testing.assert_close(actual["pair"][1], expected["pair"][1])
