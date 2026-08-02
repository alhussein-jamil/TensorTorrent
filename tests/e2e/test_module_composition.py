"""Multiple modules compile into one graph, artifact, and runtime schedule."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import streamcompiler as sc


class Split(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"positive": torch.relu(x), "negative": torch.relu(-x)}


class Join(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
        return (left + right) * scale


class Stack(nn.Module):
    def forward(self, tensors: list[torch.Tensor], *, dim: int) -> torch.Tensor:
        return torch.stack(tensors, dim=dim)


def test_compile_modules_series_is_one_program() -> None:
    modules = (nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 3))
    eager = nn.Sequential(*modules).eval()
    x = torch.randn(4, 8)

    compiled = sc.compile_modules(modules, (x,), names=("encoder", "activation", "head"), devices="cpu")

    torch.testing.assert_close(compiled(x), eager(x))
    assert compiled.program.graph_name == "ModuleGraph"
    assert compiled.specialized.schedule is not None
    assert compiled.specialized.schedule.graph_name == compiled.program.graph_name
    assert any(target.endswith("encoder.weight") for target in compiled.program.state_bindings.values())
    assert any(target.endswith("head.weight") for target in compiled.program.state_bindings.values())


def test_composed_artifact_roundtrip(tmp_path: Path) -> None:
    x = torch.randn(2, 4)
    artifact = tmp_path / "composed"
    compiled = sc.compile_modules(
        (nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)),
        (x,),
        artifact_dir=artifact,
        devices="cpu",
        config=sc.CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    expected = compiled(x)
    compiled.close()

    reloaded = sc.load_compiled(artifact)
    try:
        torch.testing.assert_close(reloaded(x), expected)
        assert reloaded.specialized.schedule is not None
        assert reloaded.program.graph_name == "ModuleGraph"
    finally:
        reloaded.close()


def test_artifact_load_requires_integrity_manifest_by_default(tmp_path: Path) -> None:
    x = torch.randn(2, 4)
    artifact = tmp_path / "unsigned"
    compiled = sc.compile_modules(
        (nn.Linear(4, 2),),
        (x,),
        artifact_dir=artifact,
        devices="cpu",
        config=sc.CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    expected = compiled(x)
    compiled.close()
    (artifact / "artifact-integrity.json").unlink()

    with pytest.raises(sc.StreamCompilerError, match="integrity manifest missing"):
        sc.load_compiled(artifact)
    legacy = sc.load_compiled(artifact, verify_integrity=False)
    try:
        torch.testing.assert_close(legacy(x), expected)
    finally:
        legacy.close()


def test_artifact_load_requires_explicit_boolean_security_switches(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="verify_integrity"):
        sc.load_compiled(tmp_path, verify_integrity=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="refresh_artifacts"):
        sc.load_compiled(tmp_path, refresh_artifacts=1)  # type: ignore[arg-type]


def test_module_graph_supports_branches_nested_outputs_and_kwargs() -> None:
    split = Split()
    left = nn.Linear(6, 6, bias=False)
    right = nn.Linear(6, 6, bias=False)
    join = Join()
    graph = sc.ModuleGraph(
        (
            sc.ModuleNode("split", split, (sc.GraphInput(0),)),
            sc.ModuleNode("left", left, (sc.NodeOutput("split", ("positive",)),)),
            sc.ModuleNode("right", right, (sc.NodeOutput("split", ("negative",)),)),
            sc.ModuleNode(
                "join",
                join,
                (sc.NodeOutput("left"), sc.NodeOutput("right")),
                {"scale": sc.GraphInput(1)},
            ),
        ),
        outputs=(sc.NodeOutput("join"), sc.NodeOutput("split", ("positive",)), sc.GraphInput(0)),
    ).eval()
    x = torch.randn(3, 6)
    scale = torch.full((3, 6), 0.5)

    expected = graph(x, scale)
    compiled = sc.compile(graph, (x, scale), devices="cpu", config=sc.CompileConfig(max_concurrent_regions=2))
    actual = compiled(x, scale)

    assert isinstance(actual, tuple) and len(actual) == 3
    torch.testing.assert_close(actual, expected)
    assert len(compiled.program.regions) >= 2
    output_names = {str(ref) for kind, ref in compiled.program.output_refs if kind == "value"}
    released = {
        inst.inputs[0]
        for inst in compiled.specialized.schedule.instructions
        if inst.opcode.value == "Release" and inst.inputs
    }
    assert output_names.isdisjoint(released)


def test_module_graph_supports_structured_arguments_and_scalar_constants() -> None:
    graph = sc.ModuleGraph(
        (
            sc.ModuleNode("left", nn.Linear(4, 2), (sc.GraphInput(0),)),
            sc.ModuleNode("right", nn.Linear(4, 2), (sc.GraphInput(0),)),
            sc.ModuleNode(
                "stack",
                Stack(),
                ([sc.NodeOutput("left"), sc.NodeOutput("right")],),
                {"dim": 1},
            ),
        )
    ).eval()
    x = torch.randn(3, 4)
    expected = graph(x)
    compiled = sc.compile(
        graph,
        (x,),
        devices="cpu",
        config=sc.CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        torch.testing.assert_close(compiled(x), expected)
        assert compiled(x).shape == (3, 2, 2)
    finally:
        compiled.close()


def test_module_graph_snapshots_mutable_argument_definitions() -> None:
    tensor_list = [sc.GraphInput(0)]
    node = sc.ModuleNode("stack", Stack(), (tensor_list,), {"dim": 0})
    graph = sc.ModuleGraph((node,))

    tensor_list.append(sc.GraphInput(0))
    assert isinstance(node.inputs[0], list)
    node.inputs[0].append(sc.GraphInput(0))

    x = torch.randn(2, 3)
    actual = graph(x)
    assert actual.shape == (1, 2, 3)
    torch.testing.assert_close(actual, torch.stack([x], dim=0))


def test_module_graph_snapshots_mutable_output_definitions() -> None:
    outputs = [sc.NodeOutput("identity")]
    graph = sc.ModuleGraph((sc.ModuleNode("identity", nn.Identity(), (sc.GraphInput(0),)),), outputs=outputs)
    outputs.append(sc.GraphInput(0))

    x = torch.randn(2, 3)
    actual = graph(x)
    assert isinstance(actual, list) and len(actual) == 1
    torch.testing.assert_close(actual[0], x)


def test_module_graph_preserves_structured_graph_outputs() -> None:
    graph = sc.ModuleGraph(
        (
            sc.ModuleNode("split", Split(), (sc.GraphInput(0),)),
            sc.ModuleNode("project", nn.Linear(4, 2), (sc.NodeOutput("split", ("positive",)),)),
        ),
        outputs={
            "features": sc.NodeOutput("project"),
            "debug": [sc.NodeOutput("split", ("negative",)), sc.GraphInput(0)],
        },
    ).eval()
    x = torch.randn(3, 4)
    expected = graph(x)
    compiled = sc.compile(
        graph,
        (x,),
        devices="cpu",
        config=sc.CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        actual = compiled(x)
        assert actual.keys() == expected.keys()
        torch.testing.assert_close(actual, expected)
    finally:
        compiled.close()


@pytest.mark.parametrize("container", [tuple, list])
def test_module_graph_preserves_single_element_output_containers(container: type) -> None:
    output = container((sc.NodeOutput("project"),))
    graph = sc.ModuleGraph(
        (sc.ModuleNode("project", nn.Linear(4, 2), (sc.GraphInput(0),)),),
        outputs=output,
    ).eval()
    x = torch.randn(3, 4)
    expected = graph(x)
    assert type(expected) is container

    compiled = sc.compile(
        graph,
        (x,),
        devices="cpu",
        config=sc.CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        actual = compiled(x)
        assert type(actual) is container
        torch.testing.assert_close(actual, expected)
    finally:
        compiled.close()


@pytest.mark.parametrize(
    "factory, message",
    (
        (lambda: sc.ModuleGraph(()), "at least one node"),
        (
            lambda: sc.ModuleGraph((sc.ModuleNode("a", nn.Identity(), (sc.NodeOutput("missing"),)),)),
            "topological order",
        ),
        (
            lambda: sc.ModuleGraph(
                (
                    sc.ModuleNode("same", nn.Identity(), (sc.GraphInput(0),)),
                    sc.ModuleNode("same", nn.Identity(), (sc.GraphInput(0),)),
                )
            ),
            "Duplicate",
        ),
        (lambda: sc.ModuleGraph((sc.ModuleNode(1, nn.Identity(), (sc.GraphInput(0),)),)), "Node name"),  # type: ignore[arg-type]
    ),
)
def test_module_graph_rejects_invalid_graphs(factory: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_module_graph_rejects_non_string_argument_mapping_keys() -> None:
    with pytest.raises(TypeError, match="mapping keys"):
        sc.ModuleNode("a", nn.Identity(), ({1: sc.GraphInput(0)},))  # type: ignore[arg-type]


def test_module_graph_reports_bad_runtime_output_selector() -> None:
    graph = sc.ModuleGraph(
        (sc.ModuleNode("split", Split(), (sc.GraphInput(0),)),),
        outputs=(sc.NodeOutput("split", ("missing",)),),
    )
    with pytest.raises(ValueError, match="Cannot select"):
        graph(torch.ones(2))


def test_series_preserves_shared_module_parameters() -> None:
    shared = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    compiled = sc.compile_modules(
        (shared, nn.ReLU(), shared),
        (x,),
        devices="cpu",
        config=sc.CompileConfig(use_torch_compile=False, measure_regions=False),
    )
    try:
        torch.testing.assert_close(compiled(x), shared(torch.relu(shared(x))))
        shared_weights = [
            compiled.program.state_tensor(name)
            for name, target in compiled.program.state_bindings.items()
            if target.endswith("weight")
        ]
        assert shared_weights
        assert len({id(tensor) for tensor in shared_weights}) == 1
    finally:
        compiled.close()


def test_composed_modules_train_with_schedule_autograd() -> None:
    x = torch.randn(3, 4)
    target = torch.randn(3, 2)
    compiled = sc.compile_modules(
        (nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)),
        (x,),
        devices="cpu",
        config=sc.CompileConfig(
            allow_training=True,
            use_torch_compile=False,
            measure_regions=False,
            max_concurrent_regions=1,
        ),
    )
    try:
        compiled.train()
        optimizer = torch.optim.SGD(compiled.parameters(), lr=0.01)
        before = [parameter.detach().clone() for parameter in compiled.parameters()]
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(compiled(x), target)
        loss.backward()
        optimizer.step()
        after = list(compiled.parameters())
        assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after, strict=True))
        compiled.eval()
        assert compiled(x).shape == target.shape
    finally:
        compiled.close()
