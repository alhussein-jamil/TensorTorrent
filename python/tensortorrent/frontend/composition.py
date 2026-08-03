"""Composable PyTorch module graphs for single-artifact compilation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class GraphInput:
    """Reference one positional input of :class:`ModuleGraph`."""

    index: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"Graph input index must be a non-negative integer, got {self.index!r}")


@dataclass(frozen=True)
class NodeOutput:
    """Reference a node result, optionally selecting nested tuple/dict values."""

    node: str
    path: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.node, str) or not self.node:
            raise ValueError("Node output reference requires a non-empty node name")
        if not isinstance(self.path, tuple):
            if not isinstance(self.path, list):
                raise TypeError("Node output path must be a tuple or list")
            object.__setattr__(self, "path", tuple(self.path))
        if any(isinstance(part, bool) or not isinstance(part, (str, int)) for part in self.path):
            raise TypeError("Node output paths may contain only string keys and integer indices")


ValueRef = GraphInput | NodeOutput
GraphArgument = (
    ValueRef
    | None
    | bool
    | int
    | float
    | str
    | tuple["GraphArgument", ...]
    | list["GraphArgument"]
    | dict[str, "GraphArgument"]
)
GraphOutput = ValueRef | tuple["GraphOutput", ...] | list["GraphOutput"] | dict[str, "GraphOutput"]


def _walk_refs(value: GraphArgument) -> tuple[ValueRef, ...]:
    if isinstance(value, (GraphInput, NodeOutput)):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(ref for item in value for ref in _walk_refs(item))
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Graph argument mapping keys must be strings")
        return tuple(ref for item in value.values() for ref in _walk_refs(item))
    if value is None or isinstance(value, (bool, int, float, str)):
        return ()
    raise TypeError(
        "Graph arguments must contain value references, scalar constants, or nested "
        f"tuple/list/dict containers; got {type(value).__name__}"
    )


def _walk_output_refs(value: GraphOutput) -> tuple[ValueRef, ...]:
    if isinstance(value, (GraphInput, NodeOutput)):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(ref for item in value for ref in _walk_output_refs(item))
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Graph output mapping keys must be strings")
        return tuple(ref for item in value.values() for ref in _walk_output_refs(item))
    raise TypeError(f"Graph outputs must contain value references or tuple/list/dict containers, got {value!r}")


def _snapshot_argument(value: GraphArgument) -> GraphArgument:
    """Copy mutable containers so validated graph definitions cannot drift."""
    if isinstance(value, (GraphInput, NodeOutput)) or value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(_snapshot_argument(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_argument(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Graph argument mapping keys must be strings")
        return {key: _snapshot_argument(item) for key, item in value.items()}
    raise TypeError(
        "Graph arguments must contain value references, scalar constants, or nested "
        f"tuple/list/dict containers; got {type(value).__name__}"
    )


def _snapshot_output(value: GraphOutput) -> GraphOutput:
    if isinstance(value, (GraphInput, NodeOutput)):
        return value
    if isinstance(value, tuple):
        return tuple(_snapshot_output(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_output(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Graph output mapping keys must be strings")
        return {key: _snapshot_output(item) for key, item in value.items()}
    raise TypeError(f"Graph outputs must contain value references or tuple/list/dict containers, got {value!r}")


@dataclass(frozen=True)
class ModuleNode:
    """One named module invocation in a :class:`ModuleGraph`."""

    name: str
    module: torch.nn.Module
    inputs: tuple[GraphArgument, ...]
    kwargs: Mapping[str, GraphArgument] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "." in self.name:
            raise ValueError(f"Node name must be non-empty and contain no '.', got {self.name!r}")
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError(f"Node {self.name!r} requires a torch.nn.Module")
        if not isinstance(self.inputs, tuple):
            object.__setattr__(self, "inputs", tuple(self.inputs))
        if not isinstance(self.kwargs, Mapping):
            raise TypeError(f"Node {self.name!r} kwargs must be a mapping")
        object.__setattr__(self, "inputs", tuple(_snapshot_argument(value) for value in self.inputs))
        object.__setattr__(self, "kwargs", {key: _snapshot_argument(value) for key, value in self.kwargs.items()})
        if any(not isinstance(key, str) or not key.isidentifier() for key in self.kwargs):
            raise ValueError(f"Node {self.name!r} keyword names must be valid Python identifiers")


class ModuleGraph(torch.nn.Module):
    """A validated DAG of real ``nn.Module`` calls.

    Nodes are declared in topological order. Export sees the whole composition,
    so TensorTorrent partitions and schedules it as one model rather than
    chaining independently compiled artifacts with hidden transfers.
    """

    def __init__(self, nodes: Sequence[ModuleNode], outputs: GraphOutput | None = None) -> None:
        super().__init__()
        if not nodes:
            raise ValueError("ModuleGraph requires at least one node")
        ordered = tuple(
            ModuleNode(
                name=node.name,
                module=node.module,
                inputs=tuple(_snapshot_argument(value) for value in node.inputs),
                kwargs={key: _snapshot_argument(value) for key, value in node.kwargs.items()},
            )
            if isinstance(node, ModuleNode)
            else node
            for node in nodes
        )
        seen: set[str] = set()
        modules: dict[str, torch.nn.Module] = {}
        max_input = -1
        for node in ordered:
            if not isinstance(node, ModuleNode):
                raise TypeError(f"Expected ModuleNode, got {type(node).__name__}")
            if node.name in seen:
                raise ValueError(f"Duplicate module graph node {node.name!r}")
            for ref in (ref for value in (*node.inputs, *node.kwargs.values()) for ref in _walk_refs(value)):
                if isinstance(ref, GraphInput):
                    max_input = max(max_input, ref.index)
                elif ref.node not in seen:
                    raise ValueError(
                        f"Node {node.name!r} references {ref.node!r} before it is defined; "
                        "nodes must be in topological order"
                    )
            seen.add(node.name)
            modules[node.name] = node.module

        resolved_outputs = _snapshot_output(outputs) if outputs is not None else NodeOutput(ordered[-1].name)
        output_refs = _walk_output_refs(resolved_outputs)
        if not output_refs:
            raise ValueError("ModuleGraph requires at least one output")
        for ref in output_refs:
            if isinstance(ref, GraphInput):
                max_input = max(max_input, ref.index)
            elif ref.node not in seen:
                raise ValueError(f"Graph output references unknown node {ref.node!r}")

        self.modules_by_name = torch.nn.ModuleDict(modules)
        self._nodes = ordered
        self._outputs = resolved_outputs
        self._input_arity = max_input + 1

    @classmethod
    def series(cls, modules: Sequence[torch.nn.Module], *, names: Sequence[str] | None = None) -> ModuleGraph:
        """Compose modules in series, passing each complete output to the next."""
        ordered = tuple(modules)
        if not ordered:
            raise ValueError("ModuleGraph.series requires at least one module")
        node_names = tuple(names) if names is not None else tuple(f"stage_{index}" for index in range(len(ordered)))
        if len(node_names) != len(ordered):
            raise ValueError(f"Expected {len(ordered)} names, received {len(node_names)}")
        nodes = []
        for index, (name, module) in enumerate(zip(node_names, ordered, strict=True)):
            source: ValueRef = GraphInput(0) if index == 0 else NodeOutput(node_names[index - 1])
            nodes.append(ModuleNode(name=name, module=module, inputs=(source,)))
        return cls(nodes)

    def forward(self, *inputs: Any) -> Any:
        if len(inputs) != self._input_arity:
            raise ValueError(f"ModuleGraph expected {self._input_arity} inputs, received {len(inputs)}")
        values: dict[str, Any] = {}
        for node in self._nodes:
            args = tuple(self._resolve_argument(arg, inputs, values) for arg in node.inputs)
            kwargs = {key: self._resolve_argument(arg, inputs, values) for key, arg in node.kwargs.items()}
            values[node.name] = self.modules_by_name[node.name](*args, **kwargs)
        return self._resolve_output(self._outputs, inputs, values)

    @staticmethod
    def _resolve(ref: ValueRef, inputs: tuple[Any, ...], values: dict[str, Any]) -> Any:
        value = inputs[ref.index] if isinstance(ref, GraphInput) else values[ref.node]
        if isinstance(ref, NodeOutput):
            for part in ref.path:
                try:
                    value = value[part]
                except (IndexError, KeyError, TypeError) as exc:
                    raise ValueError(f"Cannot select {ref.node!r} output path {ref.path!r}") from exc
        return value

    @classmethod
    def _resolve_argument(cls, argument: GraphArgument, inputs: tuple[Any, ...], values: dict[str, Any]) -> Any:
        if isinstance(argument, (GraphInput, NodeOutput)):
            return cls._resolve(argument, inputs, values)
        if isinstance(argument, tuple):
            return tuple(cls._resolve_argument(item, inputs, values) for item in argument)
        if isinstance(argument, list):
            return [cls._resolve_argument(item, inputs, values) for item in argument]
        if isinstance(argument, dict):
            return {key: cls._resolve_argument(item, inputs, values) for key, item in argument.items()}
        return argument

    @classmethod
    def _resolve_output(cls, output: GraphOutput, inputs: tuple[Any, ...], values: dict[str, Any]) -> Any:
        if isinstance(output, (GraphInput, NodeOutput)):
            return cls._resolve(output, inputs, values)
        if isinstance(output, tuple):
            return tuple(cls._resolve_output(item, inputs, values) for item in output)
        if isinstance(output, list):
            return [cls._resolve_output(item, inputs, values) for item in output]
        return {key: cls._resolve_output(item, inputs, values) for key, item in output.items()}
