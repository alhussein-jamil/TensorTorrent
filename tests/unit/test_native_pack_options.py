"""Configuration plumbing for the native multi-reader streaming store."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tensortorrent.storage.native_pack as native_pack
from tensortorrent.errors import StorageError

_MANIFEST = {
    "version": 1,
    "tensors": [
        {
            "logical_id": "w",
            "offset": 0,
            "nbytes": 4,
            "stored_dtype": "float32",
            "stored_shape": [1],
        }
    ],
}


def test_streaming_options_are_forwarded_to_native(monkeypatch: Any) -> None:
    calls: list[tuple[Any, ...]] = []

    class Store:
        @staticmethod
        def open(*args: Any) -> object:
            calls.append(args)
            return object()

    monkeypatch.setattr(native_pack, "native_available", lambda: True)
    monkeypatch.setattr(native_pack, "require_native", lambda: SimpleNamespace(NativeStreamingStore=Store))

    result = native_pack.open_native_streaming_store(
        Path("model.pack"),
        _MANIFEST,
        capacity_bytes=1024,
        io_workers=4,
        queue_limit=96,
    )
    assert result is not None
    assert calls and calls[0][2:] == (1024, 4, 96)


def test_legacy_native_open_signature_falls_back(monkeypatch: Any) -> None:
    calls: list[tuple[Any, ...]] = []

    class Store:
        @staticmethod
        def open(*args: Any) -> object:
            calls.append(args)
            if len(args) == 5:
                raise TypeError("legacy extension")
            return object()

    monkeypatch.setattr(native_pack, "native_available", lambda: True)
    monkeypatch.setattr(native_pack, "require_native", lambda: SimpleNamespace(NativeStreamingStore=Store))

    result = native_pack.open_native_streaming_store(
        "model.pack",
        _MANIFEST,
        capacity_bytes=1024,
        io_workers=3,
        queue_limit=32,
    )
    assert result is not None
    assert [len(call) for call in calls] == [5, 3]


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"capacity_bytes": 0}, "capacity_bytes"),
        ({"capacity_bytes": 1, "io_workers": 0}, "io_workers"),
        ({"capacity_bytes": 1, "queue_limit": 0}, "queue_limit"),
    ],
)
def test_invalid_streaming_options_are_rejected(kwargs: dict[str, int], field: str) -> None:
    with pytest.raises(StorageError, match=field):
        native_pack.open_native_streaming_store("model.pack", _MANIFEST, **kwargs)
