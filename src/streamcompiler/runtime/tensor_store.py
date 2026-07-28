"""Parameter provisioning for region execution.

Two production stores exist:

``ResidentParameterStore``
    Parameters stay in host RAM inside the compiled module. Zero-copy, fastest,
    used whenever the weights fit the configured budget.

``StreamingParameterStore``
    Parameters live only in the on-disk model pack. Blocks are read with real
    ``os.pread`` calls, cached under an enforced byte budget, and evicted by LRU
    once no region still needs them. A background prefetch thread fills the cache
    for upcoming regions while the current region computes (double buffering).

Both stores implement :class:`ParameterStore` so the runtime never branches on
storage strategy.
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from streamcompiler.errors import MemoryCapacityError, StorageError
from streamcompiler.storage.pack import load_pack_manifest, verify_block_checksum


class ParameterStore(ABC):
    """Supplies parameter/buffer tensors to regions on demand."""

    #: Whether :meth:`prefetch` can do anything useful. Resident stores set this to
    #: False so the executor does not build prefetch lists it will throw away.
    needs_prefetch = False

    @abstractmethod
    def acquire(self, name: str) -> torch.Tensor:
        """Return the tensor for ``name``, materializing it if necessary."""

    def release(self, names: tuple[str, ...]) -> None:
        """Signal that ``names`` are no longer needed by the caller."""
        return None

    def prefetch(self, names: tuple[str, ...]) -> None:
        """Optionally start asynchronous materialization of ``names``."""
        return None

    def stats(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None


class ResidentParameterStore(ParameterStore):
    """Serves tensors already resident in host memory."""

    kind = "resident"

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors
        # Resident tensors never change, so the report is computed once.
        total = sum(t.numel() * t.element_size() for t in tensors.values())
        self._stats = {"kind": self.kind, "resident_bytes": total, "tensor_count": len(tensors)}

    def acquire(self, name: str) -> torch.Tensor:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise StorageError(f"Unknown parameter {name}") from exc

    def stats(self) -> dict[str, Any]:
        return self._stats


@dataclass
class _Block:
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    checksum: str = ""


@dataclass
class StreamingStats:
    reads: int = 0
    bytes_read: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    prefetch_hits: int = 0
    prefetch_submitted: int = 0
    evictions: int = 0
    prefetch_dropped: int = 0
    peak_resident_bytes: int = 0
    resident_bytes: int = 0
    waits_for_prefetch: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reads": self.reads,
            "bytes_read": self.bytes_read,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_submitted": self.prefetch_submitted,
            "evictions": self.evictions,
            "prefetch_dropped": self.prefetch_dropped,
            "peak_resident_bytes": self.peak_resident_bytes,
            "resident_bytes": self.resident_bytes,
            "waits_for_prefetch": self.waits_for_prefetch,
            **self.extra,
        }


class StreamingParameterStore(ParameterStore):
    """Reads parameter blocks from a model pack under an enforced RAM budget."""

    kind = "streaming"
    needs_prefetch = True

    def __init__(
        self,
        pack_path: Path,
        bindings: dict[str, str],
        *,
        budget_bytes: int,
        pin_memory: bool = False,
    ) -> None:
        self._path = Path(pack_path)
        self._bindings = dict(bindings)
        self._budget = int(budget_bytes)
        self._pin_memory = pin_memory
        manifest = load_pack_manifest(self._path)
        self._blocks: dict[str, _Block] = {}
        by_logical = {entry["logical_id"]: entry for entry in manifest["tensors"]}
        for env_name, target in self._bindings.items():
            entry = by_logical.get(target)
            if entry is None:
                raise StorageError(f"Model pack {self._path} has no block for {target}")
            self._blocks[env_name] = _Block(
                offset=int(entry["offset"]),
                nbytes=int(entry["nbytes"]),
                shape=tuple(int(x) for x in entry["stored_shape"]),
                dtype=str(entry["stored_dtype"]),
                checksum=str(entry.get("checksum", "")),
            )
        largest = max((b.nbytes for b in self._blocks.values()), default=0)
        if largest > self._budget:
            raise MemoryCapacityError(
                f"Streaming budget {self._budget} bytes cannot hold the largest parameter block "
                f"({largest} bytes). Raise ram_budget_bytes or shard the parameter."
            )
        self._fd = os.open(self._path, os.O_RDONLY)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._pinned: dict[str, int] = {}
        self._lock = threading.RLock()
        self._inflight: dict[str, threading.Event] = {}
        self._stats = StreamingStats()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_queue: list[str] = []
        self._prefetch_cv = threading.Condition(self._lock)
        self._closed = False

    # ---- public API -------------------------------------------------
    def acquire(self, name: str) -> torch.Tensor:
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                self._cache.move_to_end(name)
                self._stats.cache_hits += 1
                if name in self._pinned:
                    self._pinned[name] += 1
                else:
                    self._pinned[name] = 1
                return cached
            event = self._inflight.get(name)
            if event is not None:
                self._stats.waits_for_prefetch += 1
        if event is not None:
            event.wait()
            with self._lock:
                cached = self._cache.get(name)
                if cached is not None:
                    self._stats.prefetch_hits += 1
                    self._cache.move_to_end(name)
                    self._pinned[name] = self._pinned.get(name, 0) + 1
                    return cached
        tensor = self._load(name, count_miss=True)
        if tensor is None:  # pragma: no cover - only droppable loads return None
            raise StorageError(f"Failed to stage parameter {name}")
        with self._lock:
            self._pinned[name] = self._pinned.get(name, 0) + 1
        return tensor

    def release(self, names: tuple[str, ...]) -> None:
        with self._lock:
            for name in names:
                remaining = self._pinned.get(name, 0) - 1
                if remaining <= 0:
                    self._pinned.pop(name, None)
                else:
                    self._pinned[name] = remaining
            self._evict_if_needed(0)

    def prefetch(self, names: tuple[str, ...]) -> None:
        wanted = [n for n in names if n in self._blocks]
        if not wanted:
            return
        with self._lock:
            if self._closed:
                return
            queued = 0
            for name in wanted:
                if name in self._cache or name in self._inflight:
                    continue
                self._inflight[name] = threading.Event()
                self._prefetch_queue.append(name)
                queued += 1
            if queued == 0:
                return
            self._stats.prefetch_submitted += queued
            if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
                self._prefetch_thread = threading.Thread(
                    target=self._prefetch_worker, name="streamcompiler-prefetch", daemon=True
                )
                self._prefetch_thread.start()
            self._prefetch_cv.notify_all()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._stats.resident_bytes = self._resident_bytes()
            data = self._stats.as_dict()
        data.update(
            {
                "kind": self.kind,
                "budget_bytes": self._budget,
                "pack_path": str(self._path),
                "block_count": len(self._blocks),
                "total_pack_bytes": sum(b.nbytes for b in self._blocks.values()),
            }
        )
        return data

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._prefetch_queue.clear()
            self._prefetch_cv.notify_all()
        thread = self._prefetch_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        with self._lock:
            for event in self._inflight.values():
                event.set()
            self._inflight.clear()
            self._cache.clear()
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1

    # ---- internals --------------------------------------------------
    def _prefetch_worker(self) -> None:
        while True:
            with self._prefetch_cv:
                while not self._prefetch_queue and not self._closed:
                    self._prefetch_cv.wait(timeout=0.5)
                if self._closed:
                    return
                name = self._prefetch_queue.pop(0)
            try:
                self._load(name, count_miss=False, droppable=True)
            except Exception:  # noqa: BLE001 - a failed prefetch must not kill execution
                with self._lock:
                    event = self._inflight.pop(name, None)
                if event is not None:
                    event.set()

    def _load(self, name: str, *, count_miss: bool, droppable: bool = False) -> torch.Tensor | None:
        """Read one parameter block from the pack.

        The store lock is held across eviction, the read and the cache insert so
        resident bytes can never transiently exceed the budget. Holding it during
        the read does not serialize compute: the executor touches the store only
        when it needs the next block, which is exactly when it would have to wait
        for that I/O anyway.

        ``droppable`` marks speculative prefetches. They are skipped rather than
        raising when pinned blocks currently fill the budget.
        """
        block = self._blocks.get(name)
        if block is None:
            raise StorageError(f"Unknown streamed parameter {name}")
        try:
            with self._lock:
                if self._closed:
                    raise StorageError("Streaming store is closed")
                if droppable and not self._can_stage(block.nbytes):
                    self._stats.prefetch_dropped += 1
                    return None
                self._evict_if_needed(block.nbytes)
                raw = os.pread(self._fd, block.nbytes, block.offset)
                if len(raw) != block.nbytes:
                    raise StorageError(f"Short read for {name}: expected {block.nbytes} bytes, read {len(raw)}")
                verify_block_checksum(raw, block.checksum, logical_id=name, path=self._path)
                dtype = getattr(torch, block.dtype, None)
                if dtype is None:
                    raise StorageError(f"Unsupported stored dtype {block.dtype} for {name}")
                tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(block.shape)
                if self._pin_memory and torch.cuda.is_available():  # pragma: no cover
                    tensor = tensor.pin_memory()
                self._stats.reads += 1
                self._stats.bytes_read += block.nbytes
                if count_miss:
                    self._stats.cache_misses += 1
                self._cache[name] = tensor
                self._cache.move_to_end(name)
                self._stats.peak_resident_bytes = max(self._stats.peak_resident_bytes, self._resident_bytes())
                return tensor
        finally:
            with self._lock:
                event = self._inflight.pop(name, None)
            if event is not None:
                event.set()

    def _resident_bytes(self) -> int:
        return sum(self._blocks[n].nbytes for n in self._cache)

    def _can_stage(self, incoming: int) -> bool:
        """True when ``incoming`` bytes fit after evicting every unpinned block."""
        evictable = sum(self._blocks[n].nbytes for n in self._cache if not self._pinned.get(n))
        return self._resident_bytes() - evictable + incoming <= self._budget

    def _evict_if_needed(self, incoming: int) -> None:
        resident = self._resident_bytes()
        if resident + incoming <= self._budget:
            return
        for name in list(self._cache.keys()):
            if resident + incoming <= self._budget:
                break
            if self._pinned.get(name):
                continue
            self._cache.pop(name, None)
            resident -= self._blocks[name].nbytes
            self._stats.evictions += 1
        if resident + incoming > self._budget:
            pinned = sum(self._blocks[n].nbytes for n in self._cache if self._pinned.get(n))
            raise MemoryCapacityError(
                f"Cannot stage {incoming} bytes within a {self._budget} byte budget; "
                f"{pinned} bytes are pinned by in-flight regions. "
                "Increase ram_budget_bytes or reduce max_region_nodes."
            )
