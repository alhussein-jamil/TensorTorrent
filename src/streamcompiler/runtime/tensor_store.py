"""Parameter provisioning for region execution.

Two production stores exist:

``ResidentParameterStore``
    Parameters stay in host RAM inside the compiled module. Zero-copy, fastest,
    used whenever the weights fit the configured budget.

``StreamingParameterStore``
    Parameters live only in the on-disk model pack. When the native extension is
    loaded, :class:`NativeStreamingStore` owns positional reads, byte cache,
    shared in-flight loads, and prefetch. Python only tensorizes bytes at the
    materialization boundary and enforces the decoded-tensor RAM budget.

Both stores implement :class:`ParameterStore` so the runtime never branches on
storage strategy.
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from streamcompiler.errors import MemoryCapacityError, StorageError
from streamcompiler.storage.pack import load_pack_manifest, verify_block_checksum
from streamcompiler.storage.native_pack import open_native_pack_reader, open_native_streaming_store


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

    def begin_execution(self) -> None:
        """Reset per-call I/O interval accounting before a new ``run``."""
        return None

    def record_compute_intervals(self, intervals: Sequence[tuple[float, float]]) -> None:
        """Optional hook: attach region compute windows for I/O overlap accounting."""
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
    compression: str = "none"
    logical_shape: tuple[int, ...] | None = None
    logical_dtype: str | None = None
    scale: float | None = None
    zero_point: int = 0


@dataclass(frozen=True)
class IoInterval:
    """One real ``os.pread`` window, timed with ``time.perf_counter``."""

    name: str
    start_s: float
    end_s: float
    nbytes: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping half-open time windows."""
    if not intervals:
        return []
    ordered = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not ordered:
        return []
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def intersect_interval_length(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> float:
    """Wall-clock seconds covered by both interval sets simultaneously."""
    a = merge_intervals(left)
    b = merge_intervals(right)
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


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
    io_time_s: float = 0.0
    acquire_stall_s: float = 0.0
    io_overlapped_with_compute_s: float = 0.0
    exposed_io_s: float = 0.0
    duplicate_reads_avoided: int = 0
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
            "io_time_s": self.io_time_s,
            "acquire_stall_s": self.acquire_stall_s,
            "io_overlapped_with_compute_s": self.io_overlapped_with_compute_s,
            "exposed_io_s": self.exposed_io_s,
            "duplicate_reads_avoided": self.duplicate_reads_avoided,
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
        self._env_to_key = dict(bindings)
        self._budget = int(budget_bytes)
        self._pin_memory = pin_memory
        manifest = load_pack_manifest(self._path)
        self._blocks: dict[str, _Block] = {}
        by_logical = {entry["logical_id"]: entry for entry in manifest["tensors"]}
        for env_name, target in self._env_to_key.items():
            if target in self._blocks:
                continue
            entry = by_logical.get(target)
            if entry is None:
                raise StorageError(f"Model pack {self._path} has no block for {target}")
            self._blocks[target] = _Block(
                offset=int(entry["offset"]),
                nbytes=int(entry["nbytes"]),
                shape=tuple(int(x) for x in entry["stored_shape"]),
                dtype=str(entry["stored_dtype"]),
                checksum=str(entry.get("checksum", "")),
                compression=str(entry.get("compression", "none")),
                logical_shape=tuple(int(x) for x in entry["logical_shape"])
                if entry.get("logical_shape") is not None
                else None,
                logical_dtype=str(entry["logical_dtype"]) if entry.get("logical_dtype") else None,
                scale=float(entry["scale"]) if entry.get("scale") is not None else None,
                zero_point=int(entry.get("zero_point", 0) or 0),
            )
            del env_name
        largest = max((b.nbytes for b in self._blocks.values()), default=0)
        if largest > self._budget:
            raise MemoryCapacityError(
                f"Streaming budget {self._budget} bytes cannot hold the largest parameter block "
                f"({largest} bytes). Raise ram_budget_bytes or shard the parameter."
            )
        self._fd = os.open(self._path, os.O_RDONLY)
        self._native_store = open_native_streaming_store(
            self._path, manifest, capacity_bytes=self._budget
        )
        # Legacy reader kept only when native streaming store is unavailable.
        self._native_reader = (
            None if self._native_store is not None else open_native_pack_reader(self._path, manifest)
        )
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._pinned: dict[str, int] = {}
        self._staging: dict[str, int] = {}
        self._lock = threading.RLock()
        self._inflight: dict[str, threading.Event] = {}
        self._stats = StreamingStats()
        self._io_intervals: list[IoInterval] = []
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_queue: list[str] = []
        self._prefetch_cv = threading.Condition(self._lock)
        self._closed = False

    def _storage_key(self, name: str) -> str:
        """Map an environment value name to its unique pack/storage id."""
        try:
            return self._env_to_key[name]
        except KeyError as exc:
            raise StorageError(f"Unknown parameter {name}") from exc

    # ---- public API -------------------------------------------------
    def acquire(self, name: str) -> torch.Tensor:
        key = self._storage_key(name)
        stall_start = time.perf_counter()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._stats.cache_hits += 1
                self._stats.duplicate_reads_avoided += 1
                if key in self._pinned:
                    self._pinned[key] += 1
                else:
                    self._pinned[key] = 1
                return cached
            event = self._inflight.get(key)
            if event is not None:
                self._stats.waits_for_prefetch += 1
        if event is not None:
            event.wait()
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._stats.prefetch_hits += 1
                    self._stats.duplicate_reads_avoided += 1
                    self._cache.move_to_end(key)
                    self._pinned[key] = self._pinned.get(key, 0) + 1
                    self._stats.acquire_stall_s += time.perf_counter() - stall_start
                    return cached
        tensor = self._load(key, count_miss=True)
        if tensor is None:  # pragma: no cover - only droppable loads return None
            raise StorageError(f"Failed to stage parameter {name}")
        with self._lock:
            self._pinned[key] = self._pinned.get(key, 0) + 1
            self._stats.acquire_stall_s += time.perf_counter() - stall_start
            return tensor

    def release(self, names: tuple[str, ...]) -> None:
        with self._lock:
            for name in names:
                if name not in self._env_to_key:
                    continue
                key = self._env_to_key[name]
                remaining = self._pinned.get(key, 0) - 1
                if remaining <= 0:
                    self._pinned.pop(key, None)
                else:
                    self._pinned[key] = remaining
            self._evict_if_needed(0)

    def prefetch(self, names: tuple[str, ...]) -> None:
        keys: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name not in self._env_to_key:
                continue
            key = self._env_to_key[name]
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
        if not keys:
            return
        pending: list[str] = []
        with self._lock:
            if self._closed:
                return
            queued = 0
            for key in keys:
                if key in self._cache or key in self._inflight or key in self._staging:
                    continue
                self._inflight[key] = threading.Event()
                self._prefetch_queue.append(key)
                pending.append(key)
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
        # Native owns byte pread + shared inflight; worker only tensorizes.
        if self._native_store is not None and pending:
            self._native_store.prefetch(pending)

    def begin_execution(self) -> None:
        """Clear per-call I/O windows so overlap stats describe this ``run`` only."""
        with self._lock:
            self._io_intervals.clear()
            self._stats.io_overlapped_with_compute_s = 0.0
            self._stats.exposed_io_s = 0.0
            self._stats.extra.pop("io_interval_count", None)
            self._stats.extra.pop("compute_interval_count", None)

    def record_compute_intervals(self, intervals: Sequence[tuple[float, float]]) -> None:
        """Score recorded ``pread`` windows against region compute intervals."""
        with self._lock:
            io_windows = [(iv.start_s, iv.end_s) for iv in self._io_intervals]
            io_union = merge_intervals(io_windows)
            compute = merge_intervals(intervals)
            overlapped = intersect_interval_length(io_union, compute)
            io_wall = sum(end - start for start, end in io_union)
            self._stats.io_overlapped_with_compute_s = overlapped
            self._stats.exposed_io_s = max(0.0, io_wall - overlapped)
            self._stats.extra["io_interval_count"] = len(self._io_intervals)
            self._stats.extra["compute_interval_count"] = len(compute)

    def io_intervals(self) -> tuple[IoInterval, ...]:
        with self._lock:
            return tuple(self._io_intervals)

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
                "native_streaming": self._native_store is not None,
            }
        )
        if self._native_store is not None:
            native = dict(self._native_store.stats())
            data["native_store"] = native
            # Surface authoritative native I/O counters alongside Python tensor-cache stats.
            data["native_bytes_read"] = int(native.get("bytes_read", 0))
            data["native_prefetch_submitted"] = int(native.get("prefetch_submitted", 0))
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
            self._staging.clear()
            self._cache.clear()
            if self._native_store is not None:
                self._native_store.close()
                self._native_store = None
            self._native_reader = None
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
            except (StorageError, MemoryCapacityError, OSError) as exc:
                # Speculative prefetch failure must not kill the worker; Load retries.
                with self._lock:
                    event = self._inflight.pop(name, None)
                    self._staging.pop(name, None)
                    self._stats.extra["prefetch_errors"] = int(self._stats.extra.get("prefetch_errors", 0)) + 1
                    self._stats.extra["last_prefetch_error"] = f"{type(exc).__name__}: {exc}"
                if event is not None:
                    event.set()

    def _load(self, name: str, *, count_miss: bool, droppable: bool = False) -> torch.Tensor | None:
        """Read one parameter block from the pack.

        Budget reservation and cache updates hold the store lock. The ``os.pread``
        itself runs unlocked so a cache hit or release on another name can proceed
        while I/O is in flight, and so I/O can overlap region compute on the caller
        thread.
        """
        block = self._blocks.get(name)
        if block is None:
            raise StorageError(f"Unknown streamed parameter {name}")
        wait_event: threading.Event | None = None
        owns_load = False
        with self._lock:
            if self._closed:
                raise StorageError("Streaming store is closed")
            cached = self._cache.get(name)
            if cached is not None:
                if count_miss:
                    self._stats.cache_hits += 1
                    self._stats.duplicate_reads_avoided += 1
                return cached
            if name in self._staging:
                if droppable:
                    return None
                wait_event = self._inflight.setdefault(name, threading.Event())
            else:
                if droppable and not self._can_stage(block.nbytes):
                    self._stats.prefetch_dropped += 1
                    event = self._inflight.pop(name, None)
                    if event is not None:
                        event.set()
                    return None
                self._evict_if_needed(block.nbytes)
                if self._resident_bytes() + block.nbytes > self._budget:
                    if droppable:
                        self._stats.prefetch_dropped += 1
                        event = self._inflight.pop(name, None)
                        if event is not None:
                            event.set()
                        return None
                    pinned = sum(self._blocks[n].nbytes for n in self._cache if self._pinned.get(n))
                    raise MemoryCapacityError(
                        f"Cannot stage {block.nbytes} bytes within a {self._budget} byte budget; "
                        f"{pinned} bytes are pinned by in-flight regions. "
                        "Increase ram_budget_bytes or reduce max_region_nodes."
                    )
                self._staging[name] = block.nbytes
                self._inflight.setdefault(name, threading.Event())
                owns_load = True
                fd = self._fd
        if wait_event is not None:
            wait_event.wait()
            with self._lock:
                cached = self._cache.get(name)
                if cached is not None:
                    return cached
            if droppable:
                return None
            raise StorageError(f"Failed to stage parameter {name}")
        try:
            io_start = time.perf_counter()
            if self._native_store is not None:
                raw = bytes(self._native_store.acquire_bytes(name))
                with self._lock:
                    self._stats.extra["native_streaming_acquire"] = (
                        int(self._stats.extra.get("native_streaming_acquire", 0)) + 1
                    )
            elif self._native_reader is not None:
                raw = bytes(self._native_reader.pread(name))
                with self._lock:
                    self._stats.extra["native_pread"] = int(self._stats.extra.get("native_pread", 0)) + 1
            else:
                raw = os.pread(fd, block.nbytes, block.offset)
                with self._lock:
                    self._stats.extra["python_pread"] = int(self._stats.extra.get("python_pread", 0)) + 1
            io_end = time.perf_counter()
            if len(raw) != block.nbytes:
                raise StorageError(f"Short read for {name}: expected {block.nbytes} bytes, read {len(raw)}")
            if self._native_store is None and self._native_reader is None:
                verify_block_checksum(raw, block.checksum, logical_id=name, path=self._path)
            # Native pread already verified checksum_crc32 when present in the pack.
            dtype = getattr(torch, block.dtype, None)
            if dtype is None:
                raise StorageError(f"Unsupported stored dtype {block.dtype} for {name}")
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(block.shape)
            if block.compression == "int8_affine":
                if block.scale is None:
                    raise StorageError(f"int8_affine block {name} missing scale")
                logical_dtype = getattr(torch, block.logical_dtype or "float32", torch.float32)
                logical_shape = block.logical_shape or block.shape
                tensor = ((tensor.float() - float(block.zero_point)) * float(block.scale)).to(logical_dtype)
                tensor = tensor.reshape(logical_shape)
            if self._pin_memory and torch.cuda.is_available():  # pragma: no cover
                tensor = tensor.pin_memory()
            if self._native_store is not None:
                self._native_store.release(name)
            with self._lock:
                self._stats.reads += 1
                self._stats.bytes_read += block.nbytes
                self._stats.io_time_s += io_end - io_start
                self._io_intervals.append(IoInterval(name=name, start_s=io_start, end_s=io_end, nbytes=block.nbytes))
                if count_miss:
                    self._stats.cache_misses += 1
                self._cache[name] = tensor
                self._cache.move_to_end(name)
                self._staging.pop(name, None)
                self._stats.peak_resident_bytes = max(self._stats.peak_resident_bytes, self._resident_bytes())
                return tensor
        except Exception:
            with self._lock:
                self._staging.pop(name, None)
            raise
        finally:
            if owns_load:
                with self._lock:
                    event = self._inflight.pop(name, None)
                if event is not None:
                    event.set()

    def _resident_bytes(self) -> int:
        cached = sum(self._blocks[n].nbytes for n in self._cache)
        staging = sum(self._staging.values())
        return cached + staging

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
            staging = sum(self._staging.values())
            raise MemoryCapacityError(
                f"Cannot stage {incoming} bytes within a {self._budget} byte budget; "
                f"{pinned} bytes are pinned by in-flight regions"
                + (f" and {staging} bytes are staging" if staging else "")
                + ". Increase ram_budget_bytes or reduce max_region_nodes."
            )
