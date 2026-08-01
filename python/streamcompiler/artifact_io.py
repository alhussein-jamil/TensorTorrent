"""Atomic, integrity-checked artifact bundle helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from streamcompiler.errors import RuntimePlanError

INTEGRITY_MANIFEST = "artifact-integrity.json"
INTEGRITY_SCHEMA = "streamcompiler-artifact-integrity-v1"

_FALLBACK_LOCKS: dict[str, threading.Lock] = {}
_FALLBACK_LOCKS_GUARD = threading.Lock()


def _fsync_directory(path: Path) -> None:
    """Durably persist directory entry updates on POSIX filesystems."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write one file atomically and durably on the containing filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimePlanError(f"Artifact file {resolved_path} escapes bundle root {resolved_root}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimePlanError(f"Invalid artifact-relative path: {relative}")
    return relative.as_posix()


def write_integrity_manifest(root: Path, files: Iterable[Path]) -> Path:
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted({Path(p) for p in files}, key=lambda p: p.as_posix()):
        if path.name == INTEGRITY_MANIFEST:
            continue
        if path.is_symlink():
            raise RuntimePlanError(f"Artifact bundles cannot contain symlinks: {path}")
        relative = _safe_relative_path(root, path)
        if not path.is_file():
            raise RuntimePlanError(f"Artifact manifest entry is not a file: {path}")
        stat = path.stat()
        entries[relative] = {
            "size": int(stat.st_size),
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema": INTEGRITY_SCHEMA,
        "created_unix": time.time(),
        "files": entries,
    }
    target = root / INTEGRITY_MANIFEST
    atomic_write_json(target, manifest)
    return target


def verify_integrity_manifest(
    root: Path,
    *,
    required: bool = False,
    reject_unexpected_files: bool = True,
) -> dict[str, Any] | None:
    if root.is_symlink():
        raise RuntimePlanError(f"Artifact bundle root cannot be a symlink: {root}")
    manifest_path = root / INTEGRITY_MANIFEST
    if manifest_path.is_symlink():
        raise RuntimePlanError(f"Artifact integrity manifest cannot be a symlink: {manifest_path}")
    if not manifest_path.exists():
        if required:
            raise RuntimePlanError(f"Artifact integrity manifest missing: {manifest_path}")
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePlanError(f"Invalid artifact integrity manifest {manifest_path}: {exc}") from exc
    if payload.get("schema") != INTEGRITY_SCHEMA:
        raise RuntimePlanError(
            f"Unsupported artifact integrity schema {payload.get('schema')!r}; expected {INTEGRITY_SCHEMA!r}"
        )
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimePlanError("Artifact integrity manifest contains no files")
    root_resolved = root.resolve()
    for relative, expected in files.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimePlanError(f"Unsafe path in artifact integrity manifest: {relative!r}")
        candidate = root / relative
        if candidate.is_symlink():
            raise RuntimePlanError(f"Artifact manifest entry is a symlink: {relative}")
        path = candidate.resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError as exc:
            raise RuntimePlanError(f"Artifact manifest path escapes bundle: {relative!r}") from exc
        if not path.is_file():
            raise RuntimePlanError(f"Artifact file missing: {relative}")
        if not isinstance(expected, dict):
            raise RuntimePlanError(f"Invalid integrity metadata for {relative}")
        actual_size = path.stat().st_size
        expected_size = int(expected.get("size", -1))
        if actual_size != expected_size:
            raise RuntimePlanError(
                f"Artifact file size mismatch for {relative}: expected {expected_size}, got {actual_size}"
            )
        actual_hash = sha256_file(path)
        expected_hash = str(expected.get("sha256", ""))
        if not expected_hash or actual_hash != expected_hash:
            raise RuntimePlanError(f"Artifact checksum mismatch for {relative}")
    if reject_unexpected_files:
        discovered = list(root.rglob("*"))
        symlinks = sorted(path.relative_to(root).as_posix() for path in discovered if path.is_symlink())
        if symlinks:
            raise RuntimePlanError(f"Artifact bundle contains symlinks: {symlinks}")
        actual_files = {
            path.relative_to(root).as_posix()
            for path in discovered
            if path.is_file() and path.name != INTEGRITY_MANIFEST
        }
        unexpected = sorted(actual_files - set(files))
        if unexpected:
            raise RuntimePlanError(f"Artifact contains unmanifested files: {unexpected}")
    return dict(payload)


@contextmanager
def _publication_lock(destination: Path) -> Iterator[None]:
    """Serialize publishers targeting one artifact directory.

    Linux deployments use ``flock`` for cross-process exclusion. A process-local
    lock is retained as a portability fallback for filesystems/platforms without
    ``fcntl``. The lock file lives beside, not inside, the integrity-checked
    artifact bundle.
    """
    lock_path = destination.with_name(f".{destination.name}.publish.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # pragma: no cover - project targets POSIX Linux
        key = str(lock_path)
        with _FALLBACK_LOCKS_GUARD:
            lock = _FALLBACK_LOCKS.setdefault(key, threading.Lock())
        with lock:
            yield
        return

    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_replace_directory(destination: Path, writer: Callable[[Path], None]) -> Path:
    """Build a complete bundle beside ``destination`` and atomically publish it.

    Existing bundles are restored if publication fails. Temporary directories are
    always on the destination filesystem so ``os.replace`` remains atomic. Writers
    targeting the same bundle are serialized across processes.
    """
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(destination):
        if destination.exists() and not destination.is_dir():
            raise RuntimePlanError(f"Artifact destination exists and is not a directory: {destination}")
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
        backup = destination.with_name(f".{destination.name}.backup-{os.getpid()}-{time.time_ns()}")
        published = False
        try:
            writer(stage)
            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(stage, destination)
                _fsync_directory(destination.parent)
                published = True
            except BaseException:
                if backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                    _fsync_directory(destination.parent)
                raise
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            return destination
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            if published and backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
