"""Local diagnostic archive of documentation captured during a model refresh.

This is not part of the catalog, lifecycle history, or model export. The
exporter does not read it. Blobs are the UTF-8 text whose SHA-256 is already
stored on fact provenance.

Default retention keeps ``DEFAULT_ARCHIVE_KEEP`` completed generations (5).
``samyak model update --archive-sources --archive-keep N`` changes that limit.
``N`` must be at least 1. The flag is off unless the operator sets it, and a
missing archive does not change catalog bytes, history, or publication.

Layout under ``<cache>/models/source-archive/``::

    .lock                                  # exclusive writer lock; not catalog state
    blobs/sha256/<prefix>/<hex>
    refreshes/<refresh_id>/manifest.json
    refreshes/<refresh_id>/failures.json   # omitted when nothing failed
                                           # optional redirects[] on a failure

A generation counts toward retention only after its directory is renamed into
place with a manifest whose ``complete`` field is true and whose ``refresh_id``
matches the directory name. Dot-directories, truncated JSON, and
``complete: false`` are incomplete. They do not occupy a retention slot and
are not expired by garbage collection.

Garbage collection deletes a blob only when a successfully removed expired
generation listed that hash and no retained generation lists it. Blobs with
no completed manifest are left in place. If a retained manifest cannot be
validated, or an expired directory is still present after the removal attempt,
no blobs are deleted. An archive write failure does not publish ``complete:
true`` and does not run garbage collection.

Writers take an exclusive cross-process lock on ``.lock`` before the first
blob write and hold it through publication and retention. A failed acquisition
records nothing and does not run retention. The lock is not a catalog lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from samyak.__version__ import __version__
from samyak.model.http import TransportSecurityError, TransportTimeoutError, redirect_hops
from samyak.store.atomic import atomic_write_bytes, atomic_write_json

DEFAULT_ARCHIVE_KEEP = 5
ARCHIVE_DIRNAME = "source-archive"
_HASH_PREFIX = "sha256:"
_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_REFRESH_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_HTTP_STATUS = re.compile(r"returned HTTP (\d{3})\b")
_LOCK_NAME = ".lock"


class _CapturedBody(Protocol):
    source_id: str
    source_url: str
    retrieved_at: str
    media_type: str
    content_hash: str

    @property
    def body_bytes(self) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ClassifiedFailure:
    """Sanitized failure fields safe to persist. No headers and no cause text."""

    stage: str
    exception_class: str
    http_status: int | None
    message: str


def refresh_id_from_stamp(stamp: str) -> str:
    """Turn a verification timestamp into a sortable directory name."""
    text = stamp.strip()
    if text.endswith("+00:00"):
        text = text[: -len("+00:00")] + "Z"
    elif text.endswith("-00:00"):
        text = text[: -len("-00:00")] + "Z"
    text = text.replace(":", "-")
    cleaned = _REFRESH_ID_SAFE.sub("-", text).strip("-")
    return cleaned or "refresh"


def open_source_archive(models_dir: Path, *, refresh_stamp: str, keep: int) -> SourceArchive:
    """Create an in-memory generation rooted at ``models_dir/source-archive``."""
    return SourceArchive(
        root=models_dir / ARCHIVE_DIRNAME,
        refresh_id=refresh_id_from_stamp(refresh_stamp),
        package_version=__version__,
        keep=keep,
    )


def close_source_archive(archive: SourceArchive | None) -> None:
    """Finalize a generation. Filesystem errors do not propagate."""
    if archive is None:
        return
    try:
        archive.finalize()
    except Exception:
        archive.write_failed = True
    if archive.write_failed:
        print(
            "warning: diagnostic source archive could not be saved.",
            file=sys.stderr,
        )


def note_captured(archive: SourceArchive | None, provider_id: str, source: _CapturedBody) -> None:
    """Persist one captured document. A write error stays inside the archive."""
    if archive is not None:
        archive.record_captured(provider_id, source)


def note_failure(
    archive: SourceArchive | None,
    provider_id: str,
    source_id: str,
    source_url: str,
    exc: BaseException,
) -> None:
    """Record one sanitized failure. A write error stays inside the archive."""
    if archive is not None:
        archive.record_failure(provider_id, source_id, source_url, exc)


def note_parse_failure(
    archive: SourceArchive | None,
    provider_id: str,
    sources: Sequence[_CapturedBody],
    exc: BaseException,
) -> None:
    """Attribute a batch parse failure only when the message names that source.

    Optional pages are recorded one document at a time through ``note_failure``
    and keep that document's id. A batch of required documents uses this
    function. When the message names none of the captured source ids, the
    record uses the provider id and an empty URL instead of the first source.
    """
    if archive is None:
        return
    text = _public_message(exc)
    chosen: _CapturedBody | None = None
    for source in sources:
        if source.source_id and source.source_id in text:
            chosen = source
            break
    if chosen is None:
        archive.record_failure(provider_id, provider_id, "", exc)
        return
    archive.record_failure(provider_id, chosen.source_id, chosen.source_url, exc)


def classify_failure(exc: BaseException) -> ClassifiedFailure:
    """Map a capture exception onto a stage without reading raw network errors."""
    message = _public_message(exc)
    exception_class = type(exc).__name__
    status_match = _HTTP_STATUS.search(message)
    if status_match is not None:
        return ClassifiedFailure(
            stage="http_status",
            exception_class=exception_class,
            http_status=int(status_match.group(1)),
            message=message,
        )
    if _chain_isinstance(exc, TransportTimeoutError) or "timed out" in message:
        return ClassifiedFailure("timeout", exception_class, None, message)
    if _chain_isinstance(exc, UnicodeDecodeError) or "not UTF-8" in message:
        return ClassifiedFailure("decode", exception_class, None, message)
    if exception_class.endswith("ParseError") or (
        exception_class.endswith("AdapterError") and "Fetch" not in exception_class
    ):
        return ClassifiedFailure("parse", exception_class, None, message)
    security = _security_message(exc)
    if security is not None or _looks_like_security(message):
        stage = "content_type" if "content type" in (security or message) else "security"
        return ClassifiedFailure(stage, exception_class, None, message)
    return ClassifiedFailure("request", exception_class, None, message)


def blob_path(root: Path, content_hash: str) -> Path:
    """Return the content-addressed path for ``sha256:<hex>``."""
    hex_digest = content_hash.removeprefix(_HASH_PREFIX)
    return root / "blobs" / "sha256" / hex_digest[:2] / hex_digest


def completed_refresh_ids(root: Path) -> tuple[str, ...]:
    """Return completed generation ids, oldest first."""
    refreshes = root / "refreshes"
    if not refreshes.is_dir():
        return ()
    found: list[str] = []
    for entry in refreshes.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        manifest = _read_manifest(entry / "manifest.json")
        if manifest is None or manifest.get("complete") is not True:
            continue
        if manifest.get("refresh_id") != entry.name:
            continue
        found.append(entry.name)
    return tuple(sorted(found))


def load_manifest(root: Path, refresh_id: str) -> dict[str, object] | None:
    """Return a completed generation manifest, or None when it is not complete."""
    if refresh_id not in completed_refresh_ids(root):
        return None
    return _read_manifest(root / "refreshes" / refresh_id / "manifest.json")


def load_failures(root: Path, refresh_id: str) -> tuple[dict[str, object], ...]:
    """Return failure records for a completed generation. Missing file means none."""
    if refresh_id not in completed_refresh_ids(root):
        return ()
    path = root / "refreshes" / refresh_id / "failures.json"
    if not path.is_file():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        return ()
    failures = payload.get("failures")
    if not isinstance(failures, list):
        return ()
    return tuple(item for item in failures if isinstance(item, dict))


class _ArchiveLock:
    """Exclusive non-blocking lock shared by processes, released on close.

    ``flock`` on POSIX and ``msvcrt.locking`` on Windows are both released when
    the holding process exits. A second process that finds the lock busy does
    not write blobs and does not run retention.
    """

    def __init__(self, root: Path) -> None:
        self._path = root / _LOCK_NAME
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> bool:
        """Return True when this archive holds the lock. Never blocks."""
        if self._fd is not None:
            return True
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self._path.parent, 0o700)
            fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            return False
        try:
            if self._path.stat().st_size == 0:
                os.write(fd, b"\0")
            _try_exclusive_lock(fd)
            os.chmod(self._path, 0o600)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        """Unlock and close. A second release is a no-op."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        with suppress(OSError):
            _unlock_fd(fd)
        os.close(fd)


class SourceArchive:
    """One refresh generation. Blobs are written immediately; the manifest is not.

    ``finalize`` publishes the generation by renaming a partial directory into
    ``refreshes/<refresh_id>/`` only when every archive write succeeded.
    ``complete`` is true only on that published manifest. Callers must treat
    write failures as diagnostic only.
    """

    def __init__(self, root: Path, *, refresh_id: str, package_version: str, keep: int) -> None:
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ValueError("source archive retention must keep at least 1 generation")
        self.root = root
        self.refresh_id = refresh_id_from_stamp(refresh_id) if ":" in refresh_id else refresh_id
        if _REFRESH_ID_SAFE.search(self.refresh_id):
            self.refresh_id = refresh_id_from_stamp(self.refresh_id)
        self.package_version = package_version
        self.keep = keep
        self.write_failed = False
        self._lock = _ArchiveLock(root)
        self._lock_denied = False
        self._sources: list[dict[str, object]] = []
        self._failures: list[dict[str, object]] = []
        self._finalized = False

    def record_captured(self, provider_id: str, source: _CapturedBody) -> None:
        """Store the UTF-8 body when its hash matches, then remember the source."""
        if self._finalized:
            return
        if self.write_failed or self._lock_denied:
            self.write_failed = True
            return
        if not self._ensure_lock():
            return
        try:
            self._store_blob(source)
            self._sources.append(
                {
                    "provider_id": provider_id,
                    "source_id": source.source_id,
                    "source_url": source.source_url,
                    "content_hash": source.content_hash,
                    "retrieved_at": source.retrieved_at,
                    "media_type": source.media_type,
                    "byte_length": len(source.body_bytes),
                    "package_version": self.package_version,
                }
            )
        except Exception:
            self.write_failed = True

    def record_failure(
        self,
        provider_id: str,
        source_id: str,
        source_url: str,
        exc: BaseException,
    ) -> None:
        """Append one sanitized failure. The response body and headers are omitted."""
        try:
            classified = classify_failure(exc)
            record: dict[str, object] = {
                "provider_id": provider_id,
                "source_id": source_id,
                "source_url": source_url,
                "stage": classified.stage,
                "exception_class": classified.exception_class,
                "http_status": classified.http_status,
                "message": classified.message,
            }
            redirects = _redirect_diagnostics(exc)
            if redirects:
                record["redirects"] = redirects
            self._failures.append(record)
        except Exception:
            self.write_failed = True

    def finalize(self) -> None:
        """Publish a truthful generation, then retain, and always release the lock.

        ``complete: true`` is written only when no archive write has failed.
        A failed lock or an earlier write failure skips publication and
        garbage collection so this attempt cannot evict a completed generation.
        """
        if self._finalized:
            return
        try:
            if not self.write_failed and self._ensure_lock():
                self._publish()
                if not self.write_failed:
                    self._retain()
        finally:
            self._finalized = True
            self._lock.release()

    def _ensure_lock(self) -> bool:
        """Hold the archive lock, or remember that acquisition failed."""
        if self._lock_denied:
            self.write_failed = True
            return False
        if self._lock.acquire():
            return True
        self._lock_denied = True
        self.write_failed = True
        return False

    def _store_blob(self, source: _CapturedBody) -> None:
        payload = source.body_bytes
        if not isinstance(payload, bytes):
            raise TypeError("captured source body must be bytes")
        digest = hashlib.sha256(payload).hexdigest()
        content_hash = f"{_HASH_PREFIX}{digest}"
        if source.content_hash != content_hash:
            raise ValueError("captured source content_hash does not match its UTF-8 bytes")
        path = blob_path(self.root, content_hash)
        if path.is_file():
            existing = path.read_bytes()
            if hashlib.sha256(existing).hexdigest() != digest:
                raise ValueError("archived blob does not match its content address")
            return
        atomic_write_bytes(path, payload)
        os.chmod(path, 0o600)
        if self.root.exists():
            os.chmod(self.root, 0o700)
        written = path.read_bytes()
        if hashlib.sha256(written).hexdigest() != digest:
            raise ValueError("archived blob hash changed while writing")

    def _publish(self) -> None:
        refreshes = self.root / "refreshes"
        refreshes.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        os.chmod(refreshes, 0o700)
        refresh_id = self._unused_refresh_id(refreshes)
        self.refresh_id = refresh_id
        partial = refreshes / f".{refresh_id}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        try:
            manifest = {
                "refresh_id": refresh_id,
                "complete": True,
                "package_version": self.package_version,
                "retention_keep": self.keep,
                "sources": self._sources,
            }
            atomic_write_json(partial / "manifest.json", manifest)
            if self._failures:
                atomic_write_json(
                    partial / "failures.json",
                    {
                        "refresh_id": refresh_id,
                        "complete": True,
                        "failures": self._failures,
                    },
                )
            _replace(partial, refreshes / refresh_id)
        except Exception:
            if partial.exists():
                shutil.rmtree(partial, ignore_errors=True)
            raise

    def _unused_refresh_id(self, refreshes: Path) -> str:
        candidate = self.refresh_id
        suffix = 2
        while (refreshes / candidate).exists():
            candidate = f"{self.refresh_id}-{suffix}"
            suffix += 1
        return candidate

    def _retain(self) -> None:
        """Expire completed generations and delete only hashes those generations listed.

        Orphan blobs stay. Validation or removal failure skips every blob deletion.
        The caller holds the archive lock.
        """
        if self.write_failed:
            return
        completed = completed_refresh_ids(self.root)
        if not completed:
            return
        retained_ids = completed[-self.keep :]
        expired_ids = completed[: max(0, len(completed) - self.keep)]
        retained_hashes: set[str] = set()
        for refresh_id in retained_ids:
            hashes = _validated_manifest_hashes(self.root, refresh_id)
            if hashes is None:
                return
            retained_hashes.update(hashes)
        expired: list[tuple[str, set[str]]] = []
        for refresh_id in expired_ids:
            hashes = _validated_manifest_hashes(self.root, refresh_id)
            if hashes is None:
                return
            expired.append((refresh_id, hashes))
        if not expired:
            return
        refreshes = self.root / "refreshes"
        removed_hashes: set[str] = set()
        for refresh_id, hashes in expired:
            if not _remove_refresh_dir(refreshes / refresh_id):
                return
            removed_hashes.update(hashes)
        for refresh_id, _hashes in expired:
            if (refreshes / refresh_id).exists():
                return
        for content_hash in removed_hashes - retained_hashes:
            _delete_unreferenced_blob(self.root, content_hash)


def _redirect_diagnostics(exc: BaseException) -> list[dict[str, object]]:
    """Persist hop status, sanitized destination, and rejection reason only."""
    records: list[dict[str, object]] = []
    for hop in redirect_hops(exc):
        records.append(
            {
                "status": hop.status,
                "destination": hop.destination,
                "accepted": hop.accepted,
                "reason": hop.reason,
            }
        )
    return records


def _public_message(exc: BaseException) -> str:
    if exc.args and isinstance(exc.args[0], str) and exc.args[0].strip():
        return exc.args[0]
    return type(exc).__name__


def _chain(exc: BaseException) -> tuple[BaseException, ...]:
    found: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        found.append(current)
        current = current.__cause__
    return tuple(found)


def _chain_isinstance(exc: BaseException, kind: type[BaseException]) -> bool:
    return any(isinstance(item, kind) for item in _chain(exc))


def _security_message(exc: BaseException) -> str | None:
    for item in _chain(exc):
        if isinstance(item, TransportSecurityError):
            return _public_message(item)
    return None


def _looks_like_security(message: str) -> bool:
    lowered = message.lower()
    markers = (
        "content type",
        "redirect",
        "https",
        "query",
        "fragment",
        "permitted size",
        "content encoding",
        "credentials",
        "documentation host",
        "documentation path",
    )
    return any(marker in lowered for marker in markers)


def _replace(src: Path, dst: Path) -> None:
    """Rename a finished generation directory into place."""
    os.replace(src, dst)


def _read_manifest(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _validated_manifest_hashes(root: Path, refresh_id: str) -> set[str] | None:
    """Return content hashes, or None when the completed manifest is not usable."""
    manifest = _read_manifest(root / "refreshes" / refresh_id / "manifest.json")
    if manifest is None or manifest.get("complete") is not True:
        return None
    if manifest.get("refresh_id") != refresh_id:
        return None
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        return None
    hashes: set[str] = set()
    for item in sources:
        if not isinstance(item, dict):
            return None
        content_hash = item.get("content_hash")
        if not isinstance(content_hash, str) or _CONTENT_HASH.fullmatch(content_hash) is None:
            return None
        hashes.add(content_hash)
    return hashes


def _remove_refresh_dir(path: Path) -> bool:
    """Remove one generation directory. True only when it is actually gone."""
    if not path.exists():
        return True
    try:
        shutil.rmtree(path)
    except OSError:
        return False
    return not path.exists()


def _delete_unreferenced_blob(root: Path, content_hash: str) -> None:
    path = blob_path(root, content_hash)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return
    prefix = path.parent
    try:
        if prefix.is_dir() and not any(prefix.iterdir()):
            prefix.rmdir()
    except OSError:
        return


def _try_exclusive_lock(fd: int) -> None:
    """Take a non-blocking exclusive lock. Raise ``OSError`` when it is busy."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_fd(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)
