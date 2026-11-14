"""Durable local transaction for a model catalog and its lifecycle history.

The committed representation remains the legacy ``catalog.json`` and
``lifecycle-history.json`` pair.  A write-ahead journal and staged after
images make an interrupted pair update recoverable without provider access.
The journal is the commit decision: recovery completes its validated after
images, never guesses from partially written live files.

External programs which open the two files independently can observe the
short interval between their atomic replacements.  Samyak readers recover a
pending journal before loading; callers that need a pair use
``load_catalog_history`` under the same lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from samyak.model.catalog import ModelCatalog
from samyak.model.errors import (
    CatalogDecodeError,
    CatalogSchemaError,
    CatalogStoreError,
    CatalogValidationError,
)
from samyak.model.history import LifecycleHistory, history_from_dict
from samyak.model.serialize import catalog_from_dict
from samyak.store.atomic import atomic_write_bytes

_JOURNAL = ".catalog-history-transaction.json"
_CATALOG_STAGE = ".catalog-history-catalog.stage"
_HISTORY_STAGE = ".catalog-history-history.stage"
_LOCK = ".catalog-history.lock"
_WAIT_SECONDS = 5.0


class CatalogHistoryTransactionError(CatalogStoreError):
    """A catalog/history transaction could not be safely committed or recovered."""


@dataclass(frozen=True, slots=True)
class _Journal:
    transaction_id: str
    state: str
    catalog_before: str | None
    catalog_after: str
    history_before: str | None
    history_after: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "transaction_id": self.transaction_id,
            "state": self.state,
            "catalog_before": self.catalog_before,
            "catalog_after": self.catalog_after,
            "history_before": self.history_before,
            "history_after": self.history_after,
        }


class CatalogHistoryTransaction:
    """Exclusive writer transaction rooted at one Samyak cache directory.

    Lock acquisition bounds waiting.  The lock spans recovery, baseline read,
    refresh calculation by the supported updater, and commit so a waiting
    writer always reads the latest committed pair.
    """

    def __init__(self, root: Path, *, wait_seconds: float = _WAIT_SECONDS) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.catalog_path = self.models_dir / "catalog.json"
        self.history_path = self.models_dir / "lifecycle-history.json"
        self._wait_seconds = wait_seconds
        self._fd: int | None = None

    def __enter__(self) -> CatalogHistoryTransaction:
        self.acquire()
        self.recover()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def acquire(self) -> None:
        if self._fd is not None:
            return
        self.models_dir.mkdir(parents=True, exist_ok=True)
        path = self.models_dir / _LOCK
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + self._wait_seconds
        try:
            while not _try_lock(fd):
                if time.monotonic() >= deadline:
                    raise CatalogHistoryTransactionError(
                        "timed out waiting for catalog/history writer lock"
                    )
                time.sleep(0.025)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        with suppress(OSError):
            _unlock(fd)
        os.close(fd)

    def load_pair(self) -> tuple[ModelCatalog | None, LifecycleHistory]:
        self._require_lock()
        catalog = None
        if self.catalog_path.is_file():
            try:
                catalog = catalog_from_dict(
                    json.loads(self.catalog_path.read_text(encoding="utf-8"))
                )
            except CatalogSchemaError as exc:
                raise CatalogHistoryTransactionError(str(exc)) from exc
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                CatalogDecodeError,
                CatalogValidationError,
                ValueError,
                TypeError,
            ) as exc:
                raise CatalogHistoryTransactionError("model catalog is corrupt") from exc
        if not self.history_path.is_file():
            return catalog, LifecycleHistory()
        try:
            history = history_from_dict(json.loads(self.history_path.read_text(encoding="utf-8")))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            CatalogDecodeError,
            CatalogValidationError,
            ValueError,
            TypeError,
        ) as exc:
            raise CatalogHistoryTransactionError("lifecycle history is corrupt") from exc
        return catalog, history

    def commit(self, catalog: ModelCatalog, history: LifecycleHistory) -> None:
        """Validate and durably publish a pair; recovery completes interruptions."""
        self._require_lock()
        try:
            catalog_from_dict(catalog.to_dict())
            history_from_dict(history.to_dict())
        except Exception as exc:
            raise CatalogHistoryTransactionError(
                "proposed catalog/history pair is invalid"
            ) from exc
        catalog_bytes = _json_bytes(catalog.to_dict())
        history_bytes = _json_bytes(history.to_dict())
        before_catalog = _digest_or_none(self.catalog_path)
        before_history = _digest_or_none(self.history_path)
        _write_durable(self.models_dir / _CATALOG_STAGE, catalog_bytes)
        _write_durable(self.models_dir / _HISTORY_STAGE, history_bytes)
        journal = _Journal(
            transaction_id=uuid.uuid4().hex,
            state="prepared",
            catalog_before=before_catalog,
            catalog_after=_digest(catalog_bytes),
            history_before=before_history,
            history_after=_digest(history_bytes),
        )
        self._write_journal(journal)
        self._apply_after_images(journal)
        self._write_journal(_replace_state(journal, "committed"))
        self._cleanup()

    def recover(self) -> None:
        """Complete a durable pending commit or fail closed on foreign changes."""
        self._require_lock()
        path = self.models_dir / _JOURNAL
        if not path.exists():
            self._cleanup_stages()
            return
        journal = self._read_journal(path)
        if journal.state == "committed":
            self._assert_after_or_before(journal, allow_before=False)
            self._cleanup()
            return
        self._assert_after_or_before(journal, allow_before=True)
        self._apply_after_images(journal)
        self._write_journal(_replace_state(journal, "committed"))
        self._cleanup()

    def _apply_after_images(self, journal: _Journal) -> None:
        catalog_stage = self.models_dir / _CATALOG_STAGE
        history_stage = self.models_dir / _HISTORY_STAGE
        if _digest_or_none(catalog_stage) != journal.catalog_after:
            raise CatalogHistoryTransactionError("catalog transaction stage is missing or corrupt")
        if _digest_or_none(history_stage) != journal.history_after:
            raise CatalogHistoryTransactionError("history transaction stage is missing or corrupt")
        _write_durable(self.catalog_path, catalog_stage.read_bytes())
        self._write_journal(_replace_state(journal, "catalog_visible"))
        _write_durable(self.history_path, history_stage.read_bytes())
        self._write_journal(_replace_state(journal, "history_visible"))

    def _assert_after_or_before(self, journal: _Journal, *, allow_before: bool) -> None:
        for path, before, after in (
            (self.catalog_path, journal.catalog_before, journal.catalog_after),
            (self.history_path, journal.history_before, journal.history_after),
        ):
            found = _digest_or_none(path)
            allowed = {after}
            if allow_before:
                allowed.add(before)
            if found not in allowed:
                raise CatalogHistoryTransactionError(
                    "catalog/history transaction does not match its journal; "
                    "manual recovery required"
                )

    def _write_journal(self, journal: _Journal) -> None:
        _write_durable(self.models_dir / _JOURNAL, _json_bytes(journal.to_dict()))

    def _read_journal(self, path: Path) -> _Journal:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("unsupported journal")
            values = {key: data.get(key) for key in _Journal.__dataclass_fields__}
            if values["state"] not in {
                "prepared",
                "catalog_visible",
                "history_visible",
                "committed",
            }:
                raise ValueError("invalid state")
            return _Journal(**values)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CatalogHistoryTransactionError(
                "catalog/history transaction journal is corrupt"
            ) from exc

    def _cleanup(self) -> None:
        for name in (_JOURNAL, _CATALOG_STAGE, _HISTORY_STAGE):
            with suppress(FileNotFoundError):
                (self.models_dir / name).unlink()
        _sync_dir(self.models_dir)

    def _cleanup_stages(self) -> None:
        for name in (_CATALOG_STAGE, _HISTORY_STAGE):
            with suppress(FileNotFoundError):
                (self.models_dir / name).unlink()

    def _require_lock(self) -> None:
        if self._fd is None:
            raise CatalogHistoryTransactionError("catalog/history transaction lock is not held")


def recover_catalog_history(root: Path) -> None:
    """Recover a pending supported update before a transaction-aware read."""
    with CatalogHistoryTransaction(root):
        pass


@contextmanager
def load_catalog_history(root: Path) -> Iterator[tuple[ModelCatalog | None, LifecycleHistory]]:
    """Read a recovered, mutually consistent pair while holding the reader lock."""
    with CatalogHistoryTransaction(root) as transaction:
        yield transaction.load_pair()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest_or_none(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _digest(path.read_bytes())


def _write_durable(path: Path, payload: bytes) -> None:
    atomic_write_bytes(path, payload)
    _sync_dir(path.parent)


def _sync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    with suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _replace_state(journal: _Journal, state: str) -> _Journal:
    return _Journal(
        transaction_id=journal.transaction_id,
        state=state,
        catalog_before=journal.catalog_before,
        catalog_after=journal.catalog_after,
        history_before=journal.history_before,
        history_after=journal.history_after,
    )


def _try_lock(fd: int) -> bool:
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
