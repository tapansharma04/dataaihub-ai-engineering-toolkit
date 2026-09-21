"""Local lifecycle-history sidecar. Not catalog current state. Not corpus runs."""

from __future__ import annotations

import json
from pathlib import Path

from samyak.model.errors import (
    CatalogDecodeError,
    CatalogValidationError,
    HistorySchemaError,
    HistoryStoreError,
)
from samyak.model.history import (
    LifecycleHistory,
    LifecycleHistoryEvent,
    history_from_dict,
)
from samyak.model.store import MODELS_DIRNAME
from samyak.store.atomic import atomic_write_json
from samyak.store.paths import default_cache_dir

HISTORY_FILENAME = "lifecycle-history.json"


class FileLifecycleHistoryStore:
    """Read and atomically replace ``<cache>/models/lifecycle-history.json``.

    Missing history is an empty log. Corrupt or unsupported history fails
    closed and is not repaired. This store never writes ``catalog.json``.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_dir()
        self.models_dir = self.root / MODELS_DIRNAME
        self.history_path = self.models_dir / HISTORY_FILENAME

    def exists(self) -> bool:
        return self.history_path.is_file()

    def load(self) -> LifecycleHistory:
        if not self.exists():
            return LifecycleHistory()
        try:
            text = self.history_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise HistoryStoreError("lifecycle history is corrupt") from exc
        except OSError as exc:
            raise HistoryStoreError("lifecycle history could not be read") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HistoryStoreError("lifecycle history is corrupt") from exc
        try:
            return history_from_dict(payload)
        except HistorySchemaError:
            raise
        except (CatalogDecodeError, CatalogValidationError, TypeError, ValueError) as exc:
            raise HistoryStoreError("lifecycle history is corrupt") from exc

    def append(self, events: tuple[LifecycleHistoryEvent, ...]) -> None:
        """Atomically append events. No-op when ``events`` is empty."""
        if not events:
            return
        current = self.load()
        combined = LifecycleHistory(events=current.events + events)
        try:
            payload = combined.to_dict()
            history_from_dict(payload)
        except (
            CatalogDecodeError,
            CatalogValidationError,
            HistorySchemaError,
            TypeError,
            ValueError,
        ) as exc:
            raise HistoryStoreError("lifecycle history could not be serialized") from exc
        try:
            atomic_write_json(self.history_path, payload)
        except OSError as exc:
            raise HistoryStoreError("lifecycle history could not be written") from exc
