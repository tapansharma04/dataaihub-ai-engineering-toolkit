"""Local Model Intelligence catalog persistence.

Domain objects do not know about files or the network. This module is the
filesystem overlay only. A bundled snapshot is a later milestone.

Layout (under the Samyak cache directory)::

    $SAMYAK_CACHE_DIR/
      runs/                 # corpus run history; never written here
      models/
        catalog.json        # live overlay; the only atomic correctness boundary
        catalog.json.bak    # best-effort previous overlay; not part of the commit invariant

``catalog.json`` is replaced atomically (temp file, fsync, ``os.replace``).
``catalog.json.bak`` is updated only after a successful live replace, and a
backup write failure must not roll back or invalidate the live catalog.

``SAMYAK_CACHE_DIR`` overrides the default (``~/.cache/samyak``,
``$XDG_CACHE_HOME/samyak``, or ``%LOCALAPPDATA%/samyak`` on Windows).
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Protocol

from samyak.model.catalog import ModelCatalog
from samyak.model.errors import (
    CatalogDecodeError,
    CatalogNotFoundError,
    CatalogSchemaError,
    CatalogStoreError,
    CatalogValidationError,
)
from samyak.model.serialize import catalog_from_dict, catalog_from_json
from samyak.store.atomic import atomic_write_bytes, atomic_write_json
from samyak.store.paths import default_cache_dir

MODELS_DIRNAME = "models"
CATALOG_FILENAME = "catalog.json"
CATALOG_BACKUP_FILENAME = "catalog.json.bak"


class CatalogSource(Protocol):
    """Load a validated catalog snapshot.

    Implementations (bundled package data, user-cache overlay, tests) live
    outside the domain types.
    """

    def load(self) -> ModelCatalog:
        """Return a validated ``ModelCatalog``."""
        ...


class FileCatalogStore:
    """Read and atomically replace ``<cache>/models/catalog.json``.

    Does not touch corpus run history. Does not fetch the network.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_dir()
        self.models_dir = self.root / MODELS_DIRNAME
        self.catalog_path = self.models_dir / CATALOG_FILENAME
        self.backup_path = self.models_dir / CATALOG_BACKUP_FILENAME

    def exists(self) -> bool:
        return self.catalog_path.is_file()

    def load(self) -> ModelCatalog:
        """Return the overlay catalog. Raises if missing or unreadable."""
        if not self.exists():
            raise CatalogNotFoundError("no local model catalog")
        try:
            text = self.catalog_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise CatalogStoreError("model catalog is corrupt") from exc
        except OSError as exc:
            raise CatalogStoreError("model catalog could not be read") from exc
        try:
            return catalog_from_json(text)
        except CatalogSchemaError:
            raise
        except (CatalogDecodeError, CatalogValidationError) as exc:
            raise CatalogStoreError("model catalog is corrupt") from exc

    def save(self, catalog: ModelCatalog) -> None:
        """Validate, serialize, then atomically replace ``catalog.json``.

        ``catalog.json`` is the only atomic correctness boundary. After a
        failed write/replace it remains the previous complete catalog.

        ``catalog.json.bak`` is a best-effort copy of that previous overlay,
        written only after the live file has been replaced. A backup failure
        does not undo or invalidate the committed catalog.
        """
        if not isinstance(catalog, ModelCatalog):
            raise CatalogStoreError("catalog is invalid")
        try:
            payload = catalog.to_dict()
            catalog_from_dict(payload)
        except (CatalogDecodeError, CatalogValidationError, TypeError, ValueError) as exc:
            raise CatalogStoreError("model catalog could not be serialized") from exc
        previous: bytes | None = None
        if self.catalog_path.is_file():
            with suppress(OSError):
                previous = self.catalog_path.read_bytes()
        try:
            atomic_write_json(self.catalog_path, payload)
        except OSError as exc:
            raise CatalogStoreError("model catalog could not be written") from exc
        if previous is not None:
            with suppress(OSError):
                atomic_write_bytes(self.backup_path, previous)
