"""Local run persistence for corpus analysis results.

Internal package — not part of the public Samyak API.

Architecture::

    AnalysisReport
        → RunStore
            → FileRunStore  (filesystem; this release)
            → future API / database / object-storage backends

The analysis engine does not depend on this package. Persistence is a
CLI/viewer concern so ``FileRunStore`` can later be replaced without
changing ``analyze_corpus``.

Retention (not implemented): ``samyak runs clear``, ``samyak runs delete``,
and TTL policies can be added later without changing the on-disk layout.
"""

from __future__ import annotations

from samyak.store.errors import (
    RunCorruptError,
    RunNotFoundError,
    RunSchemaError,
    RunStoreError,
)
from samyak.store.filesystem import FileRunStore
from samyak.store.models import RunMetadata, RunPage, StoredRun
from samyak.store.paths import default_cache_dir

__all__ = [
    "FileRunStore",
    "RunCorruptError",
    "RunMetadata",
    "RunNotFoundError",
    "RunPage",
    "RunSchemaError",
    "RunStoreError",
    "StoredRun",
    "default_cache_dir",
]
