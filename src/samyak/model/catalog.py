"""Versioned Model Intelligence catalog.

This is structured domain data, not a corpus findings report. Domain objects do
not read the filesystem or the network; see ``samyak.model.store.CatalogSource``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from samyak.__version__ import __version__
from samyak.model.errors import CatalogValidationError
from samyak.model.facts import FactStatus, validate_timestamp
from samyak.model.identity import IdentityKind, ModelIdentity
from samyak.model.records import ModelRecord

PRODUCT_NAME = "samyak"
CAPABILITY_NAME = "model"
CATALOG_SCHEMA_VERSION = 1
SUPPORTED_CATALOG_SCHEMA_VERSIONS = frozenset({CATALOG_SCHEMA_VERSION})


class FreshnessStatus(StrEnum):
    """Catalog as-of labeling. Does not mutate model lifecycle facts."""

    AS_OF = "as_of"
    STALE = "stale"
    UNKNOWN = "unknown"


class CatalogOverlay(StrEnum):
    """Where this snapshot came from. Filesystem details live outside the domain."""

    BUNDLED = "bundled"
    USER_CACHE = "user_cache"


@dataclass(frozen=True, slots=True)
class SourceRetrieval:
    """Last successful retrieve time for one named source."""

    source_id: str
    retrieved_at: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_id, str)
            or not self.source_id
            or self.source_id != self.source_id.strip()
        ):
            raise CatalogValidationError("source_id must be a non-empty string")
        validate_timestamp("retrieved_at", self.retrieved_at)

    def to_dict(self) -> dict[str, str]:
        return {"source_id": self.source_id, "retrieved_at": self.retrieved_at}


@dataclass(frozen=True, slots=True)
class CatalogFreshness:
    """How to label this catalog's age. Values are not inferred from the clock.

    ``stale_after``, when set, is the timezone-aware instant after which this
    snapshot should be labeled stale. It is not a duration string.
    """

    status: FreshnessStatus
    generated_at: str
    overlay: CatalogOverlay
    oldest_verified_at: str | None = None
    source_retrieved_at: tuple[SourceRetrieval, ...] = ()
    stale_after: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, FreshnessStatus):
            raise CatalogValidationError("freshness.status is not a known freshness status")
        validate_timestamp("freshness.generated_at", self.generated_at)
        if not isinstance(self.overlay, CatalogOverlay):
            raise CatalogValidationError("freshness.overlay is not a known overlay")
        if self.oldest_verified_at is not None:
            validate_timestamp("freshness.oldest_verified_at", self.oldest_verified_at)
        if not isinstance(self.source_retrieved_at, tuple):
            raise CatalogValidationError("freshness.source_retrieved_at must be a tuple")
        seen: set[str] = set()
        for item in self.source_retrieved_at:
            if not isinstance(item, SourceRetrieval):
                raise CatalogValidationError("freshness.source_retrieved_at values are invalid")
            if item.source_id in seen:
                raise CatalogValidationError(
                    "freshness.source_retrieved_at has duplicate source_id"
                )
            seen.add(item.source_id)
        if self.stale_after is not None:
            validate_timestamp("freshness.stale_after", self.stale_after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at,
            "overlay": self.overlay.value,
            "oldest_verified_at": self.oldest_verified_at,
            "source_retrieved_at": [item.to_dict() for item in self.source_retrieved_at],
            "stale_after": self.stale_after,
        }


@dataclass(frozen=True, slots=True)
class ProviderSummary:
    """Counts derived from catalog records. Not an independent source of truth."""

    id: str
    model_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "model_count": self.model_count}


@dataclass(frozen=True, slots=True)
class CatalogNotice:
    """Catalog-health note. Not a corpus Finding and not attached to each model."""

    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code or self.code != self.code.strip():
            raise CatalogValidationError("notice.code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise CatalogValidationError("notice.message must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """Immutable validated catalog snapshot."""

    product: str
    capability: str
    version: str
    catalog_schema_version: int
    generated_at: str
    freshness: CatalogFreshness
    providers: tuple[ProviderSummary, ...]
    models: tuple[ModelRecord, ...]
    notices: tuple[CatalogNotice, ...] = ()

    def __post_init__(self) -> None:
        validate_catalog(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "capability": self.capability,
            "version": self.version,
            "catalog_schema_version": self.catalog_schema_version,
            "generated_at": self.generated_at,
            "freshness": self.freshness.to_dict(),
            "providers": [item.to_dict() for item in self.providers],
            "models": [item.to_dict() for item in self.models],
            "notices": [item.to_dict() for item in self.notices],
        }


def build_catalog(
    *,
    models: Sequence[ModelRecord],
    generated_at: str,
    freshness: CatalogFreshness,
    version: str | None = None,
    notices: Sequence[CatalogNotice] = (),
) -> ModelCatalog:
    """Construct a validated catalog. Sorts models by ``samyak_id`` for stability."""
    ordered = tuple(sorted(models, key=lambda record: record.samyak_id))
    return ModelCatalog(
        product=PRODUCT_NAME,
        capability=CAPABILITY_NAME,
        version=__version__ if version is None else version,
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        generated_at=generated_at,
        freshness=freshness,
        providers=_provider_summaries(ordered),
        models=ordered,
        notices=tuple(notices),
    )


def validate_catalog(catalog: ModelCatalog) -> None:
    if catalog.product != PRODUCT_NAME:
        raise CatalogValidationError("product must be 'samyak'")
    if catalog.capability != CAPABILITY_NAME:
        raise CatalogValidationError("capability must be 'model'")
    if not isinstance(catalog.version, str) or not catalog.version:
        raise CatalogValidationError("version must be a non-empty string")
    if catalog.catalog_schema_version not in SUPPORTED_CATALOG_SCHEMA_VERSIONS:
        raise CatalogValidationError(
            f"unsupported catalog_schema_version {catalog.catalog_schema_version!r}"
        )
    validate_timestamp("generated_at", catalog.generated_at)
    if not isinstance(catalog.freshness, CatalogFreshness):
        raise CatalogValidationError("freshness is invalid")
    if catalog.freshness.generated_at != catalog.generated_at:
        raise CatalogValidationError("freshness.generated_at must match catalog generated_at")
    if not isinstance(catalog.models, tuple):
        raise CatalogValidationError("models must be a tuple")
    ids: list[str] = []
    for index, record in enumerate(catalog.models):
        if not isinstance(record, ModelRecord):
            raise CatalogValidationError(f"models[{index}] is not a ModelRecord")
        ids.append(record.samyak_id)
    if ids != sorted(ids):
        raise CatalogValidationError("models must be sorted by samyak_id")
    if len(ids) != len(set(ids)):
        raise CatalogValidationError("duplicate canonical identity")
    _validate_references(catalog.models)
    expected = _provider_summaries(catalog.models)
    if tuple(catalog.providers) != expected:
        raise CatalogValidationError("providers must match models")
    if not isinstance(catalog.notices, tuple):
        raise CatalogValidationError("notices must be a tuple")
    notice_codes: set[str] = set()
    for notice in catalog.notices:
        if not isinstance(notice, CatalogNotice):
            raise CatalogValidationError("notices contain an invalid entry")
        if notice.code in notice_codes:
            raise CatalogValidationError("notices must not contain duplicate codes")
        notice_codes.add(notice.code)


def _provider_summaries(models: tuple[ModelRecord, ...]) -> tuple[ProviderSummary, ...]:
    counts: dict[str, int] = {}
    for record in models:
        counts[record.provider_id] = counts.get(record.provider_id, 0) + 1
    return tuple(
        ProviderSummary(id=provider_id, model_count=counts[provider_id])
        for provider_id in sorted(counts)
    )


def _validate_references(models: tuple[ModelRecord, ...]) -> None:
    """Alias ``resolves_to`` targets must exist. Replacement ids must be well-formed.

    Dangling replacement identities are allowed: a successor may not be cataloged
    yet. ``ModelRecord`` already rejects malformed replacement identities.
    """
    present = {record.samyak_id: record for record in models}
    for record in models:
        if record.resolves_to.status is not FactStatus.KNOWN:
            continue
        target = record.resolves_to.value
        if not isinstance(target, ModelIdentity):
            raise CatalogValidationError(f"{record.samyak_id} resolves_to is invalid")
        if target.samyak_id not in present:
            raise CatalogValidationError(
                f"{record.samyak_id} resolves_to {target.samyak_id} is not in the catalog"
            )
        if (
            record.identity_kind is IdentityKind.ALIAS
            and present[target.samyak_id].identity_kind is IdentityKind.ALIAS
        ):
            raise CatalogValidationError(f"{record.samyak_id} must not resolve to another alias")
