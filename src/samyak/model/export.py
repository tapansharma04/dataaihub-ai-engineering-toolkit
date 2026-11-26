"""Read-only Model Intelligence export.

Wraps an existing catalog and optional lifecycle history. Does not change
catalog schema, history schema, or either file. Domain code only: no
filesystem and no network.

This module is part of v0.4.1, after the published v0.3.0 release. The consumer
contract is ``docs/model-export-contract.md``. Downstream importers validate
the JSON artifact and do not import this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from samyak.__version__ import __version__
from samyak.model.catalog import PRODUCT_NAME, CatalogNotice, ModelCatalog
from samyak.model.errors import CatalogValidationError
from samyak.model.facts import Fact, FactStatus
from samyak.model.historical_evidence import historical_specification_is_publishable
from samyak.model.history import LifecycleHistory
from samyak.model.records import RECORD_FACT_NAMES, ModelRecord
from samyak.model.update import (
    _FAILED_SOURCE_PREFIXES,
    _PARTIAL_NOTICE_PREFIX,
    SUPPORTED_PROVIDERS,
    _notice_provider,
    is_history_write_failure_code,
)

EXPORT_CAPABILITY_NAME = "model_export"
EXPORT_SCHEMA_VERSION = 1


class PublicationStatus(StrEnum):
    """Whether this snapshot may be published as a complete catalog."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class PublicationIntegrityStatus(StrEnum):
    """Whether the snapshot may be published without accepting bad evidence."""

    PUBLISHABLE = "publishable"
    BLOCKED = "blocked"


class PublicationCoverageStatus(StrEnum):
    """Whether every optional documentation source was available."""

    COMPLETE = "complete"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class ProviderCoverage:
    """One serving source's presence in this snapshot."""

    provider_id: str
    present: bool
    refreshed_with_snapshot: bool
    optional_source_failures: tuple[str, ...]
    coverage_notices: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "present": self.present,
            "refreshed_with_snapshot": self.refreshed_with_snapshot,
            "optional_source_failures": list(self.optional_source_failures),
            "coverage_notices": list(self.coverage_notices),
        }


@dataclass(frozen=True, slots=True)
class ExportPublication:
    """Legacy completeness plus explicit integrity and coverage decisions."""

    status: PublicationStatus
    reasons: tuple[str, ...]
    integrity_status: PublicationIntegrityStatus
    coverage_status: PublicationCoverageStatus
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reasons": list(self.reasons),
            "integrity_status": self.integrity_status.value,
            "coverage_status": self.coverage_status.value,
            "blocking_reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ModelExport:
    """Versioned wrapper around a catalog snapshot and optional history log."""

    catalog: ModelCatalog
    history: LifecycleHistory | None
    coverage: tuple[ProviderCoverage, ...]
    publication: ExportPublication
    exporter_version: str = __version__
    product: str = PRODUCT_NAME
    capability: str = EXPORT_CAPABILITY_NAME
    export_schema_version: int = EXPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, ModelCatalog):
            raise CatalogValidationError("export catalog must be a ModelCatalog")
        if self.history is not None and not isinstance(self.history, LifecycleHistory):
            raise CatalogValidationError("export history must be a LifecycleHistory or omitted")
        if self.product != PRODUCT_NAME:
            raise CatalogValidationError("product must be 'samyak'")
        if self.capability != EXPORT_CAPABILITY_NAME:
            raise CatalogValidationError("capability must be 'model_export'")
        if self.export_schema_version != EXPORT_SCHEMA_VERSION:
            raise CatalogValidationError(
                f"unsupported export_schema_version {self.export_schema_version!r}"
            )
        if not isinstance(self.exporter_version, str) or not self.exporter_version:
            raise CatalogValidationError("exporter_version must be a non-empty string")
        if not isinstance(self.publication, ExportPublication):
            raise CatalogValidationError("publication is invalid")
        if not isinstance(self.publication.status, PublicationStatus):
            raise CatalogValidationError("publication.status is not a known publication status")
        if not isinstance(self.publication.integrity_status, PublicationIntegrityStatus):
            raise CatalogValidationError("publication.integrity_status is not a known status")
        if not isinstance(self.publication.coverage_status, PublicationCoverageStatus):
            raise CatalogValidationError("publication.coverage_status is not a known status")
        complete = self.publication.status is PublicationStatus.COMPLETE
        if complete and self.publication.reasons:
            raise CatalogValidationError("a complete export must not include publication reasons")
        if not complete and not self.publication.reasons:
            raise CatalogValidationError("an incomplete export must include publication reasons")
        if tuple(sorted(self.publication.blocking_reasons + self.publication.warnings)) != (
            self.publication.reasons
        ):
            raise CatalogValidationError(
                "publication reasons must be classified as blocking or warnings"
            )
        expected_warnings = tuple(
            reason
            for reason in self.publication.reasons
            if reason.startswith("optional_source_failed:")
        )
        if self.publication.warnings != expected_warnings:
            raise CatalogValidationError("publication warnings must be optional-source failures")
        expected_blocking = tuple(
            reason for reason in self.publication.reasons if reason not in expected_warnings
        )
        if self.publication.blocking_reasons != expected_blocking:
            raise CatalogValidationError("publication blocking reasons are invalid")
        if self.publication.integrity_status is PublicationIntegrityStatus.PUBLISHABLE:
            if self.publication.blocking_reasons:
                raise CatalogValidationError(
                    "a publishable export must not include blocking reasons"
                )
        elif not self.publication.blocking_reasons:
            raise CatalogValidationError("a blocked export must include blocking reasons")
        if self.publication.coverage_status is PublicationCoverageStatus.COMPLETE:
            if self.publication.warnings:
                raise CatalogValidationError("complete coverage must not include warnings")
        elif not self.publication.warnings:
            raise CatalogValidationError("degraded coverage must include warnings")
        provider_ids = tuple(item.provider_id for item in self.coverage)
        if provider_ids[: len(SUPPORTED_PROVIDERS)] != SUPPORTED_PROVIDERS:
            raise CatalogValidationError("coverage must list every supported provider")

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "capability": self.capability,
            "export_schema_version": self.export_schema_version,
            "exporter_version": self.exporter_version,
            "publication": self.publication.to_dict(),
            "coverage": [item.to_dict() for item in self.coverage],
            "catalog": self.catalog.to_dict(),
            "history": None if self.history is None else self.history.to_dict(),
        }


def build_model_export(
    catalog: ModelCatalog,
    history: LifecycleHistory | None,
) -> ModelExport:
    """Build a publication-gated export. Does not read or write files.

    ``history`` is ``None`` when the lifecycle-history file is absent. A
    loaded history, including an empty log, is embedded as-is.
    """
    if not isinstance(catalog, ModelCatalog):
        raise CatalogValidationError("export catalog must be a ModelCatalog")
    if history is not None and not isinstance(history, LifecycleHistory):
        raise CatalogValidationError("export history must be a LifecycleHistory or omitted")

    records_by_provider: dict[str, tuple[ModelRecord, ...]] = {}
    for record in catalog.models:
        records_by_provider.setdefault(record.provider_id, ())
        records_by_provider[record.provider_id] = records_by_provider[record.provider_id] + (
            record,
        )

    failures_by_provider, notices_by_provider, failure_codes = _classify_notices(catalog.notices)
    extra_providers = tuple(
        sorted(
            provider_id
            for provider_id in records_by_provider
            if provider_id not in SUPPORTED_PROVIDERS
        )
    )
    ordered_providers = SUPPORTED_PROVIDERS + extra_providers

    coverage: list[ProviderCoverage] = []
    reasons: list[str] = []
    generated_at = catalog.freshness.generated_at
    for provider_id in ordered_providers:
        records = records_by_provider.get(provider_id, ())
        present = bool(records)
        refreshed = present and _refreshed_with_snapshot(records, generated_at)
        failures = tuple(sorted(failures_by_provider.get(provider_id, ())))
        notices = tuple(sorted(notices_by_provider.get(provider_id, ())))
        coverage.append(
            ProviderCoverage(
                provider_id=provider_id,
                present=present,
                refreshed_with_snapshot=refreshed,
                optional_source_failures=failures,
                coverage_notices=notices,
            )
        )
        if provider_id in SUPPORTED_PROVIDERS and not present:
            reasons.append(f"provider_missing:{provider_id}")
        if present and not refreshed:
            reasons.append(f"facts_not_from_this_snapshot:{provider_id}")

    reasons.extend(f"optional_source_failed:{code}" for code in failure_codes)
    if any(is_history_write_failure_code(notice.code) for notice in catalog.notices):
        reasons.append("history_write_failed")
    unique_reasons = tuple(sorted(set(reasons)))
    warnings = tuple(
        reason for reason in unique_reasons if reason.startswith("optional_source_failed:")
    )
    blocking_reasons = tuple(reason for reason in unique_reasons if reason not in warnings)
    status = PublicationStatus.COMPLETE if not unique_reasons else PublicationStatus.INCOMPLETE
    return ModelExport(
        catalog=catalog,
        history=history,
        coverage=tuple(coverage),
        publication=ExportPublication(
            status=status,
            reasons=unique_reasons,
            integrity_status=(
                PublicationIntegrityStatus.PUBLISHABLE
                if not blocking_reasons
                else PublicationIntegrityStatus.BLOCKED
            ),
            coverage_status=(
                PublicationCoverageStatus.COMPLETE
                if not warnings
                else PublicationCoverageStatus.DEGRADED
            ),
            blocking_reasons=blocking_reasons,
            warnings=warnings,
        ),
        exporter_version=__version__,
    )


def model_export_to_json(export: ModelExport) -> str:
    """Serialize an export with stable key order and a trailing newline."""
    if not isinstance(export, ModelExport):
        raise CatalogValidationError("export must be a ModelExport")
    return json.dumps(export.to_dict(), indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def _classify_notices(
    notices: tuple[CatalogNotice, ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Split notices into optional-source failures and other coverage notes.

    ``DOCUMENTED_SUBSET:`` is a coverage limitation, including the Fireworks
    notice emitted on every successful refresh. It is not an optional-source
    failure. Failure codes follow the catalog's existing notice prefixes.
    """
    failures: dict[str, list[str]] = {}
    coverage: dict[str, list[str]] = {}
    failure_codes: list[str] = []
    for notice in notices:
        code = notice.code
        provider_id = _notice_provider(code)
        if _is_optional_source_failure(code):
            failure_codes.append(code)
            if provider_id is not None:
                failures.setdefault(provider_id, []).append(code)
            continue
        if provider_id is not None:
            coverage.setdefault(provider_id, []).append(code)
    return (
        {key: tuple(value) for key, value in failures.items()},
        {key: tuple(value) for key, value in coverage.items()},
        tuple(failure_codes),
    )


def _is_optional_source_failure(code: str) -> bool:
    if code == _PARTIAL_NOTICE_PREFIX or code.startswith(f"{_PARTIAL_NOTICE_PREFIX}:"):
        return True
    return any(code.startswith(prefix) for prefix in _FAILED_SOURCE_PREFIXES)


def _refreshed_with_snapshot(records: tuple[ModelRecord, ...], generated_at: str) -> bool:
    """True when every provenanced fact was verified at this catalog's stamp.

    ``not_verified`` facts have no provenance and are not evidence of an
    earlier snapshot. A missing ``verified_at`` does not match.
    """
    for record in records:
        for name in RECORD_FACT_NAMES:
            fact = getattr(record, name)
            if not isinstance(fact, Fact):
                continue
            for verified_at in _fact_verified_timestamps(fact):
                if verified_at != generated_at and not historical_specification_is_publishable(
                    record,
                    name,
                    fact,
                    generated_at=generated_at,
                ):
                    return False
    return True


def _fact_verified_timestamps(fact: Fact[Any]) -> tuple[str | None, ...]:
    if fact.status is FactStatus.NOT_VERIFIED:
        return ()
    if fact.status is FactStatus.CONFLICT:
        return tuple(claim.provenance.verified_at for claim in fact.claims)
    if fact.provenance is None:
        return ()
    return (fact.provenance.verified_at,)
