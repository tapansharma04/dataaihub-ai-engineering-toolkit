"""Application Readiness domain types.

These are not corpus ``Finding`` / ``AnalysisReport`` types. Findings may
repeat a ``code``. Line numbers are omitted when the extractor does not have
exact lines — never invented as line 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

PRODUCT_NAME = "samyak"
CAPABILITY_NAME = "app"
REPORT_SCHEMA_VERSION = 2

NOTICE_ANALYSIS_TRUNCATED = "ANALYSIS_TRUNCATED"
NOTICE_FILE_SKIPPED_TOO_LARGE = "FILE_SKIPPED_TOO_LARGE"
NOTICE_FILE_SKIPPED_BINARY = "FILE_SKIPPED_BINARY"
NOTICE_FILE_PARSE_FAILED = "FILE_PARSE_FAILED"
NOTICE_CATALOG_ABSENT = "CATALOG_ABSENT"
NOTICE_CATALOG_STALE = "CATALOG_STALE"
NOTICE_CATALOG_UNREADABLE = "CATALOG_UNREADABLE"
NOTICE_CATALOG_SCHEMA = "CATALOG_SCHEMA"
NOTICE_PROVIDER_NOT_IN_CATALOG = "PROVIDER_NOT_IN_CATALOG"
NOTICE_MODEL_REFERENCE_UNCHECKED = "MODEL_REFERENCE_UNCHECKED"
NOTICE_MODEL_CATALOG_CANDIDATE = "MODEL_CATALOG_CANDIDATE"
NOTICE_MODEL_LIFECYCLE_CONFLICT = "MODEL_LIFECYCLE_CONFLICT"

FINDING_MODEL_REFERENCE_DEPRECATED = "MODEL_REFERENCE_DEPRECATED"
FINDING_MODEL_REFERENCE_RETIRED = "MODEL_REFERENCE_RETIRED"
FINDING_MODEL_REFERENCE_UNKNOWN = "MODEL_REFERENCE_UNKNOWN"
FINDING_MODEL_PROVIDER_AMBIGUOUS = "MODEL_PROVIDER_AMBIGUOUS"
CATEGORY_MODEL = "model"


class Severity(StrEnum):
    """Finding impact if the finding is true. Not detector confidence."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Confidence(StrEnum):
    """Detector certainty. Not model quality and not catalog source-class confidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MatchKind(StrEnum):
    """How a model reference related to the local catalog. Not a lifecycle value."""

    EXACT_IDENTITY = "exact_identity"
    MODEL_ID_UNIQUE = "model_id_unique"
    MODEL_ID_AMBIGUOUS = "model_id_ambiguous"
    NOT_IN_CATALOG = "not_in_catalog"
    PROVIDER_NOT_IN_CATALOG = "provider_not_in_catalog"
    CATALOG_ABSENT = "catalog_absent"
    CATALOG_UNREADABLE = "catalog_unreadable"


class EvidenceType(StrEnum):
    """How a source value was observed.

    ``python_import`` is observation-only (provider SDK imports). Finding
    evidence in later phases uses the architecture's first-slice set.
    """

    PYTHON_LITERAL = "python_literal"
    PYTHON_KEYWORD = "python_keyword"
    PYTHON_IMPORT = "python_import"
    PYTHON_CONSTANT = "python_constant"
    JSON_VALUE = "json_value"
    TOML_VALUE = "toml_value"


class ApplicationIdentityState(StrEnum):
    """Application-plane identity. Not Model Intelligence FactStatus.

    ``unknown`` is reserved for catalog-unknown findings. Bound literals that
    are not established, candidate, or ambiguous are ``BOUND`` internally and
    serialize as ``resolved``.
    """

    ESTABLISHED = "established"
    CANDIDATE = "candidate"
    AMBIGUOUS = "ambiguous"
    BOUND = "bound"
    UNRESOLVED = "unresolved"
    DYNAMIC = "dynamic"


def public_identity_value(state: ApplicationIdentityState) -> str:
    """User-facing identity vocabulary. ``BOUND`` is presented as ``resolved``."""
    if state is ApplicationIdentityState.BOUND:
        return "resolved"
    return state.value


class AiOperationKind(StrEnum):
    """Closed AI-operation kinds. First capability: model inference only."""

    MODEL_INFERENCE = "model_inference"


class ResolutionMethod(StrEnum):
    """How a bound model id was obtained from source. Not catalog resolution."""

    SAME_FILE_CONSTANT = "same_file_constant"


class UnboundReason(StrEnum):
    """Why a model-shaped slot did not yield a bound model id."""

    NAME_NOT_BOUND = "name_not_bound"
    UNRESOLVED_ATTRIBUTE = "unresolved_attribute"
    UNRESOLVED_EXPRESSION = "unresolved_expression"
    DYNAMIC_EXPRESSION = "dynamic_expression"
    DYNAMIC_CALL = "dynamic_call"
    DYNAMIC_SUBSCRIPT = "dynamic_subscript"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Deterministic first-pass bounds. Not a user questionnaire."""

    max_file_bytes: int = 1_048_576
    nul_probe_bytes: int = 8_192
    max_analyzed_files: int = 10_000
    max_notices_per_code: int = 50

    def __post_init__(self) -> None:
        for name in (
            "max_file_bytes",
            "nul_probe_bytes",
            "max_analyzed_files",
            "max_notices_per_code",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_file_bytes": self.max_file_bytes,
            "nul_probe_bytes": self.nul_probe_bytes,
            "max_analyzed_files": self.max_analyzed_files,
            "max_notices_per_code": self.max_notices_per_code,
        }


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Project-relative evidence pointer. JSON/TOML lines stay unset when unknown."""

    path: str
    evidence_type: EvidenceType
    detected_value: str
    line_start: int | None = None
    line_end: int | None = None
    symbol: str | None = None
    config_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "symbol": self.symbol,
            "config_key": self.config_key,
            "detected_value": self.detected_value,
            "evidence_type": self.evidence_type.value,
        }


@dataclass(frozen=True, slots=True)
class CatalogEvidence:
    """Catalog consultation for a finding. Distinct from source evidence.

    ``samyak_id`` is set only for source-established exact matches.
    ``provider_id`` is copied from source evidence when established, never from
    overlay uniqueness. ``lifecycle_status`` is omitted on candidate, unknown,
    and ambiguous rows. Catalog ``fact_confidence`` is source-class confidence,
    not detector ``Confidence``.
    """

    match_kind: MatchKind
    provider_model_id: str
    samyak_id: str | None = None
    provider_id: str | None = None
    identity_kind: str | None = None
    lifecycle_status: str | None = None
    lifecycle_value: str | None = None
    catalog_freshness: dict[str, str] | None = None
    candidate_samyak_ids: tuple[str, ...] = ()
    replacement_samyak_ids: tuple[str, ...] = ()
    fact_confidence: str | None = None
    source_id: str | None = None
    source_url: str | None = None
    resolves_to_samyak_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "match_kind": self.match_kind.value,
            "provider_model_id": self.provider_model_id,
        }
        if self.samyak_id is not None:
            payload["samyak_id"] = self.samyak_id
        if self.provider_id is not None:
            payload["provider_id"] = self.provider_id
        if self.identity_kind is not None:
            payload["identity_kind"] = self.identity_kind
        if self.lifecycle_status is not None:
            status: dict[str, str] = {"status": self.lifecycle_status}
            if self.lifecycle_value is not None:
                status["value"] = self.lifecycle_value
            payload["lifecycle_status"] = status
        if self.catalog_freshness is not None:
            payload["catalog_freshness"] = dict(self.catalog_freshness)
        if self.candidate_samyak_ids:
            payload["candidate_samyak_ids"] = list(self.candidate_samyak_ids)
        if self.replacement_samyak_ids:
            payload["replacement_samyak_ids"] = list(self.replacement_samyak_ids)
        if self.fact_confidence is not None:
            payload["fact_confidence"] = self.fact_confidence
        if self.source_id is not None:
            payload["source_id"] = self.source_id
        if self.source_url is not None:
            payload["source_url"] = self.source_url
        if self.resolves_to_samyak_id is not None:
            payload["resolves_to_samyak_id"] = self.resolves_to_samyak_id
        return payload


@dataclass(frozen=True, slots=True)
class CatalogContext:
    """Whether a local overlay was loaded. Not a copy of catalog records."""

    present: bool
    freshness_status: str | None = None
    generated_at: str | None = None
    overlay: str | None = None
    providers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if not self.present:
            return {"present": False}
        return {
            "present": True,
            "freshness": {
                "status": self.freshness_status,
                "generated_at": self.generated_at,
                "overlay": self.overlay,
            },
            "providers": list(self.providers),
        }


@dataclass(frozen=True, slots=True)
class AppNotice:
    """Analysis health note. Not a scored finding."""

    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True, slots=True)
class AppFinding:
    """One occurrence of a rule. Many findings may share ``code``."""

    code: str
    category: str
    severity: Severity
    confidence: Confidence
    title: str
    message: str
    why_it_matters: str
    recommendation: str
    evidence: SourceLocation
    catalog_evidence: CatalogEvidence | None = None
    resolution_source: SourceLocation | None = None
    reference_expression: str | None = None

    @property
    def instance_key(self) -> str:
        location = self.evidence
        line = "" if location.line_start is None else str(location.line_start)
        key = location.config_key or ""
        return f"{self.code}|{location.path}|{line}|{key}|{location.detected_value}"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "instance_key": self.instance_key,
            "category": self.category,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "title": self.title,
            "message": self.message,
            "why_it_matters": self.why_it_matters,
            "recommendation": self.recommendation,
            "evidence": self.evidence.to_dict(),
        }
        if self.catalog_evidence is not None:
            payload["catalog_evidence"] = self.catalog_evidence.to_dict()
        if self.resolution_source is not None:
            payload["resolution_source"] = self.resolution_source.to_dict()
        if self.reference_expression is not None:
            payload["reference_expression"] = self.reference_expression
        return payload


def finding_sort_key(finding: AppFinding) -> tuple[object, ...]:
    """Deterministic order: path, then line when present, then code."""
    location = finding.evidence
    return (
        location.path,
        location.line_start is None,
        location.line_start if location.line_start is not None else 0,
        finding.code,
        location.config_key or "",
        location.detected_value,
        finding.instance_key,
    )


def sort_findings(items: list[AppFinding] | tuple[AppFinding, ...]) -> tuple[AppFinding, ...]:
    return tuple(sorted(items, key=finding_sort_key))


_IDENTITY_SORT_ORDER = {
    ApplicationIdentityState.ESTABLISHED: 0,
    ApplicationIdentityState.CANDIDATE: 1,
    ApplicationIdentityState.AMBIGUOUS: 2,
    ApplicationIdentityState.BOUND: 3,
    ApplicationIdentityState.UNRESOLVED: 4,
    ApplicationIdentityState.DYNAMIC: 5,
}


@dataclass(frozen=True, slots=True)
class InventoryOperation:
    """One bound AI operation (use site) under a model dependency."""

    path: str
    evidence_type: EvidenceType
    detected_model_id: str
    line_start: int | None = None
    line_end: int | None = None
    slot: str | None = None
    symbol: str | None = None
    expression: str | None = None
    resolution_method: ResolutionMethod | None = None
    resolution_source: SourceLocation | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "slot": self.slot,
            "symbol": self.symbol,
            "detected_model_id": self.detected_model_id,
            "evidence_type": self.evidence_type.value,
        }
        if self.expression is not None:
            payload["expression"] = self.expression
        if self.resolution_method is not None:
            payload["resolution_method"] = self.resolution_method.value
        if self.resolution_source is not None:
            payload["resolution_source"] = self.resolution_source.to_dict()
        return payload


def inventory_operation_sort_key(item: InventoryOperation) -> tuple[object, ...]:
    return (
        item.path,
        item.line_start is None,
        item.line_start if item.line_start is not None else 0,
        item.line_end if item.line_end is not None else 0,
        item.slot or "",
        item.symbol or "",
        item.detected_model_id,
        item.evidence_type.value,
        item.expression or "",
        item.resolution_method.value if item.resolution_method is not None else "",
        "" if item.resolution_source is None else item.resolution_source.path,
        0
        if item.resolution_source is None or item.resolution_source.line_start is None
        else item.resolution_source.line_start,
    )


@dataclass(frozen=True, slots=True)
class ModelDependency:
    """Grouped bound model uses. Catalog facts are cited, not copied."""

    application_identity: ApplicationIdentityState
    detected_model_id: str
    operations: tuple[InventoryOperation, ...]
    provider_id: str | None = None
    samyak_id: str | None = None
    match_kind: MatchKind | None = None
    catalog_evidence: CatalogEvidence | None = None

    @property
    def identity_label(self) -> str:
        if self.samyak_id is not None:
            return self.samyak_id
        if self.provider_id is not None:
            return f"{self.provider_id}:{self.detected_model_id}"
        return self.detected_model_id

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "application_identity": public_identity_value(self.application_identity),
            "detected_model_id": self.detected_model_id,
            "provider_id": self.provider_id,
            "samyak_id": self.samyak_id,
            "match_kind": None if self.match_kind is None else self.match_kind.value,
            "operations": [item.to_dict() for item in self.operations],
        }
        if self.catalog_evidence is not None:
            payload["catalog_evidence"] = self.catalog_evidence.to_dict()
        else:
            payload["catalog_evidence"] = None
        return payload


def dependency_sort_key(item: ModelDependency) -> tuple[object, ...]:
    first = item.operations[0] if item.operations else None
    return (
        _IDENTITY_SORT_ORDER[item.application_identity],
        item.identity_label,
        item.detected_model_id,
        item.provider_id or "",
        item.samyak_id or "",
        first.path if first is not None else "",
        first.line_start if first is not None and first.line_start is not None else 0,
    )


@dataclass(frozen=True, slots=True)
class UnresolvedOperation:
    """AI operation whose model id was not statically bound. Not a dependency."""

    application_identity: ApplicationIdentityState
    unbound_reason: UnboundReason
    expression: str
    path: str
    evidence_type: EvidenceType
    line_start: int | None = None
    line_end: int | None = None
    slot: str | None = None
    symbol: str | None = None
    operation_kind: AiOperationKind = AiOperationKind.MODEL_INFERENCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_identity": public_identity_value(self.application_identity),
            "unbound_reason": self.unbound_reason.value,
            "expression": self.expression,
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "slot": self.slot,
            "symbol": self.symbol,
            "evidence_type": self.evidence_type.value,
        }


def unresolved_operation_sort_key(item: UnresolvedOperation) -> tuple[object, ...]:
    return (
        item.path,
        item.line_start is None,
        item.line_start if item.line_start is not None else 0,
        item.line_end if item.line_end is not None else 0,
        _IDENTITY_SORT_ORDER[item.application_identity],
        item.unbound_reason.value,
        item.expression,
        item.slot or "",
        item.symbol or "",
        item.evidence_type.value,
    )


@dataclass(frozen=True, slots=True)
class AppInventory:
    """First-class application model inventory. Not findings and not observations."""

    dependencies: tuple[ModelDependency, ...]
    unresolved_operations: tuple[UnresolvedOperation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependencies": [item.to_dict() for item in self.dependencies],
            "unresolved_operations": [item.to_dict() for item in self.unresolved_operations],
        }


def empty_inventory() -> AppInventory:
    return AppInventory(dependencies=(), unresolved_operations=())


@dataclass(frozen=True, slots=True)
class AppSummary:
    """Inventory of what was inspected. Not a readiness score."""

    project_root: str
    analyzed_files: int
    skipped_oversized: int
    skipped_binary: int
    parse_failures: int
    truncated: bool
    analyzed_by_extension: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "analyzed_files": self.analyzed_files,
            "skipped_oversized": self.skipped_oversized,
            "skipped_binary": self.skipped_binary,
            "parse_failures": self.parse_failures,
            "truncated": self.truncated,
            "analyzed_by_extension": dict(sorted(self.analyzed_by_extension.items())),
        }


@dataclass(frozen=True, slots=True)
class AppReport:
    """Inspection envelope. Catalog context records overlay presence, not model rows."""

    product: str
    capability: str
    version: str
    report_schema_version: int
    summary: AppSummary
    catalog: CatalogContext
    inventory: AppInventory
    notices: tuple[AppNotice, ...]
    findings: tuple[AppFinding, ...]
    config: dict[str, Any]

    def severity_counts(self) -> dict[str, int]:
        counts = {item.value: 0 for item in Severity}
        for finding in self.findings:
            counts[finding.severity.value] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "capability": self.capability,
            "version": self.version,
            "report_schema_version": self.report_schema_version,
            "summary": self.summary.to_dict(),
            "catalog": self.catalog.to_dict(),
            "inventory": self.inventory.to_dict(),
            "notices": [item.to_dict() for item in self.notices],
            "findings": [item.to_dict() for item in self.findings],
            "severity_counts": self.severity_counts(),
            "config": dict(self.config),
        }
