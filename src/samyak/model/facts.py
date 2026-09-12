"""Typed facts, fact status, and fact-level provenance.

A fact is never truthy/falsy. Unknown is not false. Known false is an explicit
``status=known`` value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from samyak.model.errors import CatalogValidationError
from samyak.model.identity import ModelIdentity, validate_provider_id


class FactStatus(StrEnum):
    """How a fact's value should be interpreted."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    NOT_VERIFIED = "not_verified"
    CONFLICT = "conflict"


class SourceKind(StrEnum):
    """Class of authoritative source. Not a quality ranking of models."""

    PROVIDER_DOCS = "provider_docs"
    PROVIDER_API = "provider_api"
    PROVIDER_DEPRECATIONS = "provider_deprecations"
    PROVIDER_CHANGELOG = "provider_changelog"


class Confidence(StrEnum):
    """Confidence in the *source class*, not in whether a model is 'good'."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LifecycleState(StrEnum):
    """Canonical lifecycle.

    Unevaluated or unstated lifecycle is a fact status (``not_verified`` /
    ``unknown``), not a lifecycle value. There is no ``unknown`` state.

    ``legacy`` is a first-class provider lifecycle used when documentation
    names that state explicitly. It is not a fact status and must not be
    inferred from marketing lineup labels.
    """

    ACTIVE = "active"
    LEGACY = "legacy"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class ContextWindowKind(StrEnum):
    """What a provider's context-window number actually counts."""

    INPUT = "input"
    OUTPUT = "output"
    COMBINED = "combined"
    UNKNOWN = "unknown"


class Modality(StrEnum):
    """Canonical modality tokens. Adapters map provider words onto these."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class AvailabilityScope(StrEnum):
    """How widely a documented offering applies. Unknown is a fact status."""

    GLOBAL = "global"
    REGIONAL = "regional"
    ACCOUNT = "account"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a fact came from. Missing timestamps stay null; they are not invented."""

    provider: str
    source_id: str
    source_kind: SourceKind
    source_url: str | None = None
    content_hash: str | None = None
    retrieved_at: str | None = None
    observed_at: str | None = None
    verified_at: str | None = None
    confidence: Confidence | None = None

    def __post_init__(self) -> None:
        validate_provider_id(self.provider)
        _require_nonempty_str("source_id", self.source_id)
        if not isinstance(self.source_kind, SourceKind):
            raise CatalogValidationError("source_kind is not a known source kind")
        _optional_nonempty_str("source_url", self.source_url)
        _optional_nonempty_str("content_hash", self.content_hash)
        _optional_timestamp("retrieved_at", self.retrieved_at)
        _optional_timestamp("observed_at", self.observed_at)
        _optional_timestamp("verified_at", self.verified_at)
        if self.confidence is not None and not isinstance(self.confidence, Confidence):
            raise CatalogValidationError("confidence is not a known confidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "source_url": self.source_url,
            "content_hash": self.content_hash,
            "retrieved_at": self.retrieved_at,
            "observed_at": self.observed_at,
            "verified_at": self.verified_at,
            "confidence": None if self.confidence is None else self.confidence.value,
        }


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """A documented token window plus what that number measures."""

    tokens: int
    kind: ContextWindowKind

    def __post_init__(self) -> None:
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, int):
            raise CatalogValidationError("context_window.tokens must be an integer")
        if self.tokens <= 0:
            raise CatalogValidationError("context_window.tokens must be > 0")
        if not isinstance(self.kind, ContextWindowKind):
            raise CatalogValidationError("context_window.kind is not a known kind")

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "kind": self.kind.value}


@dataclass(frozen=True, slots=True)
class FactClaim[T]:
    """One source's asserted value for a field, with that source's provenance."""

    value: T
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.value is None:
            raise CatalogValidationError("conflict claims must have a value")
        if not isinstance(self.provenance, Provenance):
            raise CatalogValidationError("conflict claims must have provenance")


@dataclass(frozen=True, slots=True)
class Fact[T]:
    """A typed catalog fact with explicit status and optional provenance.

    ``value`` is set only when ``status`` is ``known``. Conflicts keep every
    competing value on ``claims``; they never drop values and keep provenance
    only. Boolean facts must not be interpreted with Python truthiness.
    """

    status: FactStatus
    value: T | None = None
    provenance: Provenance | None = None
    claims: tuple[FactClaim[T], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, FactStatus):
            raise CatalogValidationError("fact status is not a known status")
        if not isinstance(self.claims, tuple):
            raise CatalogValidationError("fact claims must be a tuple")
        if any(not isinstance(item, FactClaim) for item in self.claims):
            raise CatalogValidationError("fact claims must be FactClaim values")

        if self.status is FactStatus.KNOWN:
            if self.value is None:
                raise CatalogValidationError("known facts must have a value")
            if self.provenance is None:
                raise CatalogValidationError("known facts must have provenance")
            if self.claims:
                raise CatalogValidationError("known facts must not have competing claims")
            return

        if self.value is not None:
            raise CatalogValidationError(
                f"{self.status.value} facts must not have a value (unknown is not false)"
            )

        if self.status is FactStatus.NOT_VERIFIED:
            if self.provenance is not None or self.claims:
                raise CatalogValidationError("not_verified facts must not carry provenance")
            return

        if self.status is FactStatus.CONFLICT:
            if self.provenance is not None:
                raise CatalogValidationError("conflict facts store competing values on claims only")
            if len(self.claims) < 2:
                raise CatalogValidationError("conflict facts must have at least two claims")
            distinct: list[object] = []
            for claim in self.claims:
                if claim.value not in distinct:
                    distinct.append(claim.value)
            if len(distinct) < 2:
                raise CatalogValidationError(
                    "conflict facts must preserve at least two distinct competing values"
                )
            return

        if self.provenance is None:
            raise CatalogValidationError(f"{self.status.value} facts must have provenance")
        if self.claims:
            raise CatalogValidationError(
                f"{self.status.value} facts must not have competing claims"
            )

    def __bool__(self) -> bool:
        raise TypeError(
            "Fact cannot be used as a boolean; check status and value explicitly "
            "(unknown is not false)"
        )

    def is_known_true(self) -> bool:
        """Return True only for a known boolean True. Unknown is not True."""
        return self.status is FactStatus.KNOWN and self.value is True

    def is_known_false(self) -> bool:
        """Return True only for a known boolean False. Unknown is not False."""
        return self.status is FactStatus.KNOWN and self.value is False

    def to_dict(self, value_to_json: Any = None) -> dict[str, Any]:
        if self.status is FactStatus.KNOWN:
            encoded = self.value if value_to_json is None else value_to_json(self.value)
        else:
            encoded = None
        encoded_claims: list[dict[str, Any]] = []
        for claim in self.claims:
            claim_value = claim.value if value_to_json is None else value_to_json(claim.value)
            encoded_claims.append({"value": claim_value, "provenance": claim.provenance.to_dict()})
        return {
            "status": self.status.value,
            "value": encoded,
            "provenance": None if self.provenance is None else self.provenance.to_dict(),
            "claims": encoded_claims,
        }

    def provenance_source_ids(self) -> tuple[str, ...]:
        """Source ids attached to this fact, in first-seen order."""
        ids: list[str] = []
        seen: set[str] = set()
        if self.provenance is not None and self.provenance.source_id not in seen:
            ids.append(self.provenance.source_id)
            seen.add(self.provenance.source_id)
        for claim in self.claims:
            source_id = claim.provenance.source_id
            if source_id not in seen:
                ids.append(source_id)
                seen.add(source_id)
        return tuple(ids)


def unverified_fact() -> Fact[Any]:
    """Return a fact that this catalog has not evaluated."""
    return Fact(status=FactStatus.NOT_VERIFIED)


def validate_iso_date(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CatalogValidationError(f"{name} must be an ISO calendar date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise CatalogValidationError(f"{name} is not a valid ISO calendar date") from exc
    serialized = parsed.isoformat()
    if serialized != value:
        raise CatalogValidationError(f"{name} must be YYYY-MM-DD")
    return value


def validate_timestamp(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CatalogValidationError(f"{name} must be a timezone-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CatalogValidationError(f"{name} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CatalogValidationError(f"{name} must include a timezone offset")
    return value


def _require_nonempty_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CatalogValidationError(f"{name} must be a non-empty string")
    return value


def _optional_nonempty_str(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_nonempty_str(name, value)


def _optional_timestamp(name: str, value: object) -> str | None:
    if value is None:
        return None
    return validate_timestamp(name, value)


def identity_from_mapping(name: str, value: object) -> ModelIdentity:
    if not isinstance(value, dict):
        raise CatalogValidationError(f"{name} must be an object")
    expected = {"provider_id", "provider_model_id", "samyak_id"}
    extra = set(value) - expected
    if extra:
        raise CatalogValidationError(f"{name} has unexpected fields: {sorted(extra)}")
    identity = ModelIdentity(
        provider_id=_require_nonempty_str(f"{name}.provider_id", value.get("provider_id")),
        provider_model_id=_require_nonempty_str(
            f"{name}.provider_model_id", value.get("provider_model_id")
        ),
    )
    raw_id = value.get("samyak_id")
    if raw_id is not None and raw_id != identity.samyak_id:
        raise CatalogValidationError(
            f"{name}.samyak_id does not match provider_id/provider_model_id"
        )
    return identity
