"""Canonical model records: identity plus typed facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from samyak.model.errors import CatalogValidationError
from samyak.model.facts import (
    AvailabilityScope,
    ContextWindow,
    Fact,
    FactStatus,
    LifecycleState,
    Modality,
    identity_from_mapping,
    unverified_fact,
    validate_iso_date,
)
from samyak.model.identity import IdentityKind, ModelIdentity

_UNVERIFIED = unverified_fact

RECORD_FACT_NAMES: tuple[str, ...] = (
    "display_name",
    "aliases",
    "resolves_to",
    "family",
    "lifecycle",
    "deprecated_at",
    "retirement_at",
    "replacement",
    "context_window",
    "max_input_tokens",
    "max_output_tokens",
    "input_modalities",
    "output_modalities",
    "tool_calling",
    "structured_output",
    "api_access",
    "availability_scope",
    "regions",
)


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """One cataloged model or alias. Facts default to not_verified, not false."""

    identity: ModelIdentity
    identity_kind: IdentityKind = IdentityKind.CANONICAL
    display_name: Fact[str] = field(default_factory=_UNVERIFIED)
    aliases: Fact[tuple[str, ...]] = field(default_factory=_UNVERIFIED)
    resolves_to: Fact[ModelIdentity] = field(default_factory=_UNVERIFIED)
    family: Fact[str] = field(default_factory=_UNVERIFIED)
    lifecycle: Fact[LifecycleState] = field(default_factory=_UNVERIFIED)
    deprecated_at: Fact[str] = field(default_factory=_UNVERIFIED)
    retirement_at: Fact[str] = field(default_factory=_UNVERIFIED)
    replacement: Fact[tuple[ModelIdentity, ...]] = field(default_factory=_UNVERIFIED)
    context_window: Fact[ContextWindow] = field(default_factory=_UNVERIFIED)
    max_input_tokens: Fact[int] = field(default_factory=_UNVERIFIED)
    max_output_tokens: Fact[int] = field(default_factory=_UNVERIFIED)
    input_modalities: Fact[tuple[Modality, ...]] = field(default_factory=_UNVERIFIED)
    output_modalities: Fact[tuple[Modality, ...]] = field(default_factory=_UNVERIFIED)
    tool_calling: Fact[bool] = field(default_factory=_UNVERIFIED)
    structured_output: Fact[bool] = field(default_factory=_UNVERIFIED)
    api_access: Fact[bool] = field(default_factory=_UNVERIFIED)
    availability_scope: Fact[AvailabilityScope] = field(default_factory=_UNVERIFIED)
    regions: Fact[tuple[str, ...]] = field(default_factory=_UNVERIFIED)

    def __post_init__(self) -> None:
        validate_model_record(self)

    @property
    def provider_id(self) -> str:
        return self.identity.provider_id

    @property
    def provider_model_id(self) -> str:
        return self.identity.provider_model_id

    @property
    def samyak_id(self) -> str:
        return self.identity.samyak_id

    @property
    def observed_sources(self) -> tuple[str, ...]:
        """Source ids derived from fact-level provenance, first-seen order.

        Not a separately stored list. JSON still emits this field; load rejects
        a list that does not match the derived value.
        """
        return derived_observed_sources(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "samyak_id": self.samyak_id,
            "provider_id": self.provider_id,
            "provider_model_id": self.provider_model_id,
            "identity_kind": self.identity_kind.value,
            "display_name": self.display_name.to_dict(),
            "aliases": self.aliases.to_dict(list),
            "resolves_to": self.resolves_to.to_dict(
                lambda value: value.to_dict() if value is not None else None
            ),
            "family": self.family.to_dict(),
            "lifecycle": self.lifecycle.to_dict(
                lambda value: value.value if isinstance(value, LifecycleState) else value
            ),
            "deprecated_at": self.deprecated_at.to_dict(),
            "retirement_at": self.retirement_at.to_dict(),
            "replacement": self.replacement.to_dict(
                lambda value: [item.to_dict() for item in value]
            ),
            "context_window": self.context_window.to_dict(
                lambda value: value.to_dict() if value is not None else None
            ),
            "max_input_tokens": self.max_input_tokens.to_dict(),
            "max_output_tokens": self.max_output_tokens.to_dict(),
            "input_modalities": self.input_modalities.to_dict(
                lambda value: [item.value for item in value]
            ),
            "output_modalities": self.output_modalities.to_dict(
                lambda value: [item.value for item in value]
            ),
            "tool_calling": self.tool_calling.to_dict(),
            "structured_output": self.structured_output.to_dict(),
            "api_access": self.api_access.to_dict(),
            "availability_scope": self.availability_scope.to_dict(
                lambda value: value.value if isinstance(value, AvailabilityScope) else value
            ),
            "regions": self.regions.to_dict(list),
            "observed_sources": list(self.observed_sources),
        }


def validate_model_record(record: ModelRecord) -> None:
    if not isinstance(record.identity, ModelIdentity):
        raise CatalogValidationError("identity must be a ModelIdentity")
    if not isinstance(record.identity_kind, IdentityKind):
        raise CatalogValidationError("identity_kind is not a known identity kind")

    _typed_fact("display_name", record.display_name, _require_str)
    _typed_fact("aliases", record.aliases, _require_str_tuple)
    _typed_fact("resolves_to", record.resolves_to, _require_identity)
    _typed_fact("family", record.family, _require_str)
    _typed_fact("lifecycle", record.lifecycle, _require_lifecycle)
    _typed_fact("deprecated_at", record.deprecated_at, lambda n, v: validate_iso_date(n, v))
    _typed_fact("retirement_at", record.retirement_at, lambda n, v: validate_iso_date(n, v))
    _typed_fact("replacement", record.replacement, _require_identity_tuple)
    _typed_fact("context_window", record.context_window, _require_context_window)
    _typed_fact("max_input_tokens", record.max_input_tokens, _require_positive_int)
    _typed_fact("max_output_tokens", record.max_output_tokens, _require_positive_int)
    _typed_fact("input_modalities", record.input_modalities, _require_modalities)
    _typed_fact("output_modalities", record.output_modalities, _require_modalities)
    _typed_fact("tool_calling", record.tool_calling, _require_bool)
    _typed_fact("structured_output", record.structured_output, _require_bool)
    _typed_fact("api_access", record.api_access, _require_bool)
    _typed_fact("availability_scope", record.availability_scope, _require_scope)
    _typed_fact("regions", record.regions, _require_str_tuple)

    if (
        record.identity_kind is IdentityKind.CANONICAL
        and record.resolves_to.status is FactStatus.KNOWN
    ):
        raise CatalogValidationError("canonical records must not resolve to another model")
    if record.replacement.status is FactStatus.KNOWN and record.identity in (
        record.replacement.value or ()
    ):
        raise CatalogValidationError("replacement must not include the model itself")
    _validate_lifecycle_dates(record)


def _typed_fact(name: str, fact: object, check: Any) -> None:
    if not isinstance(fact, Fact):
        raise CatalogValidationError(f"{name} must be a Fact")
    if fact.status is FactStatus.KNOWN:
        check(name, fact.value)
        return
    if fact.status is FactStatus.CONFLICT:
        for index, claim in enumerate(fact.claims):
            check(f"{name}.claims[{index}].value", claim.value)


def derived_observed_sources(record: ModelRecord) -> tuple[str, ...]:
    """Collect provenance source ids from every fact, first-seen, field order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for name in RECORD_FACT_NAMES:
        fact = getattr(record, name)
        if not isinstance(fact, Fact):
            continue
        for source_id in fact.provenance_source_ids():
            if source_id not in seen:
                seen.add(source_id)
                ordered.append(source_id)
    return tuple(ordered)


def _validate_lifecycle_dates(record: ModelRecord) -> None:
    """Reject a known retirement date earlier than a known deprecation date.

    Comparison runs only when both dates are ``known``. Missing, unknown,
    not_verified, or conflicting dates are left alone so adapters are not
    forced to invent a date the provider did not publish.
    """
    if (
        record.deprecated_at.status is not FactStatus.KNOWN
        or record.retirement_at.status is not FactStatus.KNOWN
    ):
        return
    deprecated = record.deprecated_at.value
    retired = record.retirement_at.value
    if not isinstance(deprecated, str) or not isinstance(retired, str):
        raise CatalogValidationError("lifecycle dates must be strings when known")
    if date.fromisoformat(retired) < date.fromisoformat(deprecated):
        raise CatalogValidationError("retirement_at must not be earlier than deprecated_at")


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CatalogValidationError(f"{name} must be a non-empty string")
    return value


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise CatalogValidationError(f"{name} must be a boolean when known")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CatalogValidationError(f"{name} must be an integer when known")
    if value <= 0:
        raise CatalogValidationError(f"{name} must be > 0")
    return value


def _require_lifecycle(name: str, value: object) -> LifecycleState:
    if not isinstance(value, LifecycleState):
        raise CatalogValidationError(f"{name} is not a known lifecycle state")
    return value


def _require_scope(name: str, value: object) -> AvailabilityScope:
    if not isinstance(value, AvailabilityScope):
        raise CatalogValidationError(f"{name} is not a known availability scope")
    return value


def _require_context_window(name: str, value: object) -> ContextWindow:
    if not isinstance(value, ContextWindow):
        raise CatalogValidationError(f"{name} must be a ContextWindow when known")
    return value


def _require_identity(name: str, value: object) -> ModelIdentity:
    if not isinstance(value, ModelIdentity):
        raise CatalogValidationError(f"{name} must be a ModelIdentity when known")
    return value


def _require_identity_tuple(name: str, value: object) -> tuple[ModelIdentity, ...]:
    if not isinstance(value, tuple) or not value:
        raise CatalogValidationError(f"{name} must be a non-empty tuple when known")
    if any(not isinstance(item, ModelIdentity) for item in value):
        raise CatalogValidationError(f"{name} values must be ModelIdentity")
    if len(set(item.samyak_id for item in value)) != len(value):
        raise CatalogValidationError(f"{name} must not contain duplicate identities")
    return value


def _require_str_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise CatalogValidationError(f"{name} must be a non-empty tuple when known")
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item or item != item.strip():
            raise CatalogValidationError(f"{name}[{index}] must be a non-empty string")
        if item in seen:
            raise CatalogValidationError(f"{name} must not contain duplicates")
        seen.add(item)
    return value


def _require_modalities(name: str, value: object) -> tuple[Modality, ...]:
    if not isinstance(value, tuple) or not value:
        raise CatalogValidationError(f"{name} must be a non-empty tuple when known")
    seen: set[Modality] = set()
    for item in value:
        if not isinstance(item, Modality):
            raise CatalogValidationError(f"{name} contains an unknown modality")
        if item in seen:
            raise CatalogValidationError(f"{name} must not contain duplicates")
        seen.add(item)
    return value


def replacement_identities_from_json(name: str, value: object) -> tuple[ModelIdentity, ...]:
    if not isinstance(value, list) or not value:
        raise CatalogValidationError(f"{name} must be a non-empty array when known")
    return tuple(
        identity_from_mapping(f"{name}[{index}]", item) for index, item in enumerate(value)
    )
