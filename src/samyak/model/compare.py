"""Evidence-safe, read-only comparisons of two or three model records."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from samyak.model.facts import AvailabilityScope, ContextWindow, Fact, FactStatus, LifecycleState
from samyak.model.records import EvidenceDisposition, ModelRecord

_CURRENT = EvidenceDisposition.CURRENT_DOCUMENTED
_COMPARISON_FACTS = (
    "display_name",
    "lifecycle",
    "availability_scope",
    "api_access",
    "tool_calling",
    "structured_output",
    "input_modalities",
    "output_modalities",
    "context_window",
)


class ModelComparisonError(ValueError):
    """Raised when a requested comparison cannot be selected safely."""


@dataclass(frozen=True, slots=True)
class ComparisonClaim:
    """One unresolved fact claim, retaining its original provenance."""

    value: Any
    source_id: str
    verified_at: str | None


@dataclass(frozen=True, slots=True)
class ComparisonFact:
    """A fact projection that never converts missing evidence into a value."""

    name: str
    status: FactStatus
    value: Any | None
    source_ids: tuple[str, ...]
    verified_at: tuple[str, ...]
    claims: tuple[ComparisonClaim, ...]


@dataclass(frozen=True, slots=True)
class ComparisonDimension:
    """Whether selected records have evidence suitable for direct comparison."""

    name: str
    comparable: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class ComparedModel:
    samyak_id: str
    provider_id: str
    documentation_disposition: EvidenceDisposition | None
    display_name: ComparisonFact
    lifecycle: ComparisonFact
    availability_scope: ComparisonFact
    api_access: ComparisonFact
    tool_calling: ComparisonFact
    structured_output: ComparisonFact
    input_modalities: ComparisonFact
    output_modalities: ComparisonFact
    context_window: ComparisonFact
    detail_href: str


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """Ordered compared records plus explicit cross-record comparability."""

    models: tuple[ComparedModel, ...]
    dimensions: tuple[ComparisonDimension, ...]
    warnings: tuple[str, ...]


def compare_models(
    records: Iterable[ModelRecord],
    model_ids: Sequence[str],
    *,
    include_legacy: bool = False,
    documentation: Iterable[EvidenceDisposition | None] | None = None,
) -> ModelComparison:
    """Compare exactly two or three explicitly selected stable model IDs.

    Selection is strict-current-documentation by default. ``documentation``
    adds named dispositions to that default, while ``include_legacy`` adds the
    legacy ``None`` disposition. This function only projects supplied records;
    it neither resolves aliases nor reads/writes storage.
    """
    requested = tuple(model_ids)
    if len(requested) not in {2, 3}:
        raise ModelComparisonError("compare exactly two or three model IDs")
    if len(set(requested)) != len(requested):
        raise ModelComparisonError("comparison model IDs must be unique")

    indexed: dict[str, ModelRecord] = {}
    for record in records:
        if record.samyak_id in indexed:
            raise ModelComparisonError("comparison records must have unique samyak_id values")
        indexed[record.samyak_id] = record
    unknown = tuple(model_id for model_id in requested if model_id not in indexed)
    if unknown:
        raise ModelComparisonError(f"unknown comparison model ID: {unknown[0]}")

    allowed: set[EvidenceDisposition | None] = {_CURRENT}
    if documentation is not None:
        allowed.update(documentation)
    if include_legacy:
        allowed.add(None)
    selected = tuple(indexed[model_id] for model_id in requested)
    ineligible = next(
        (record for record in selected if record.evidence_disposition not in allowed), None
    )
    if ineligible is not None:
        disposition = _disposition_name(ineligible.evidence_disposition)
        raise ModelComparisonError(
            f"{ineligible.samyak_id} has documentation disposition {disposition}; opt in explicitly"
        )

    compared = tuple(_compared_model(record) for record in selected)
    return ModelComparison(
        models=compared,
        dimensions=_dimensions(compared),
        warnings=_warnings(compared),
    )


def comparison_to_dict(comparison: ModelComparison) -> dict[str, object]:
    """Return a deterministic, JSON-compatible comparison projection."""
    return {
        "models": [_compared_model_dict(model) for model in comparison.models],
        "dimensions": [
            {"name": item.name, "comparable": item.comparable, "reason": item.reason}
            for item in comparison.dimensions
        ],
        "warnings": list(comparison.warnings),
    }


def _compared_model(record: ModelRecord) -> ComparedModel:
    facts = {name: _comparison_fact(name, getattr(record, name)) for name in _COMPARISON_FACTS}
    return ComparedModel(
        samyak_id=record.samyak_id,
        provider_id=record.provider_id,
        documentation_disposition=record.evidence_disposition,
        display_name=facts["display_name"],
        lifecycle=facts["lifecycle"],
        availability_scope=facts["availability_scope"],
        api_access=facts["api_access"],
        tool_calling=facts["tool_calling"],
        structured_output=facts["structured_output"],
        input_modalities=facts["input_modalities"],
        output_modalities=facts["output_modalities"],
        context_window=facts["context_window"],
        detail_href="/models/" + quote(record.samyak_id, safe=":@._-"),
    )


def _comparison_fact(name: str, fact: Fact[Any]) -> ComparisonFact:
    provenances = []
    if fact.provenance is not None:
        provenances.append(fact.provenance)
    provenances.extend(claim.provenance for claim in fact.claims)
    return ComparisonFact(
        name=name,
        status=fact.status,
        value=fact.value,
        source_ids=tuple(item.source_id for item in provenances),
        verified_at=tuple(item.verified_at for item in provenances if item.verified_at is not None),
        claims=tuple(
            ComparisonClaim(claim.value, claim.provenance.source_id, claim.provenance.verified_at)
            for claim in fact.claims
        ),
    )


def _dimensions(models: tuple[ComparedModel, ...]) -> tuple[ComparisonDimension, ...]:
    boolean = ("api_access", "tool_calling", "structured_output")
    result = [_all_known(name, tuple(getattr(model, name) for model in models)) for name in boolean]
    result.extend(
        _all_known(name, tuple(getattr(model, name) for model in models))
        for name in ("input_modalities", "output_modalities", "lifecycle")
    )
    result.append(_context_dimension(tuple(model.context_window for model in models)))
    result.append(_availability_dimension(tuple(model.availability_scope for model in models)))
    return tuple(result)


def _all_known(name: str, facts: tuple[ComparisonFact, ...]) -> ComparisonDimension:
    if all(item.status is FactStatus.KNOWN for item in facts):
        return ComparisonDimension(name, True, None)
    return ComparisonDimension(name, False, "not all selected records have verified evidence")


def _context_dimension(facts: tuple[ComparisonFact, ...]) -> ComparisonDimension:
    if not all(
        item.status is FactStatus.KNOWN and isinstance(item.value, ContextWindow) for item in facts
    ):
        return ComparisonDimension(
            "context_window", False, "not all selected records have context evidence"
        )
    kinds = {item.value.kind for item in facts if isinstance(item.value, ContextWindow)}
    if len(kinds) != 1 or next(iter(kinds)).value == "unknown":
        return ComparisonDimension(
            "context_window", False, "context kinds are not directly comparable"
        )
    return ComparisonDimension("context_window", True, None)


def _availability_dimension(facts: tuple[ComparisonFact, ...]) -> ComparisonDimension:
    if not all(item.status is FactStatus.KNOWN for item in facts):
        return ComparisonDimension(
            "availability_scope", False, "availability scope is not verified for all"
        )
    scopes = {item.value for item in facts}
    if len(scopes) != 1 or not all(isinstance(item, AvailabilityScope) for item in scopes):
        return ComparisonDimension("availability_scope", False, "availability scopes differ")
    return ComparisonDimension("availability_scope", True, None)


def _warnings(models: tuple[ComparedModel, ...]) -> tuple[str, ...]:
    dispositions = {model.documentation_disposition for model in models}
    if dispositions == {_CURRENT}:
        return ()
    labels = ", ".join(sorted(_disposition_name(item) for item in dispositions))
    return (f"comparison includes multiple documentation certification levels: {labels}",)


def _compared_model_dict(model: ComparedModel) -> dict[str, object]:
    return {
        "samyak_id": model.samyak_id,
        "provider_id": model.provider_id,
        "documentation_disposition": _disposition_name(model.documentation_disposition),
        "detail_href": model.detail_href,
        "facts": {name: _fact_dict(getattr(model, name)) for name in _COMPARISON_FACTS},
    }


def _fact_dict(fact: ComparisonFact) -> dict[str, object]:
    return {
        "name": fact.name,
        "status": fact.status.value,
        "value": _json_value(fact.value),
        "source_ids": list(fact.source_ids),
        "verified_at": list(fact.verified_at),
        "claims": [
            {
                "value": _json_value(item.value),
                "source_id": item.source_id,
                "verified_at": item.verified_at,
            }
            for item in fact.claims
        ],
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, ContextWindow):
        return value.to_dict()
    if isinstance(value, LifecycleState | AvailabilityScope):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _disposition_name(value: EvidenceDisposition | None) -> str:
    return "legacy_uncertified" if value is None else value.value
