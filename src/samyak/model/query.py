"""Evidence-aware, read-only queries over typed model records.

This module intentionally has no catalog-store, archive, provider-parser, CLI,
or HTML dependency.  It only evaluates facts already present on immutable
``ModelRecord`` values; unknown, unverified, and conflicting facts are never
silently converted into negative capabilities.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from samyak.model.facts import ContextWindowKind, Fact, FactStatus, LifecycleState, Modality
from samyak.model.records import EvidenceDisposition, ModelRecord

_DEFAULT_DISPOSITIONS = frozenset({EvidenceDisposition.CURRENT_DOCUMENTED})
_CONTEXT_KINDS = frozenset(
    {ContextWindowKind.INPUT, ContextWindowKind.OUTPUT, ContextWindowKind.COMBINED}
)


@dataclass(frozen=True, slots=True)
class ModelQuery:
    """Typed, conjunctive predicates for an existing collection of records.

    ``documentation_dispositions`` is deliberately explicit.  The default is
    current documentation only; callers must opt in to legacy, historical, or
    not-observed identities and that opt-in does not make them current.
    """

    providers: frozenset[str] | None = None
    documentation_dispositions: frozenset[EvidenceDisposition | None] = _DEFAULT_DISPOSITIONS
    lifecycle: LifecycleState | None = None
    api_access: bool | None = None
    tool_calling: bool | None = None
    structured_output: bool | None = None
    input_modalities: frozenset[Modality] = frozenset()
    output_modalities: frozenset[Modality] = frozenset()
    minimum_context_window: int | None = None
    context_window_kind: ContextWindowKind | None = None

    def __post_init__(self) -> None:
        if self.providers is not None and not self.providers:
            raise ValueError("providers must be non-empty when supplied")
        if not self.documentation_dispositions:
            raise ValueError("documentation_dispositions must be non-empty")
        if self.minimum_context_window is not None:
            if isinstance(self.minimum_context_window, bool) or self.minimum_context_window <= 0:
                raise ValueError("minimum_context_window must be a positive integer")
            if self.context_window_kind not in _CONTEXT_KINDS:
                raise ValueError(
                    "minimum_context_window requires input, output, or combined context_window_kind"
                )
        elif self.context_window_kind is not None:
            raise ValueError("context_window_kind requires minimum_context_window")


@dataclass(frozen=True, slots=True)
class FactEvidence:
    """A matched fact and the exact provenance that supports it."""

    name: str
    status: FactStatus
    value: Any
    source_ids: tuple[str, ...]
    verified_at: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryMatch:
    samyak_id: str
    provider_id: str
    documentation_disposition: EvidenceDisposition | None
    lifecycle: FactEvidence
    matched_facts: tuple[FactEvidence, ...]


@dataclass(frozen=True, slots=True)
class QueryExclusion:
    samyak_id: str
    provider_id: str
    documentation_disposition: EvidenceDisposition | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QuerySummary:
    total_records_considered: int
    documentation_eligible_records: int
    matching_records: int
    excluded_records: int
    exclusion_reason_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class QueryResult:
    query: ModelQuery
    matches: tuple[QueryMatch, ...]
    exclusions: tuple[QueryExclusion, ...]
    summary: QuerySummary


@dataclass(frozen=True, slots=True)
class PredicateExplanation:
    """Evidence-safe explanation for one evaluated predicate."""

    predicate: str
    outcome: str
    reason_code: str
    reason: str
    source_ids: tuple[str, ...] = ()
    verified_at: tuple[str, ...] = ()


def explain_query_result(result: QueryResult) -> dict[str, tuple[PredicateExplanation, ...]]:
    """Return deterministic explanations without changing query semantics.

    Exclusion counts remain overlapping: a record can fail more than one
    predicate.  Reasons deliberately describe evidence, never availability.
    """
    labels = {
        "documentation_not_eligible": (
            "documentation",
            "excluded_documentation",
            "documentation disposition is not eligible for this query",
        ),
        "provider_mismatch": (
            "provider",
            "excluded_provider",
            "provider does not match the requested provider filter",
        ),
        "lifecycle_not_verified": (
            "lifecycle",
            "missing_verified_evidence",
            "lifecycle is not verified",
        ),
        "lifecycle_mismatch": (
            "lifecycle",
            "excluded_lifecycle",
            "verified lifecycle does not match the requested lifecycle",
        ),
        "context_window_kind_mismatch": (
            "context_window",
            "incompatible_context_kind",
            "verified context kind does not match the requested kind",
        ),
        "context_window_below_minimum": (
            "context_window",
            "below_context_threshold",
            "verified context window is below the requested threshold",
        ),
        "missing_compatible_context_limit": (
            "context_window",
            "missing_verified_evidence",
            "no verified compatible context limit",
        ),
    }

    def explain(reason: str) -> PredicateExplanation:
        if reason in labels:
            predicate, code, text = labels[reason]
        elif reason.endswith("_not_verified"):
            predicate, code, text = (
                reason.removesuffix("_not_verified"),
                "missing_verified_evidence",
                "requested fact is unknown or not verified",
            )
        elif reason.endswith("_mismatch"):
            predicate, code, text = (
                reason.removesuffix("_mismatch"),
                "verified_opposite",
                "verified fact does not satisfy the requested condition",
            )
        else:
            predicate, code, text = reason, reason, reason
        return PredicateExplanation(predicate, "excluded", code, text)

    excluded = {
        item.samyak_id: tuple(explain(reason) for reason in item.reasons)
        for item in result.exclusions
    }
    matched = {
        item.samyak_id: tuple(
            PredicateExplanation(
                predicate=fact.name,
                outcome="matched",
                reason_code="matched_verified",
                reason="matched fact is supported by verified evidence",
                source_ids=fact.source_ids,
                verified_at=fact.verified_at,
            )
            for fact in item.matched_facts
        )
        for item in result.matches
    }
    return matched | excluded


def query_models(records: Iterable[ModelRecord], query: ModelQuery | None = None) -> QueryResult:
    """Evaluate ``query`` deterministically without mutating ``records``.

    Every supplied predicate is combined with AND.  Exclusions retain all
    failed predicates for a record, while the summary counts reasons (so its
    reason counts can exceed ``excluded_records`` for multi-predicate queries).
    """
    selected = ModelQuery() if query is None else query
    ordered = tuple(sorted(records, key=lambda record: record.samyak_id))
    matches: list[QueryMatch] = []
    exclusions: list[QueryExclusion] = []
    eligible = 0
    reasons: Counter[str] = Counter()
    for record in ordered:
        if record.evidence_disposition not in selected.documentation_dispositions:
            exclusion = QueryExclusion(
                samyak_id=record.samyak_id,
                provider_id=record.provider_id,
                documentation_disposition=record.evidence_disposition,
                reasons=("documentation_not_eligible",),
            )
            exclusions.append(exclusion)
            reasons.update(exclusion.reasons)
            continue
        eligible += 1
        failed = _failed_predicates(record, selected)
        if failed:
            exclusion = QueryExclusion(
                samyak_id=record.samyak_id,
                provider_id=record.provider_id,
                documentation_disposition=record.evidence_disposition,
                reasons=tuple(failed),
            )
            exclusions.append(exclusion)
            reasons.update(exclusion.reasons)
            continue
        matches.append(_match(record, selected))
    return QueryResult(
        query=selected,
        matches=tuple(matches),
        exclusions=tuple(exclusions),
        summary=QuerySummary(
            total_records_considered=len(ordered),
            documentation_eligible_records=eligible,
            matching_records=len(matches),
            excluded_records=len(exclusions),
            exclusion_reason_counts=tuple(sorted(reasons.items())),
        ),
    )


def _failed_predicates(record: ModelRecord, query: ModelQuery) -> list[str]:
    reasons: list[str] = []
    if query.providers is not None and record.provider_id not in query.providers:
        reasons.append("provider_mismatch")
    if query.lifecycle is not None:
        reasons.extend(_enum_failure(record.lifecycle, query.lifecycle, "lifecycle"))
    for name, expected in (
        ("api_access", query.api_access),
        ("tool_calling", query.tool_calling),
        ("structured_output", query.structured_output),
    ):
        if expected is not None:
            reasons.extend(_bool_failure(getattr(record, name), expected, name))
    for name, requested in (
        ("input_modalities", query.input_modalities),
        ("output_modalities", query.output_modalities),
    ):
        if requested:
            reasons.extend(_modalities_failure(getattr(record, name), requested, name))
    if query.minimum_context_window is not None:
        reasons.extend(_context_failure(record, query))
    return reasons


def _enum_failure(fact: Fact[Any], expected: LifecycleState, name: str) -> list[str]:
    if fact.status is not FactStatus.KNOWN:
        return [f"{name}_not_verified"]
    return [] if fact.value is expected else [f"{name}_mismatch"]


def _bool_failure(fact: Fact[Any], expected: bool, name: str) -> list[str]:
    if fact.status is not FactStatus.KNOWN or not isinstance(fact.value, bool):
        return [f"{name}_not_verified"]
    return [] if fact.value is expected else [f"{name}_mismatch"]


def _modalities_failure(fact: Fact[Any], requested: frozenset[Modality], name: str) -> list[str]:
    if fact.status is not FactStatus.KNOWN or not isinstance(fact.value, tuple):
        return [f"{name}_not_verified"]
    actual = frozenset(fact.value)
    return [] if requested <= actual else [f"{name}_mismatch"]


def _context_failure(record: ModelRecord, query: ModelQuery) -> list[str]:
    fact = record.context_window
    if fact.status is not FactStatus.KNOWN or fact.value is None:
        return ["missing_compatible_context_limit"]
    context = fact.value
    if context.kind is not query.context_window_kind:
        return ["context_window_kind_mismatch"]
    if context.tokens < query.minimum_context_window:  # type: ignore[operator]
        return ["context_window_below_minimum"]
    return []


def _match(record: ModelRecord, query: ModelQuery) -> QueryMatch:
    names = ["lifecycle"]
    if query.api_access is not None:
        names.append("api_access")
    if query.tool_calling is not None:
        names.append("tool_calling")
    if query.structured_output is not None:
        names.append("structured_output")
    if query.input_modalities:
        names.append("input_modalities")
    if query.output_modalities:
        names.append("output_modalities")
    if query.minimum_context_window is not None:
        names.append("context_window")
    evidence = tuple(_fact_evidence(name, getattr(record, name)) for name in names)
    return QueryMatch(
        samyak_id=record.samyak_id,
        provider_id=record.provider_id,
        documentation_disposition=record.evidence_disposition,
        lifecycle=evidence[0],
        matched_facts=evidence,
    )


def _fact_evidence(name: str, fact: Fact[Any]) -> FactEvidence:
    provenances = []
    if fact.provenance is not None:
        provenances.append(fact.provenance)
    provenances.extend(claim.provenance for claim in fact.claims)
    return FactEvidence(
        name=name,
        status=fact.status,
        value=fact.value,
        source_ids=tuple(provenance.source_id for provenance in provenances),
        verified_at=tuple(
            provenance.verified_at
            for provenance in provenances
            if provenance.verified_at is not None
        ),
    )
