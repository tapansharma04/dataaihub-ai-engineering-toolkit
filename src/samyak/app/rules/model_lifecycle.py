"""First-slice model lifecycle rules over CatalogView resolutions.

Deprecated/retired findings fire only for source-established provider plus
exact catalog identity. Overlay uniqueness never establishes provider.
"""

from __future__ import annotations

from samyak.app.catalog_view import CatalogView, Resolution
from samyak.app.models import (
    CATEGORY_MODEL,
    FINDING_MODEL_PROVIDER_AMBIGUOUS,
    FINDING_MODEL_REFERENCE_DEPRECATED,
    FINDING_MODEL_REFERENCE_RETIRED,
    FINDING_MODEL_REFERENCE_UNKNOWN,
    NOTICE_MODEL_CATALOG_CANDIDATE,
    NOTICE_MODEL_LIFECYCLE_CONFLICT,
    NOTICE_MODEL_REFERENCE_UNCHECKED,
    NOTICE_PROVIDER_NOT_IN_CATALOG,
    AppFinding,
    AppNotice,
    CatalogEvidence,
    Confidence,
    MatchKind,
    Severity,
    sort_findings,
)
from samyak.app.observations import Observation, ObservationKind
from samyak.model.facts import FactStatus, LifecycleState, Provenance
from samyak.model.identity import IdentityKind, ModelIdentity
from samyak.model.records import ModelRecord


def apply_model_lifecycle(
    observations: tuple[Observation, ...],
    view: CatalogView | None,
    *,
    max_notices_per_code: int = 50,
) -> tuple[tuple[AppFinding, ...], tuple[AppNotice, ...]]:
    """Emit lifecycle findings and honesty notices. No-op when catalog is missing."""
    if view is None:
        return (), ()

    freshness = view.freshness_dict()
    findings: list[AppFinding] = []
    missing_providers: set[str] = set()
    unchecked_ids: set[str] = set()
    unique_candidates: dict[str, str] = {}
    conflict_by_id: dict[str, tuple[str, ...]] = {}

    for observation in observations:
        if observation.kind is not ObservationKind.MODEL_REFERENCE:
            continue
        resolution = view.resolve(observation)
        if resolution is None:
            continue
        kind = resolution.match_kind
        if kind is MatchKind.PROVIDER_NOT_IN_CATALOG:
            if resolution.source_provider_id is not None:
                missing_providers.add(resolution.source_provider_id)
            continue
        if kind is MatchKind.NOT_IN_CATALOG:
            if resolution.source_provider_id is None:
                unchecked_ids.add(resolution.provider_model_id)
                continue
            findings.append(_unknown_finding(observation, resolution, freshness))
            continue
        if kind is MatchKind.MODEL_ID_UNIQUE:
            candidate = resolution.candidates[0]
            unique_candidates.setdefault(resolution.provider_model_id, candidate.samyak_id)
            continue
        if kind is MatchKind.MODEL_ID_AMBIGUOUS:
            findings.append(_ambiguous_finding(observation, resolution, freshness))
            continue
        if kind is MatchKind.EXACT_IDENTITY and resolution.record is not None:
            record = resolution.record
            _record_lifecycle_conflict(record, conflict_by_id)
            finding = _lifecycle_finding(observation, resolution, record, freshness)
            if finding is not None:
                findings.append(finding)

    notices = _honesty_notices(
        missing_providers=missing_providers,
        unchecked_ids=unchecked_ids,
        unique_candidates=unique_candidates,
        conflict_by_id=conflict_by_id,
        max_notices_per_code=max_notices_per_code,
    )
    return sort_findings(findings), notices


def _lifecycle_finding(
    observation: Observation,
    resolution: Resolution,
    record: ModelRecord,
    freshness: dict[str, str],
) -> AppFinding | None:
    fact = record.lifecycle
    if fact.status is not FactStatus.KNOWN or fact.value is None:
        return None
    state = fact.value
    if state is LifecycleState.ACTIVE or state is LifecycleState.LEGACY:
        return None
    if state is LifecycleState.DEPRECATED:
        return _exact_lifecycle_finding(
            observation,
            resolution,
            record,
            freshness,
            code=FINDING_MODEL_REFERENCE_DEPRECATED,
            severity=Severity.MEDIUM,
            title="Deprecated model reference",
            why_it_matters="Deprecated models may lose support or change behavior.",
        )
    if state is LifecycleState.RETIRED:
        return _exact_lifecycle_finding(
            observation,
            resolution,
            record,
            freshness,
            code=FINDING_MODEL_REFERENCE_RETIRED,
            severity=Severity.HIGH,
            title="Retired model reference",
            why_it_matters="Retired models are no longer a supported production target.",
        )
    return None


def _exact_lifecycle_finding(
    observation: Observation,
    resolution: Resolution,
    record: ModelRecord,
    freshness: dict[str, str],
    *,
    code: str,
    severity: Severity,
    title: str,
    why_it_matters: str,
) -> AppFinding:
    detected = observation.location.detected_value
    provider = resolution.source_provider_id or ""
    state = record.lifecycle.value
    state_label = state.value if isinstance(state, LifecycleState) else str(state)
    identity_note = ""
    if record.identity_kind is IdentityKind.ALIAS:
        identity_note = f" Catalog identity is alias {record.samyak_id}."
    resolves_note = ""
    resolves_to = _resolves_to_samyak_id(record)
    if record.identity_kind is IdentityKind.ALIAS and resolves_to is not None:
        resolves_note = f" Catalog alias resolves_to {resolves_to}."
    replacements = _known_replacement_ids(record)
    provenance = record.lifecycle.provenance
    evidence = CatalogEvidence(
        match_kind=MatchKind.EXACT_IDENTITY,
        provider_model_id=detected,
        samyak_id=record.samyak_id,
        provider_id=resolution.source_provider_id,
        identity_kind=record.identity_kind.value,
        lifecycle_status=record.lifecycle.status.value,
        lifecycle_value=state_label,
        catalog_freshness=dict(freshness),
        replacement_samyak_ids=replacements,
        fact_confidence=_fact_confidence(provenance),
        source_id=None if provenance is None else provenance.source_id,
        source_url=None if provenance is None else provenance.source_url,
        resolves_to_samyak_id=resolves_to,
    )
    return AppFinding(
        code=code,
        category=CATEGORY_MODEL,
        severity=severity,
        confidence=Confidence.HIGH,
        title=title,
        message=(
            f"Source references model {detected!r} (provider {provider!r}) "
            f"which the local catalog records as {state_label}.{identity_note}{resolves_note}"
        ),
        why_it_matters=why_it_matters,
        recommendation=_replacement_recommendation(record),
        evidence=observation.location,
        catalog_evidence=evidence,
        resolution_source=observation.resolution_source,
        reference_expression=(
            observation.expression if observation.resolution_source is not None else None
        ),
    )


def _unknown_finding(
    observation: Observation,
    resolution: Resolution,
    freshness: dict[str, str],
) -> AppFinding:
    detected = observation.location.detected_value
    provider = resolution.source_provider_id or ""
    return AppFinding(
        code=FINDING_MODEL_REFERENCE_UNKNOWN,
        category=CATEGORY_MODEL,
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        title="Unknown model reference",
        message=(
            f"Provider {provider!r} is established from source, but {detected!r} "
            "is not in that provider's local catalog. Unknown is not invalid, "
            "unsafe, deprecated, or retired."
        ),
        why_it_matters="Absence from the local overlay is not a lifecycle classification.",
        recommendation=(
            "Confirm the model id, or run `samyak model update` separately. "
            "This inspection does not refresh the catalog."
        ),
        evidence=observation.location,
        catalog_evidence=CatalogEvidence(
            match_kind=MatchKind.NOT_IN_CATALOG,
            provider_model_id=detected,
            provider_id=resolution.source_provider_id,
            catalog_freshness=dict(freshness),
        ),
        resolution_source=observation.resolution_source,
        reference_expression=(
            observation.expression if observation.resolution_source is not None else None
        ),
    )


def _ambiguous_finding(
    observation: Observation,
    resolution: Resolution,
    freshness: dict[str, str],
) -> AppFinding:
    detected = observation.location.detected_value
    candidate_ids = tuple(item.samyak_id for item in resolution.candidates)
    listed = ", ".join(candidate_ids)
    return AppFinding(
        code=FINDING_MODEL_PROVIDER_AMBIGUOUS,
        category=CATEGORY_MODEL,
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        title="Ambiguous model provider",
        message=(
            f"Model id {detected!r} matches multiple catalog records ({listed}); "
            "provider is not established from source."
        ),
        why_it_matters="Overlay overlap is not source-established provider identity.",
        recommendation=(
            "Establish the provider in source, for example with a provider SDK "
            "import in the same file."
        ),
        evidence=observation.location,
        catalog_evidence=CatalogEvidence(
            match_kind=MatchKind.MODEL_ID_AMBIGUOUS,
            provider_model_id=detected,
            catalog_freshness=dict(freshness),
            candidate_samyak_ids=candidate_ids,
        ),
        resolution_source=observation.resolution_source,
        reference_expression=(
            observation.expression if observation.resolution_source is not None else None
        ),
    )


def _honesty_notices(
    *,
    missing_providers: set[str],
    unchecked_ids: set[str],
    unique_candidates: dict[str, str],
    conflict_by_id: dict[str, tuple[str, ...]],
    max_notices_per_code: int,
) -> tuple[AppNotice, ...]:
    notices: list[AppNotice] = []
    for provider_id in sorted(missing_providers):
        notices.append(
            AppNotice(
                code=NOTICE_PROVIDER_NOT_IN_CATALOG,
                message=(
                    f"Provider {provider_id!r} is established from source but has "
                    "no records in the local catalog."
                ),
            )
        )
    if unchecked_ids:
        listed = ", ".join(sorted(unchecked_ids))
        notices.append(
            AppNotice(
                code=NOTICE_MODEL_REFERENCE_UNCHECKED,
                message=(
                    "Model ids without a source-established provider were not "
                    f"found in the local catalog: {listed}."
                ),
            )
        )
    for model_id, samyak_id in sorted(unique_candidates.items()):
        notices.append(
            AppNotice(
                code=NOTICE_MODEL_CATALOG_CANDIDATE,
                message=(
                    f"Model id {model_id!r} has a unique catalog candidate "
                    f"{samyak_id}; provider is not established from source, so "
                    "no lifecycle finding was emitted."
                ),
            )
        )
    for samyak_id, values in sorted(conflict_by_id.items()):
        listed = ", ".join(values)
        notices.append(
            AppNotice(
                code=NOTICE_MODEL_LIFECYCLE_CONFLICT,
                message=(
                    f"Catalog lifecycle for {samyak_id} is in conflict ({listed}); "
                    "no deprecated or retired finding was emitted."
                ),
            )
        )
    return tuple(_cap_notices(notices, max_notices_per_code))


def _cap_notices(notices: list[AppNotice], max_per_code: int) -> list[AppNotice]:
    counts: dict[str, int] = {}
    kept: list[AppNotice] = []
    for notice in notices:
        used = counts.get(notice.code, 0)
        if used >= max_per_code:
            continue
        counts[notice.code] = used + 1
        kept.append(notice)
    return kept


def _record_lifecycle_conflict(
    record: ModelRecord, conflict_by_id: dict[str, tuple[str, ...]]
) -> None:
    fact = record.lifecycle
    if fact.status is not FactStatus.CONFLICT:
        return
    values = tuple(
        sorted(
            {
                claim.value.value if isinstance(claim.value, LifecycleState) else str(claim.value)
                for claim in fact.claims
            }
        )
    )
    conflict_by_id.setdefault(record.samyak_id, values)


def _replacement_recommendation(record: ModelRecord) -> str:
    replacements = _known_replacement_ids(record)
    if replacements:
        listed = ", ".join(replacements)
        if len(replacements) == 1:
            return f"Replace this reference with {listed}."
        return f"Replace this reference with one of: {listed}."
    if record.replacement.status is FactStatus.CONFLICT:
        return (
            "Catalog replacement is in conflict; no successor was selected. "
            "Do not assume a documented successor."
        )
    return (
        "The local catalog does not establish a successor. "
        "Confirm a currently supported model from provider documentation."
    )


def _known_replacement_ids(record: ModelRecord) -> tuple[str, ...]:
    fact = record.replacement
    if fact.status is not FactStatus.KNOWN or fact.value is None:
        return ()
    identities = fact.value
    return tuple(item.samyak_id for item in identities if isinstance(item, ModelIdentity))


def _resolves_to_samyak_id(record: ModelRecord) -> str | None:
    fact = record.resolves_to
    if fact.status is not FactStatus.KNOWN or fact.value is None:
        return None
    target = fact.value
    if isinstance(target, ModelIdentity):
        return target.samyak_id
    return None


def _fact_confidence(provenance: Provenance | None) -> str | None:
    if provenance is None or provenance.confidence is None:
        return None
    return provenance.confidence.value
