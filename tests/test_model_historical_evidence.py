"""Regression tests for Google historical specification reconciliation."""

from __future__ import annotations

from samyak.model.catalog import CatalogFreshness, CatalogOverlay, FreshnessStatus, build_catalog
from samyak.model.facts import (
    Confidence,
    ContextWindow,
    ContextWindowKind,
    Fact,
    FactStatus,
    LifecycleState,
    Provenance,
    SourceKind,
)
from samyak.model.historical_evidence import reconcile_historical_specifications
from samyak.model.identity import ModelIdentity
from samyak.model.records import EvidenceDisposition, ModelRecord

OLD = "2026-08-01T12:00:00+00:00"
NOW = "2026-09-01T12:00:00+00:00"


def _provenance(stamp: str, source_id: str) -> Provenance:
    return Provenance(
        provider="google",
        source_id=source_id,
        source_kind=SourceKind.PROVIDER_DOCS,
        source_url=f"https://example.invalid/{source_id}",
        content_hash="sha256:abc",
        retrieved_at=stamp,
        verified_at=stamp,
        confidence=Confidence.HIGH,
    )


def _known(value: object, stamp: str, source_id: str) -> Fact[object]:
    return Fact(status=FactStatus.KNOWN, value=value, provenance=_provenance(stamp, source_id))


def _catalog(*records: ModelRecord, generated_at: str) -> object:
    return build_catalog(
        models=records,
        generated_at=generated_at,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=generated_at,
            overlay=CatalogOverlay.USER_CACHE,
        ),
    )


def _record(
    model_id: str,
    disposition: EvidenceDisposition,
    **fields: object,
) -> ModelRecord:
    return ModelRecord(
        identity=ModelIdentity(provider_id="google", provider_model_id=model_id),
        evidence_disposition=disposition,
        **fields,  # type: ignore[arg-type]
    )


def test_google_historical_reconciliation_preserves_only_unavailable_specifications() -> None:
    old_context = _known(
        ContextWindow(tokens=128_000, kind=ContextWindowKind.COMBINED), OLD, "historic-specs"
    )
    old_input = _known(100_000, OLD, "historic-specs")
    old_tool_calling = _known(True, OLD, "historic-specs")
    prior = _record(
        "retired",
        EvidenceDisposition.CURRENT_DOCUMENTED,
        context_window=old_context,
        max_input_tokens=old_input,
        tool_calling=old_tool_calling,
    )
    current_tool_calling = _known(False, NOW, "current-deprecations")
    current_api_access = _known(False, NOW, "current-deprecations")
    current_lifecycle = _known(LifecycleState.RETIRED, NOW, "current-deprecations")
    incoming = _record(
        "retired",
        EvidenceDisposition.HISTORICAL_REFERENCE_ONLY,
        tool_calling=current_tool_calling,
        api_access=current_api_access,
        lifecycle=current_lifecycle,
    )
    isolated = _record("other-retired", EvidenceDisposition.HISTORICAL_REFERENCE_ONLY)
    current = _record("still-current", EvidenceDisposition.CURRENT_DOCUMENTED)
    previous = _catalog(prior, generated_at=OLD)
    refresh = _catalog(incoming, isolated, current, generated_at=NOW)

    reconciled = reconcile_historical_specifications(previous, refresh, provider_id="google")
    retained = next(item for item in reconciled.models if item.provider_model_id == "retired")
    assert retained.context_window is old_context
    assert retained.max_input_tokens is old_input
    assert retained.tool_calling is current_tool_calling  # Explicit False wins.
    assert retained.api_access is current_api_access  # Operational fields never transfer.
    assert retained.lifecycle is current_lifecycle
    assert retained.context_window.provenance is old_context.provenance
    other = next(item for item in reconciled.models if item.provider_model_id == "other-retired")
    still_current = next(
        item for item in reconciled.models if item.provider_model_id == "still-current"
    )
    assert other.context_window.status is FactStatus.NOT_VERIFIED
    assert still_current.context_window.status is FactStatus.NOT_VERIFIED

    repeated = reconcile_historical_specifications(reconciled, refresh, provider_id="google")
    repeat_record = next(item for item in repeated.models if item.provider_model_id == "retired")
    assert repeat_record.context_window is old_context
    assert repeat_record.context_window.provenance.verified_at == OLD


def test_reconciliation_is_not_enabled_for_another_provider() -> None:
    prior = _record(
        "retired",
        EvidenceDisposition.CURRENT_DOCUMENTED,
        max_output_tokens=_known(4_096, OLD, "historic-specs"),
    )
    incoming = _record("retired", EvidenceDisposition.HISTORICAL_REFERENCE_ONLY)
    reconciled = reconcile_historical_specifications(
        _catalog(prior, generated_at=OLD),
        _catalog(incoming, generated_at=NOW),
        provider_id="openai",
    )
    assert reconciled.models[0].max_output_tokens.status is FactStatus.NOT_VERIFIED
