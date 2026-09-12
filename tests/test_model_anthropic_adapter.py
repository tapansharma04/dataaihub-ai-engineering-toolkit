"""Offline tests for the Anthropic Model Intelligence adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from samyak.model.catalog import CatalogOverlay
from samyak.model.facts import (
    Confidence,
    ContextWindow,
    ContextWindowKind,
    FactStatus,
    LifecycleState,
    Modality,
    SourceKind,
)
from samyak.model.identity import IdentityKind
from samyak.model.providers.anthropic import (
    SOURCE_ID_DEPRECATIONS,
    SOURCE_ID_MODEL_PAGE_PREFIX,
    SOURCE_ID_MODELS_INDEX,
    AnthropicParseError,
    captured_markdown,
    catalog_from_anthropic_sources,
    model_page_source_id,
    model_page_source_id_from_url,
    parse_anthropic_sources,
)
from samyak.model.providers.anthropic.sources import (
    DEPRECATIONS_URL,
    MODELS_INDEX_URL,
    content_hash_for_bytes,
    model_page_url,
)
from samyak.model.serialize import catalog_from_json, catalog_to_json

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "anthropic"
RETRIEVED_AT = "2026-09-11T11:59:00+00:00"
VERIFIED_AT = "2026-09-11T12:00:00+00:00"
GENERATED_AT = "2026-09-11T12:00:00+00:00"


def _source(source_id: str, relative: str, url: str, retrieved_at: str = RETRIEVED_AT):
    body = (FIXTURES / relative).read_text(encoding="utf-8")
    return captured_markdown(
        source_id=source_id,
        source_url=url,
        body=body,
        retrieved_at=retrieved_at,
    )


def _model_page(relative: str, url: str, retrieved_at: str = RETRIEVED_AT):
    return _source(model_page_source_id_from_url(url), relative, url, retrieved_at)


def _representative_sources():
    return (
        _source(SOURCE_ID_MODELS_INDEX, "overview.md", MODELS_INDEX_URL),
        _source(SOURCE_ID_DEPRECATIONS, "deprecations.md", DEPRECATIONS_URL),
        _model_page("models/opus-5.md", model_page_url("opus-5")),
        _model_page("models/haiku-4-5.md", model_page_url("haiku-4-5")),
        _model_page("models/sonnet-5.md", model_page_url("sonnet-5")),
        _model_page("models/fable-5-1.md", model_page_url("fable-5-1")),
        _model_page("models/sparse-test.md", model_page_url("sparse-test")),
        _model_page("models/no-api-access.md", model_page_url("no-api-access")),
        _model_page("models/index-only.md", model_page_url("index-only")),
    )


def _catalog():
    return catalog_from_anthropic_sources(
        _representative_sources(),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )


def _record(samyak_id: str):
    catalog = _catalog()
    matches = [item for item in catalog.models if item.samyak_id == samyak_id]
    assert matches, f"missing {samyak_id}"
    return matches[0]


def test_current_model_identity_and_normalization() -> None:
    record = _record("anthropic:claude-opus-5")
    assert record.provider_id == "anthropic"
    assert record.provider_model_id == "claude-opus-5"
    assert record.identity.samyak_id == "anthropic:claude-opus-5"
    assert record.identity_kind is IdentityKind.CANONICAL
    assert record.display_name.value == "Claude Opus 5"
    assert record.family.value == "Claude Opus"
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.retirement_at.status is FactStatus.UNKNOWN
    assert record.context_window.value == ContextWindow(
        tokens=1_000_000, kind=ContextWindowKind.UNKNOWN
    )
    assert record.max_output_tokens.value == 128_000
    assert record.max_input_tokens.status in {FactStatus.UNKNOWN, FactStatus.NOT_VERIFIED}
    assert record.input_modalities.value == (Modality.TEXT, Modality.IMAGE)
    assert record.output_modalities.value == (Modality.TEXT,)
    assert record.tool_calling.value is True
    assert record.structured_output.status is FactStatus.UNKNOWN
    assert record.api_access.value is True
    assert record.lifecycle.provenance.source_id == SOURCE_ID_DEPRECATIONS
    assert record.context_window.provenance.source_id == f"{SOURCE_ID_MODEL_PAGE_PREFIX}:opus-5"


def test_pinned_id_is_not_stripped_and_alias_is_separate() -> None:
    pinned = _record("anthropic:claude-haiku-4-5-20251001")
    assert pinned.identity_kind is IdentityKind.CANONICAL
    assert pinned.aliases.status is FactStatus.KNOWN
    assert pinned.aliases.value == ("claude-haiku-4-5",)
    alias = _record("anthropic:claude-haiku-4-5")
    assert alias.identity_kind is IdentityKind.ALIAS
    assert alias.resolves_to.value.samyak_id == "anthropic:claude-haiku-4-5-20251001"


def test_dateless_id_is_not_an_alias_of_itself() -> None:
    record = _record("anthropic:claude-sonnet-5")
    assert record.identity_kind is IdentityKind.CANONICAL
    assert record.aliases.status is not FactStatus.KNOWN or "claude-sonnet-5" not in (
        record.aliases.value or ()
    )


def test_partner_platform_ids_are_not_cataloged() -> None:
    ids = {item.provider_model_id for item in _catalog().models}
    assert not any(item.startswith("anthropic.") for item in ids)
    assert not any("@" in item for item in ids)


def test_overview_legacy_prose_is_not_lifecycle_legacy() -> None:
    record = _record("anthropic:claude-opus-5")
    assert record.lifecycle.value is LifecycleState.ACTIVE


def test_explicit_legacy_status_maps_to_legacy() -> None:
    record = _record("anthropic:claude-legacy-test")
    assert record.lifecycle.value is LifecycleState.LEGACY
    assert record.retirement_at.status is FactStatus.UNKNOWN


def test_upcoming_deprecation_stays_deprecated() -> None:
    record = _record("anthropic:claude-soon-deprecated")
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert record.deprecated_at.value == "2026-08-01"
    assert record.retirement_at.value == "2026-12-01"
    assert record.replacement.value[0].samyak_id == "anthropic:claude-opus-5"
    assert record.api_access.value is True


def test_elapsed_retirement_is_aged_against_verified_at() -> None:
    record = _record("anthropic:claude-elapsed-deprecated")
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.retirement_at.value == "2026-06-01"


def test_retired_model_from_status_table() -> None:
    record = _record("anthropic:claude-opus-4-1-20250805")
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.deprecated_at.value == "2026-06-05"
    assert record.retirement_at.value == "2026-08-05"
    assert record.replacement.value[0].provider_model_id == "claude-opus-4-8"
    assert record.api_access.value is False


def test_history_only_retired_model() -> None:
    record = _record("anthropic:claude-2.0")
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.retirement_at.value == "2025-07-21"
    assert record.provider_model_id == "claude-2.0"


def test_prose_deprecated_model_is_included() -> None:
    record = _record("anthropic:claude-mythos-preview")
    assert record.lifecycle.value is LifecycleState.DEPRECATED


def test_json_prose_is_not_structured_output() -> None:
    record = _record("anthropic:claude-opus-5")
    assert record.structured_output.status is FactStatus.UNKNOWN
    assert record.structured_output.value is None


def test_explicit_false_capabilities() -> None:
    catalog = catalog_from_anthropic_sources(
        (_model_page("models/structured-false.md", model_page_url("structured-false")),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.structured_output.value is False
    assert record.tool_calling.value is False


def test_partner_only_platforms_are_not_claude_api_access() -> None:
    record = _record("anthropic:claude-partner-only")
    assert record.api_access.value is False


def test_index_listing_does_not_infer_api_access() -> None:
    catalog = catalog_from_anthropic_sources(
        (
            _source(SOURCE_ID_MODELS_INDEX, "overview.md", MODELS_INDEX_URL),
            _source(SOURCE_ID_DEPRECATIONS, "deprecations.md", DEPRECATIONS_URL),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(
        item for item in catalog.models if item.samyak_id == "anthropic:claude-index-only"
    )
    assert record.api_access.status is FactStatus.NOT_VERIFIED


def test_sparse_page_unknown_capabilities() -> None:
    record = _record("anthropic:claude-sparse-test")
    assert record.context_window.status is FactStatus.UNKNOWN
    assert record.structured_output.status is FactStatus.UNKNOWN
    assert record.tool_calling.status is FactStatus.UNKNOWN


def test_not_verified_without_model_page() -> None:
    catalog = catalog_from_anthropic_sources(
        (
            _source(SOURCE_ID_MODELS_INDEX, "overview.md", MODELS_INDEX_URL),
            _source(SOURCE_ID_DEPRECATIONS, "deprecations.md", DEPRECATIONS_URL),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(item for item in catalog.models if item.samyak_id == "anthropic:claude-2.0")
    assert record.context_window.status is FactStatus.NOT_VERIFIED
    assert record.structured_output.status is FactStatus.NOT_VERIFIED


def test_current_event_wins_over_historical_retirement() -> None:
    catalog = catalog_from_anthropic_sources(
        (_source(SOURCE_ID_DEPRECATIONS, "deprecations-events.md", DEPRECATIONS_URL),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    by_id = {item.provider_model_id: item for item in catalog.models}
    assert by_id["claude-upcoming"].lifecycle.value is LifecycleState.DEPRECATED
    assert by_id["claude-elapsed"].lifecycle.value is LifecycleState.RETIRED
    assert by_id["claude-current-retired"].lifecycle.value is LifecycleState.RETIRED
    assert by_id["claude-current-retired"].replacement.value[0].provider_model_id == "claude-opus-5"
    assert by_id["claude-history-only"].lifecycle.value is LifecycleState.ACTIVE
    assert by_id["claude-conflict"].lifecycle.status is FactStatus.CONFLICT


def test_equal_rank_context_conflict() -> None:
    catalog = catalog_from_anthropic_sources(
        (
            _model_page("models/conflict-window-a.md", model_page_url("conflict-window-a")),
            _model_page("models/conflict-window-b.md", model_page_url("conflict-window-b")),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    fact = catalog.models[0].context_window
    assert fact.status is FactStatus.CONFLICT
    assert {claim.value.tokens for claim in fact.claims} == {200_000, 1_000_000}


def test_malformed_overview_fails_closed() -> None:
    with pytest.raises(AnthropicParseError, match="Compare models"):
        parse_anthropic_sources(
            (_source(SOURCE_ID_MODELS_INDEX, "malformed.md", MODELS_INDEX_URL),)
        )


def test_missing_model_id_fails_page() -> None:
    with pytest.raises(AnthropicParseError, match="Model ID"):
        parse_anthropic_sources(
            (_model_page("models/missing-id.md", model_page_url("missing-id")),)
        )


def test_source_ids_and_content_hash_are_deterministic() -> None:
    record = _record("anthropic:claude-opus-5")
    body = (FIXTURES / "models" / "opus-5.md").read_bytes()
    expected = "sha256:" + hashlib.sha256(body).hexdigest()
    assert content_hash_for_bytes(body) == expected
    assert record.context_window.provenance.content_hash == expected
    assert record.context_window.provenance.source_url == model_page_url("opus-5")
    assert model_page_source_id("opus-5") != model_page_source_id("haiku-4-5")
    source_ids = {item.source_id for item in _catalog().freshness.source_retrieved_at}
    assert SOURCE_ID_MODELS_INDEX in source_ids
    assert SOURCE_ID_DEPRECATIONS in source_ids
    assert f"{SOURCE_ID_MODEL_PAGE_PREFIX}:opus-5" in source_ids
    assert f"{SOURCE_ID_MODEL_PAGE_PREFIX}:haiku-4-5" in source_ids


def test_catalog_round_trip_and_freshness() -> None:
    catalog = _catalog()
    restored = catalog_from_json(catalog_to_json(catalog))
    assert restored.to_dict() == catalog.to_dict()
    assert catalog.freshness.generated_at == GENERATED_AT
    assert catalog.providers[0].id == "anthropic"
    assert catalog.providers[0].model_count == len(catalog.models)
    assert {item.provider_id for item in catalog.models} == {"anthropic"}


def test_confidence_ranks_overview_below_deprecations() -> None:
    record = _record("anthropic:claude-opus-5")
    assert record.lifecycle.provenance.confidence is Confidence.HIGH
    assert record.lifecycle.provenance.source_kind is SourceKind.PROVIDER_DEPRECATIONS
    assert record.display_name.provenance.source_id == f"{SOURCE_ID_MODEL_PAGE_PREFIX}:opus-5"
    assert record.display_name.provenance.confidence is Confidence.HIGH
