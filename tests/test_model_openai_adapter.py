"""Offline tests for the OpenAI Model Intelligence adapter."""

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
from samyak.model.providers.openai import (
    SOURCE_ID_DEPRECATIONS,
    SOURCE_ID_MODEL_PAGE_PREFIX,
    SOURCE_ID_MODELS_INDEX,
    OpenAIAdapterError,
    OpenAIParseError,
    captured_markdown,
    catalog_from_openai_sources,
    model_page_source_id,
    model_page_source_id_from_url,
    parse_openai_sources,
)
from samyak.model.providers.openai.sources import (
    DEPRECATIONS_URL,
    MODELS_INDEX_URL,
    content_hash_for_bytes,
    model_page_url,
)
from samyak.model.serialize import catalog_from_json, catalog_to_json

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "openai"
RETRIEVED_AT = "2026-09-09T11:59:00+00:00"
VERIFIED_AT = "2026-09-09T12:00:00+00:00"
GENERATED_AT = "2026-09-09T12:00:00+00:00"


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
        _source(SOURCE_ID_MODELS_INDEX, "models.md", MODELS_INDEX_URL),
        _source(SOURCE_ID_DEPRECATIONS, "deprecations.md", DEPRECATIONS_URL),
        _model_page("models/gpt-5.6-sol.md", model_page_url("gpt-5.6-sol")),
        _model_page("models/gpt-5.6.md", model_page_url("gpt-5.6")),
        _model_page("models/gpt-4.5-preview.md", model_page_url("gpt-4.5-preview")),
        _model_page("models/sparse-test.md", model_page_url("sparse-test")),
    )


def _catalog():
    return catalog_from_openai_sources(
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
    record = _record("openai:gpt-5.6-sol")
    assert record.provider_id == "openai"
    assert record.provider_model_id == "gpt-5.6-sol"
    assert record.identity.samyak_id == "openai:gpt-5.6-sol"
    assert record.identity_kind is IdentityKind.CANONICAL
    assert record.display_name.status is FactStatus.KNOWN
    assert record.display_name.value == "GPT-5.6 Sol"
    assert record.family.status is FactStatus.KNOWN
    assert record.family.value == "GPT-5.6"
    assert record.lifecycle.status is FactStatus.KNOWN
    assert record.lifecycle.value is LifecycleState.ACTIVE


def test_deprecated_model_lifecycle_dates_and_replacement() -> None:
    record = _record("openai:gpt-4.5-preview")
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert record.deprecated_at.value == "2026-06-11"
    assert record.retirement_at.value == "2026-12-11"
    assert record.replacement.status is FactStatus.KNOWN
    assert record.replacement.value is not None
    assert [item.samyak_id for item in record.replacement.value] == ["openai:gpt-5.6-sol"]


def test_retired_model_uses_verified_at_not_absence() -> None:
    record = _record("openai:gpt-old-preview")
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.retirement_at.value == "2025-07-14"
    assert record.context_window.status is FactStatus.NOT_VERIFIED


def _events_sources():
    return (_source(SOURCE_ID_DEPRECATIONS, "deprecations-events.md", DEPRECATIONS_URL),)


def _events_catalog():
    return catalog_from_openai_sources(
        _events_sources(),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )


def _events_record(provider_model_id: str):
    catalog = _events_catalog()
    matches = [item for item in catalog.models if item.provider_model_id == provider_model_id]
    assert matches, f"missing {provider_model_id}"
    return matches[0]


def _replacement_ids(record) -> list[str]:
    assert record.replacement.value is not None
    return [item.samyak_id for item in record.replacement.value]


def test_upcoming_event_retires_when_shutdown_has_elapsed() -> None:
    """Listing chooses the current event; verified_at then ages it.

    An Upcoming row whose shutdown is on/before verified_at is still the
    current announcement. It is not kept as deprecated, and it does not fall
    through to a historical Past row.
    """
    observations = parse_openai_sources(_events_sources())
    solo = [item for item in observations if item.provider_model_id == "upcoming-elapsed"]
    assert len(solo) == 1
    assert solo[0].deprecation_listing == "upcoming"
    assert solo[0].retirement_at == "2026-08-01"
    assert solo[0].retirement_at <= VERIFIED_AT[:10]

    mixed = [
        item for item in observations if item.provider_model_id == "upcoming-elapsed-with-past"
    ]
    assert {item.deprecation_listing for item in mixed} == {"upcoming", "past"}
    upcoming = next(item for item in mixed if item.deprecation_listing == "upcoming")
    past = next(item for item in mixed if item.deprecation_listing == "past")

    record = _events_record("upcoming-elapsed")
    assert record.lifecycle.status is FactStatus.KNOWN
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.deprecated_at.value == "2026-03-01"
    assert record.retirement_at.value == "2026-08-01"
    assert _replacement_ids(record) == ["openai:gpt-elapsed-repl"]

    composed = _events_record("upcoming-elapsed-with-past")
    assert composed.lifecycle.status is FactStatus.KNOWN
    assert composed.lifecycle.value is LifecycleState.RETIRED
    assert composed.deprecated_at.status is FactStatus.KNOWN
    assert composed.deprecated_at.value == upcoming.deprecated_at
    assert composed.retirement_at.status is FactStatus.KNOWN
    assert composed.retirement_at.value == upcoming.retirement_at
    assert composed.replacement.status is FactStatus.KNOWN
    assert _replacement_ids(composed) == ["openai:gpt-elapsed-repl"]
    assert composed.retirement_at.value != past.retirement_at
    assert _replacement_ids(composed) != ["openai:gpt-old-elapsed-repl"]


def test_past_and_upcoming_uses_upcoming_event() -> None:
    observations = parse_openai_sources(_events_sources())
    events = [item for item in observations if item.provider_model_id == "listing-sensitive"]
    listings = {item.deprecation_listing for item in events}
    assert listings == {"upcoming", "past"}
    past = next(item for item in events if item.deprecation_listing == "past")
    upcoming = next(item for item in events if item.deprecation_listing == "upcoming")
    assert past.deprecated_at is not None
    assert upcoming.deprecated_at is not None
    assert past.deprecated_at > upcoming.deprecated_at

    record = _events_record("listing-sensitive")
    assert record.lifecycle.status is FactStatus.KNOWN
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert record.deprecated_at.status is FactStatus.KNOWN
    assert record.deprecated_at.value == upcoming.deprecated_at
    assert record.retirement_at.status is FactStatus.KNOWN
    assert record.retirement_at.value == upcoming.retirement_at
    assert record.replacement.status is FactStatus.KNOWN
    assert _replacement_ids(record) == ["openai:gpt-upcoming-repl"]
    assert record.deprecated_at.value != past.deprecated_at
    assert record.retirement_at.value != past.retirement_at


def test_incompatible_upcoming_events_preserve_conflict() -> None:
    observations = parse_openai_sources(_events_sources())
    events = [item for item in observations if item.provider_model_id == "two-upcoming"]
    assert {item.deprecation_listing for item in events} == {"upcoming"}
    identities = {(item.deprecated_at, item.retirement_at, item.replacements) for item in events}
    assert len(identities) == 2

    record = _events_record("two-upcoming")
    assert record.lifecycle.status is not FactStatus.CONFLICT
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert record.deprecated_at.status is FactStatus.CONFLICT
    assert record.retirement_at.status is FactStatus.CONFLICT
    assert record.replacement.status is FactStatus.CONFLICT
    assert {claim.value for claim in record.deprecated_at.claims} == {"2026-01-01", "2026-06-01"}
    assert {claim.value for claim in record.retirement_at.claims} == {"2026-12-01", "2027-06-01"}
    assert {
        tuple(item.samyak_id for item in claim.value) for claim in record.replacement.claims
    } == {("openai:repl-a",), ("openai:repl-b",)}


def test_multiple_past_events_use_latest_announcement() -> None:
    observations = parse_openai_sources(_events_sources())
    events = [item for item in observations if item.provider_model_id == "two-past"]
    assert {item.deprecation_listing for item in events} == {"past"}
    older = next(item for item in events if item.deprecated_at == "2024-08-29")
    current = next(item for item in events if item.deprecated_at == "2025-09-26")
    assert older.retirement_at is not None
    assert current.retirement_at is not None
    assert older.retirement_at > current.retirement_at

    record = _events_record("two-past")
    assert record.lifecycle.status is FactStatus.KNOWN
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.deprecated_at.status is FactStatus.KNOWN
    assert record.deprecated_at.value == current.deprecated_at
    assert record.retirement_at.status is FactStatus.KNOWN
    assert record.retirement_at.value == current.retirement_at
    assert record.replacement.status is FactStatus.KNOWN
    assert _replacement_ids(record) == ["openai:gpt-current-repl"]
    assert record.retirement_at.value != older.retirement_at
    assert _replacement_ids(record) != ["openai:gpt-old-repl"]


def test_ambiguous_past_events_fail_closed() -> None:
    record = _events_record("two-past-tied")
    assert record.lifecycle.status is FactStatus.KNOWN
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.deprecated_at.status is FactStatus.KNOWN
    assert record.deprecated_at.value == "2025-01-01"
    assert record.retirement_at.status is FactStatus.CONFLICT
    assert record.replacement.status is FactStatus.CONFLICT
    assert {claim.value for claim in record.retirement_at.claims} == {"2025-06-01", "2025-08-01"}


def test_index_listing_may_establish_active() -> None:
    listed = _record("openai:index-only")
    assert listed.lifecycle.status is FactStatus.KNOWN
    assert listed.lifecycle.value is LifecycleState.ACTIVE
    assert listed.lifecycle.provenance is not None
    assert listed.lifecycle.provenance.source_id == SOURCE_ID_MODELS_INDEX
    assert listed.context_window.status is FactStatus.NOT_VERIFIED


def test_absence_of_deprecation_does_not_establish_active() -> None:
    catalog = catalog_from_openai_sources(
        (_model_page("models/gpt-5.6-sol.md", model_page_url("gpt-5.6-sol")),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(item for item in catalog.models if item.provider_model_id == "gpt-5.6-sol")
    assert record.lifecycle.status is FactStatus.UNKNOWN
    assert record.lifecycle.value is None


def test_absence_from_index_does_not_establish_retired() -> None:
    catalog = catalog_from_openai_sources(
        (_model_page("models/sparse-test.md", model_page_url("sparse-test")),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.lifecycle.status is FactStatus.UNKNOWN
    assert record.lifecycle.value is not LifecycleState.RETIRED


def test_alias_snapshot_normalization() -> None:
    alias = _record("openai:gpt-5.6")
    canonical = _record("openai:gpt-5.6-sol")
    assert alias.identity_kind is IdentityKind.ALIAS
    assert alias.resolves_to.status is FactStatus.KNOWN
    assert alias.resolves_to.value is not None
    assert alias.resolves_to.value.samyak_id == "openai:gpt-5.6-sol"
    assert canonical.aliases.status is FactStatus.KNOWN
    assert canonical.aliases.value is not None
    assert "gpt-5.6" in canonical.aliases.value


def test_unrelated_alias_prose_does_not_create_alias_records() -> None:
    catalog = catalog_from_openai_sources(
        (_model_page("models/alias-prose-trap.md", model_page_url("trap-model")),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    ids = {item.provider_model_id for item in catalog.models}
    assert ids == {"trap-model"}
    record = catalog.models[0]
    assert record.identity_kind is IdentityKind.CANONICAL
    assert record.resolves_to.status is FactStatus.NOT_VERIFIED
    assert record.aliases.status is FactStatus.UNKNOWN


def test_context_window_and_modalities() -> None:
    record = _record("openai:gpt-5.6-sol")
    assert record.context_window.value == ContextWindow(
        tokens=1_050_000, kind=ContextWindowKind.UNKNOWN
    )
    assert record.max_input_tokens.value == 922_000
    assert record.max_output_tokens.value == 128_000
    assert record.input_modalities.value == (Modality.TEXT, Modality.IMAGE)
    assert record.output_modalities.value == (Modality.TEXT,)


def test_tool_calling_structured_output_and_api_access() -> None:
    record = _record("openai:gpt-5.6-sol")
    assert record.tool_calling.is_known_true()
    assert record.structured_output.is_known_true()
    assert record.api_access.is_known_true()


def test_omitted_capability_is_unknown_not_false() -> None:
    catalog = catalog_from_openai_sources(
        (_model_page("models/feature-sparse.md", model_page_url("feature-sparse")),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.tool_calling.status is FactStatus.UNKNOWN
    assert record.tool_calling.value is None
    assert not record.tool_calling.is_known_false()
    assert record.structured_output.status is FactStatus.UNKNOWN
    assert not record.structured_output.is_known_false()


def test_explicit_endpoint_negatives_set_api_access_false() -> None:
    catalog = catalog_from_openai_sources(
        (_model_page("models/no-api-access.md", model_page_url("no-api-access")),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.api_access.status is FactStatus.KNOWN
    assert record.api_access.is_known_false()
    assert record.api_access.provenance is not None
    assert record.api_access.provenance.source_id == model_page_source_id("no-api-access")


def test_missing_optional_fields_are_unknown_not_false() -> None:
    record = _record("openai:sparse-test")
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.tool_calling.status is FactStatus.UNKNOWN
    assert record.tool_calling.value is None
    assert not record.tool_calling.is_known_false()
    assert record.structured_output.status is FactStatus.UNKNOWN
    assert record.context_window.status is FactStatus.UNKNOWN
    assert record.max_input_tokens.status is FactStatus.UNKNOWN
    assert record.availability_scope.status is FactStatus.UNKNOWN
    assert record.regions.status is FactStatus.UNKNOWN
    assert record.family.status is FactStatus.UNKNOWN


def test_availability_is_not_inferred_from_silence() -> None:
    record = _record("openai:gpt-5.6-sol")
    assert record.availability_scope.status is FactStatus.UNKNOWN
    assert record.regions.status is FactStatus.UNKNOWN


def test_malformed_source_is_rejected() -> None:
    source = _source(SOURCE_ID_MODELS_INDEX, "malformed.md", MODELS_INDEX_URL)
    with pytest.raises(OpenAIParseError, match="Models heading") as caught:
        parse_openai_sources((source,))
    message = str(caught.value)
    assert "split_sections" not in message
    assert "KeyError" not in message
    assert "BeautifulSoup" not in message


def test_missing_required_identity_is_rejected() -> None:
    source = _model_page("models/missing-id.md", model_page_url("missing-id"))
    with pytest.raises(OpenAIParseError, match="Model ID") as caught:
        parse_openai_sources((source,))
    assert "__post_init__" not in str(caught.value)


def test_contradictory_deprecation_dates_fail_closed() -> None:
    source = _source(SOURCE_ID_DEPRECATIONS, "contradictory-dates.md", DEPRECATIONS_URL)
    with pytest.raises(
        OpenAIAdapterError, match="retirement_at earlier than deprecated_at"
    ) as caught:
        parse_openai_sources((source,))
    assert "deprecated_at = None" not in str(caught.value)


def test_provenance_and_deterministic_content_hash() -> None:
    path = FIXTURES / "models" / "gpt-5.6-sol.md"
    expected_hash = content_hash_for_bytes(path.read_bytes())
    assert expected_hash == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    record = _record("openai:gpt-5.6-sol")
    provenance = record.context_window.provenance
    assert provenance is not None
    assert provenance.provider == "openai"
    assert provenance.source_id == model_page_source_id("gpt-5.6-sol")
    assert provenance.source_kind is SourceKind.PROVIDER_DOCS
    assert provenance.source_url == model_page_url("gpt-5.6-sol")
    assert provenance.content_hash == expected_hash
    assert provenance.retrieved_at == RETRIEVED_AT
    assert provenance.observed_at is None
    assert provenance.verified_at == VERIFIED_AT
    assert provenance.confidence is Confidence.HIGH
    lifecycle = record.lifecycle.provenance
    assert lifecycle is not None
    assert lifecycle.source_id == SOURCE_ID_MODELS_INDEX
    assert lifecycle.confidence is Confidence.MEDIUM


def test_source_retrieved_at_keeps_every_model_page() -> None:
    earlier = "2026-09-09T11:00:00+00:00"
    later = "2026-09-09T11:30:00+00:00"
    catalog = catalog_from_openai_sources(
        (
            _model_page("models/gpt-5.6-sol.md", model_page_url("gpt-5.6-sol"), earlier),
            _model_page("models/sparse-test.md", model_page_url("sparse-test"), later),
            _model_page("models/gpt-4.5-preview.md", model_page_url("gpt-4.5-preview"), later),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    retrieved = {
        item.source_id: item.retrieved_at for item in catalog.freshness.source_retrieved_at
    }
    assert retrieved[model_page_source_id("gpt-5.6-sol")] == earlier
    assert retrieved[model_page_source_id("sparse-test")] == later
    assert retrieved[model_page_source_id("gpt-4.5-preview")] == later
    assert SOURCE_ID_MODEL_PAGE_PREFIX not in retrieved
    page_ids = [
        item.source_id
        for item in catalog.freshness.source_retrieved_at
        if item.source_id.startswith(f"{SOURCE_ID_MODEL_PAGE_PREFIX}:")
    ]
    assert len(page_ids) == 3
    assert len(set(page_ids)) == 3

    representative = _catalog()
    representative_pages = [
        item.source_id
        for item in representative.freshness.source_retrieved_at
        if item.source_id.startswith(f"{SOURCE_ID_MODEL_PAGE_PREFIX}:")
    ]
    assert set(representative_pages) == {
        model_page_source_id("gpt-5.6-sol"),
        model_page_source_id("gpt-5.6"),
        model_page_source_id("gpt-4.5-preview"),
        model_page_source_id("sparse-test"),
    }


def test_shared_model_page_source_id_is_rejected() -> None:
    with pytest.raises(OpenAIParseError, match="not a known documentation source"):
        captured_markdown(
            source_id=SOURCE_ID_MODEL_PAGE_PREFIX,
            source_url=model_page_url("gpt-5.6-sol"),
            body="# X\n\nModel ID: `gpt-5.6-sol`\n",
            retrieved_at=RETRIEVED_AT,
        )


def test_dangling_replacement_is_allowed() -> None:
    record = _record("openai:gpt-dangling-source")
    assert record.replacement.status is FactStatus.KNOWN
    assert record.replacement.value is not None
    assert record.replacement.value[0].samyak_id == "openai:gpt-missing-successor"
    catalog = _catalog()
    assert "openai:gpt-missing-successor" not in {item.samyak_id for item in catalog.models}


def test_no_fabricated_facts_from_unlabeled_prose() -> None:
    record = _record("openai:sparse-test")
    assert record.context_window.status is FactStatus.UNKNOWN
    assert record.context_window.value is None


def test_lifecycle_unknown_when_sources_do_not_establish_it() -> None:
    catalog = catalog_from_openai_sources(
        (_model_page("models/sparse-test.md", model_page_url("sparse-test")),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.lifecycle.status is FactStatus.UNKNOWN
    assert record.lifecycle.value is None
    assert not hasattr(record.lifecycle, "LEGACY")


def test_conflict_preserves_competing_values() -> None:
    sources = (
        _model_page(
            "models/conflict-window-a.md",
            "https://developers.openai.com/api/docs/models/conflict-window-a.md",
        ),
        _model_page(
            "models/conflict-window-b.md",
            "https://developers.openai.com/api/docs/models/conflict-window-b.md",
        ),
    )
    catalog = catalog_from_openai_sources(
        sources,
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    fact = catalog.models[0].context_window
    assert fact.status is FactStatus.CONFLICT
    assert fact.value is None
    assert [claim.value.tokens for claim in fact.claims] == [128_000, 200_000]
    assert {claim.provenance.source_id for claim in fact.claims} == {
        model_page_source_id("conflict-window-a"),
        model_page_source_id("conflict-window-b"),
    }
    assert {claim.provenance.source_url for claim in fact.claims} == {
        "https://developers.openai.com/api/docs/models/conflict-window-a.md",
        "https://developers.openai.com/api/docs/models/conflict-window-b.md",
    }
    assert len({claim.provenance.content_hash for claim in fact.claims}) == 2
    rebuilt = catalog_from_json(catalog_to_json(catalog))
    assert rebuilt.to_dict() == catalog.to_dict()


def test_catalog_output_is_deterministic() -> None:
    first = _catalog()
    second = _catalog()
    assert catalog_to_json(first) == catalog_to_json(second)
    assert catalog_from_json(catalog_to_json(first)).to_dict() == first.to_dict()
    assert first.freshness.overlay is CatalogOverlay.BUNDLED
    assert first.product == "samyak"
    assert first.capability == "model"


def test_deprecations_provenance_source_kind() -> None:
    record = _record("openai:gpt-4.5-preview")
    assert record.retirement_at.provenance is not None
    assert record.retirement_at.provenance.source_id == SOURCE_ID_DEPRECATIONS
    assert record.retirement_at.provenance.source_kind is SourceKind.PROVIDER_DEPRECATIONS
    assert record.retirement_at.provenance.source_url == DEPRECATIONS_URL
