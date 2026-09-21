"""Offline tests for the Google Gemini API Model Intelligence adapter."""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path

import pytest

from samyak.model.catalog import CatalogOverlay
from samyak.model.facts import (
    Confidence,
    ContextWindowKind,
    FactStatus,
    LifecycleState,
    Modality,
    SourceKind,
)
from samyak.model.identity import IdentityKind
from samyak.model.markdown import strip_trailing_markdown_footnote
from samyak.model.providers.google import (
    SOURCE_ID_DEPRECATIONS,
    SOURCE_ID_DERIVED_CONTEXT_WINDOW,
    SOURCE_ID_GEMINI_3,
    SOURCE_ID_MODEL_PAGE_PREFIX,
    SOURCE_ID_MODELS_INDEX,
    GoogleParseError,
    captured_markdown,
    catalog_from_google_sources,
    model_page_source_id_from_url,
    parse_google_sources,
)
from samyak.model.providers.google.observations import GoogleLifecycle
from samyak.model.providers.google.sources import (
    DEPRECATIONS_URL,
    GEMINI_3_URL,
    MODELS_INDEX_URL,
    TOKENS_URL,
    content_hash_for_bytes,
    model_page_url,
)
from samyak.model.serialize import catalog_from_json, catalog_to_json

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "google"
RETRIEVED_AT = "2026-09-12T11:59:00+00:00"
VERIFIED_AT = "2026-09-12T12:00:00+00:00"
GENERATED_AT = "2026-09-12T12:00:00+00:00"


@pytest.fixture(autouse=True)
def deny_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Model Intelligence adapter tests must not use the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("urllib.request.urlopen", blocked)


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
        _source(SOURCE_ID_GEMINI_3, "gemini-3.md", GEMINI_3_URL),
        _model_page("models/gemini-3.8-flash.md", model_page_url("gemini-3.8-flash")),
        _model_page("models/gemini-3.1-flash-image.md", model_page_url("gemini-3.1-flash-image")),
        _model_page("models/gemini-3.1-pro-preview.md", model_page_url("gemini-3.1-pro-preview")),
        _model_page("models/gemini-3.5-transcribe.md", model_page_url("gemini-3.5-transcribe")),
        _model_page("models/index-only.md", model_page_url("index-only")),
        _model_page("models/sparse-test.md", model_page_url("sparse-test")),
        _model_page("models/gemini-stable-dated.md", model_page_url("gemini-stable-dated")),
        _model_page("models/gemini-alias-target.md", model_page_url("gemini-alias-target")),
        _model_page("models/imagen.md", model_page_url("imagen")),
        _model_page("models/gemini-2.0-flash.md", model_page_url("gemini-2.0-flash")),
    )


def _catalog():
    return catalog_from_google_sources(
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
    record = _record("google:gemini-3.8-flash")
    assert record.provider_id == "google"
    assert record.provider_model_id == "gemini-3.8-flash"
    assert record.identity.samyak_id == "google:gemini-3.8-flash"
    assert record.identity_kind is IdentityKind.CANONICAL
    assert record.display_name.value == "Gemini 3.8 Flash"
    assert record.family.status is FactStatus.UNKNOWN
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.api_access.value is True
    assert record.max_input_tokens.value == 1_048_576
    assert record.max_output_tokens.value == 65_536
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.tokens == 1_114_112
    assert record.context_window.value.kind is ContextWindowKind.COMBINED
    assert record.context_window.provenance.source_id == SOURCE_ID_DERIVED_CONTEXT_WINDOW
    assert record.context_window.provenance.source_url == TOKENS_URL
    assert record.tool_calling.value is True
    assert record.structured_output.value is True
    assert record.input_modalities.value == (
        Modality.TEXT,
        Modality.IMAGE,
        Modality.VIDEO,
        Modality.AUDIO,
    )
    assert record.output_modalities.value == (Modality.TEXT,)


def test_plain_token_limits_property_still_parses() -> None:
    record = _record("google:gemini-3.8-flash")
    assert record.max_input_tokens.status is FactStatus.KNOWN
    assert record.max_input_tokens.value == 1_048_576
    assert record.max_output_tokens.status is FactStatus.KNOWN
    assert record.max_output_tokens.value == 65_536
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.tokens == 1_048_576 + 65_536
    assert record.max_input_tokens.status is FactStatus.KNOWN
    assert record.max_output_tokens.status is FactStatus.KNOWN


def test_footnote_suffix_on_token_limits_property_is_normalized() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-token-limits-footnote.md",
                model_page_url("gemini-token-limits-footnote"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(
        item for item in catalog.models if item.samyak_id == "google:gemini-token-limits-footnote"
    )
    assert record.max_input_tokens.status is FactStatus.KNOWN
    assert record.max_input_tokens.value == 1_048_576
    assert record.max_output_tokens.status is FactStatus.KNOWN
    assert record.max_output_tokens.value == 65_536
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.tokens == 1_114_112


def test_live_caret_link_footnote_on_token_limits_property_is_normalized() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-token-limits-footnote-link.md",
                model_page_url("gemini-token-limits-footnote-link"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(
        item
        for item in catalog.models
        if item.samyak_id == "google:gemini-token-limits-footnote-link"
    )
    assert record.max_input_tokens.status is FactStatus.KNOWN
    assert record.max_input_tokens.value == 1_048_576
    assert record.max_output_tokens.status is FactStatus.KNOWN
    assert record.max_output_tokens.value == 65_536
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.tokens == 1_114_112


def test_trailing_markdown_footnote_is_stripped_from_property_names_only() -> None:
    assert strip_trailing_markdown_footnote("Token limits^{[*]}") == "Token limits"
    assert strip_trailing_markdown_footnote("Token limits ^{[*]}") == "Token limits"
    assert (
        strip_trailing_markdown_footnote(
            "Token limits^[\\[\\*\\]](https://ai.google.dev/gemini-api/docs/tokens)^"
        )
        == "Token limits"
    )
    assert strip_trailing_markdown_footnote("Token limits") == "Token limits"
    assert strip_trailing_markdown_footnote("Supported data types") == "Supported data types"
    assert strip_trailing_markdown_footnote("C^{n} complexity") == "C^{n} complexity"


def test_display_name_is_not_the_api_id() -> None:
    record = _record("google:gemini-3.1-flash-image")
    assert record.display_name.value == "Nano Banana 2"
    assert record.provider_model_id == "gemini-3.1-flash-image"
    assert record.identity_kind is IdentityKind.CANONICAL


def test_preview_version_channel_is_not_a_lifecycle_state() -> None:
    record = _record("google:gemini-3.1-pro-preview")
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.provider_model_id == "gemini-3.1-pro-preview"


def test_latest_example_is_not_cataloged() -> None:
    ids = {item.provider_model_id for item in _catalog().models}
    assert "gemini-flash-latest" not in ids


def test_explicit_alias_is_a_separate_record() -> None:
    alias = _record("google:gemini-alias-latest")
    assert alias.identity_kind is IdentityKind.ALIAS
    assert alias.resolves_to.value.samyak_id == "google:gemini-alias-target"
    target = _record("google:gemini-alias-target")
    assert target.identity_kind is IdentityKind.CANONICAL
    assert "gemini-alias-latest" in (target.aliases.value or ())


def test_dateless_id_is_not_an_alias_of_itself() -> None:
    record = _record("google:gemini-3.8-flash")
    assert record.identity_kind is IdentityKind.CANONICAL
    assert record.resolves_to.status is not FactStatus.KNOWN


def test_two_endpoints_in_one_cell_are_not_aliases() -> None:
    unary = _record("google:gemini-3.5-transcribe")
    live = _record("google:gemini-3.5-transcribe-live")
    assert unary.identity_kind is IdentityKind.CANONICAL
    assert live.identity_kind is IdentityKind.CANONICAL
    assert unary.tool_calling.value is False
    assert unary.structured_output.value is False
    assert live.tool_calling.status is FactStatus.NOT_VERIFIED
    assert live.api_access.value is True


def test_vertex_and_cloud_ids_are_not_cataloged() -> None:
    ids = {item.provider_model_id for item in _catalog().models}
    assert not any("/" in item for item in ids)
    assert not any(item.startswith("projects/") for item in ids)


def test_deprecated_index_label_without_exact_retirement_date() -> None:
    record = _record("google:imagen-4.0-generate-001")
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert record.api_access.value is True
    assert record.retirement_at.status is FactStatus.UNKNOWN
    assert record.replacement.value[0].samyak_id == "google:gemini-3.1-flash-image"


def test_earliest_possible_shutdown_date_is_not_retirement_at() -> None:
    record = _record("google:gemini-stable-dated")
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.retirement_at.status is FactStatus.UNKNOWN
    assert record.replacement.value[0].samyak_id == "google:gemini-3.8-flash"


def test_shut_down_index_and_has_been_shut_down_page() -> None:
    record = _record("google:gemini-2.0-flash")
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.api_access.value is False
    assert record.retirement_at.value == "2026-06-01"
    snapshot = _record("google:gemini-2.0-flash-001")
    assert snapshot.lifecycle.value is LifecycleState.RETIRED
    assert snapshot.api_access.value is False
    assert snapshot.identity_kind is IdentityKind.CANONICAL


def test_index_shut_down_wins_over_deprecated_only_page() -> None:
    catalog = catalog_from_google_sources(
        (
            captured_markdown(
                source_id=SOURCE_ID_MODELS_INDEX,
                source_url=MODELS_INDEX_URL,
                body=(
                    "# Models\n\n"
                    "## Gemini 3\n\n"
                    "| Model | Endpoint |\n"
                    "|---|---|\n"
                    "| Gemini Current | `gemini-current-index` |\n"
                    "## Previous models\n\n"
                    "| Model | Endpoint |\n"
                    "|---|---|\n"
                    "| [Shut Page](https://ai.google.dev/gemini-api/docs/models/gemini-shut-page) "
                    "(Shut down) | `gemini-shut-page` |\n"
                ),
                retrieved_at=RETRIEVED_AT,
            ),
            captured_markdown(
                source_id=model_page_source_id_from_url(model_page_url("gemini-shut-page")),
                source_url=model_page_url("gemini-shut-page"),
                body=(
                    "# gemini-shut-page\n\n"
                    "> [!WARNING]\n"
                    "> This model is deprecated.\n\n"
                    "## gemini-shut-page\n\n"
                    "| Property | Description |\n"
                    "|---|---|\n"
                    "| Model code | `gemini-shut-page` |\n"
                ),
                retrieved_at=RETRIEVED_AT,
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(item for item in catalog.models if item.samyak_id == "google:gemini-shut-page")
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.api_access.value is False


def test_previous_models_section_is_not_lifecycle_or_api_access_evidence() -> None:
    observations = parse_google_sources(
        (_source(SOURCE_ID_MODELS_INDEX, "models.md", MODELS_INDEX_URL),)
    )
    unlabeled = next(
        item for item in observations if item.provider_model_id == "gemini-previous-unlabeled"
    )
    assert unlabeled.lifecycle is None
    assert unlabeled.listed_on_index is False
    assert unlabeled.api_access is None
    deprecated = next(
        item for item in observations if item.provider_model_id == "gemini-previous-deprecated"
    )
    assert deprecated.lifecycle is GoogleLifecycle.DEPRECATED
    assert deprecated.listed_on_index is False
    assert deprecated.api_access is None
    shut_down = next(item for item in observations if item.provider_model_id == "gemini-2.0-flash")
    assert shut_down.lifecycle is GoogleLifecycle.RETIRED
    assert shut_down.listed_on_index is False
    assert shut_down.api_access is False
    current = next(item for item in observations if item.provider_model_id == "gemini-3.8-flash")
    assert current.listed_on_index is True
    assert current.api_access is True
    imagen = next(
        item for item in observations if item.provider_model_id == "imagen-4.0-generate-001"
    )
    assert imagen.lifecycle is GoogleLifecycle.DEPRECATED
    assert imagen.listed_on_index is True
    assert imagen.api_access is True

    unlabeled_record = _record("google:gemini-previous-unlabeled")
    assert unlabeled_record.lifecycle.status is FactStatus.UNKNOWN
    assert unlabeled_record.api_access.status is FactStatus.NOT_VERIFIED
    deprecated_record = _record("google:gemini-previous-deprecated")
    assert deprecated_record.lifecycle.value is LifecycleState.DEPRECATED
    assert deprecated_record.api_access.status is FactStatus.NOT_VERIFIED


def test_deprecations_only_id_does_not_invent_lifecycle() -> None:
    record = _record("google:gemini-old-preview")
    assert record.lifecycle.status is FactStatus.UNKNOWN
    assert record.retirement_at.status is FactStatus.UNKNOWN
    assert record.replacement.value[0].samyak_id == "google:gemini-3.1-pro-preview"
    assert record.api_access.status is FactStatus.NOT_VERIFIED


def test_vertex_replacement_is_not_copied_as_a_google_identity() -> None:
    record = _record("google:veo-3.0-generate-001")
    assert record.replacement.value[0].samyak_id == "google:veo-3.1-generate-preview"
    assert len(record.replacement.value) == 1


def test_sparse_page_unknown_capabilities() -> None:
    record = _record("google:gemini-sparse-test")
    assert record.tool_calling.status is FactStatus.UNKNOWN
    assert record.structured_output.status is FactStatus.UNKNOWN
    assert record.max_input_tokens.status is FactStatus.UNKNOWN


def test_index_listing_without_page_is_not_verified() -> None:
    catalog = catalog_from_google_sources(
        (
            _source(SOURCE_ID_MODELS_INDEX, "models.md", MODELS_INDEX_URL),
            _source(SOURCE_ID_DEPRECATIONS, "deprecations.md", DEPRECATIONS_URL),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(item for item in catalog.models if item.samyak_id == "google:gemini-index-only")
    assert record.tool_calling.status is FactStatus.NOT_VERIFIED
    assert record.api_access.value is True
    assert record.lifecycle.value is LifecycleState.ACTIVE


def test_current_deprecation_event_wins_over_past() -> None:
    catalog = catalog_from_google_sources(
        (_source(SOURCE_ID_DEPRECATIONS, "deprecations-events.md", DEPRECATIONS_URL),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(item for item in catalog.models if item.samyak_id == "google:gemini-event-test")
    assert record.replacement.value[0].samyak_id == "google:gemini-3.8-flash"


def test_explicit_limits_context_window_is_known() -> None:
    catalog = catalog_from_google_sources(
        (_model_page("models/gemini-omni-flash.md", model_page_url("gemini-omni-flash")),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(
        item for item in catalog.models if item.samyak_id == "google:gemini-omni-1.1-flash"
    )
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.tokens == 1_048_576
    assert record.context_window.value.kind is ContextWindowKind.UNKNOWN
    assert record.max_input_tokens.status is FactStatus.UNKNOWN
    assert record.max_output_tokens.status is FactStatus.UNKNOWN
    assert record.context_window.provenance.source_id == (
        f"{SOURCE_ID_MODEL_PAGE_PREFIX}:gemini-omni-flash"
    )


def test_context_window_property_row_conflicts_when_derived_sum_differs() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-context-window-property.md",
                model_page_url("gemini-context-window-property"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.max_input_tokens.value == 1_048_576
    assert record.max_output_tokens.value == 65_536
    assert record.context_window.status is FactStatus.CONFLICT
    assert {claim.value.tokens for claim in record.context_window.claims} == {
        1_048_576,
        1_114_112,
    }


def test_unbolded_context_window_colon_form_is_parsed() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-context-window-colon.md",
                model_page_url("gemini-context-window-colon"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.context_window.value.tokens == 2_097_152
    assert record.max_input_tokens.status is FactStatus.UNKNOWN


def test_input_context_window_is_not_combined_context() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-input-context-window.md",
                model_page_url("gemini-input-context-window"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.tokens == 1_114_112
    assert record.context_window.value.kind is ContextWindowKind.COMBINED
    assert record.context_window.provenance.source_id == SOURCE_ID_DERIVED_CONTEXT_WINDOW
    assert record.max_input_tokens.value == 1_048_576
    assert record.max_output_tokens.value == 65_536
    assert record.context_window.value.tokens != record.max_input_tokens.value
    record = _record("google:gemini-3.8-flash")
    assert record.max_input_tokens.value == 1_048_576
    assert record.max_output_tokens.value == 65_536
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.tokens == 1_114_112


def test_limits_text_input_is_not_a_context_window() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-limits-text-input.md",
                model_page_url("gemini-limits-text-input"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.context_window.status is FactStatus.UNKNOWN
    assert record.max_input_tokens.status is FactStatus.UNKNOWN


def test_intro_million_token_prose_is_not_captured() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-context-unrelated.md",
                model_page_url("gemini-context-unrelated"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.tokens == 9_216
    assert record.context_window.value.kind is ContextWindowKind.COMBINED
    assert record.context_window.value.tokens != 1_000_000
    assert record.max_input_tokens.value == 8_192
    assert record.max_output_tokens.value == 1_024


def test_limits_footnote_still_exposes_context_window() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-limits-footnote.md",
                model_page_url("gemini-limits-footnote"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.context_window.value.tokens == 1_048_576
    assert record.max_input_tokens.status is FactStatus.UNKNOWN


def test_index_listing_without_page_leaves_context_not_verified() -> None:
    catalog = catalog_from_google_sources(
        (
            _source(SOURCE_ID_MODELS_INDEX, "models.md", MODELS_INDEX_URL),
            _source(SOURCE_ID_DEPRECATIONS, "deprecations.md", DEPRECATIONS_URL),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(item for item in catalog.models if item.samyak_id == "google:gemini-index-only")
    assert record.context_window.status is FactStatus.NOT_VERIFIED
    assert record.context_window.value is None


def test_conflicting_context_windows_are_preserved() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page("models/conflict-context-a.md", model_page_url("conflict-context-a")),
            _model_page("models/conflict-context-b.md", model_page_url("conflict-context-b")),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.context_window.status is FactStatus.CONFLICT
    assert {claim.value.tokens for claim in record.context_window.claims} == {1000, 2000}


def test_gemini_3_in_out_are_input_and_output_limits() -> None:
    catalog = catalog_from_google_sources(
        (_source(SOURCE_ID_GEMINI_3, "gemini-3.md", GEMINI_3_URL),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    expected = {
        "gemini-3.1-flash-lite": (1_000_000, 64_000),
        "gemini-3.1-pro-preview": (1_000_000, 64_000),
        "gemini-3-flash-preview": (1_000_000, 64_000),
        "gemini-3.1-flash-image-preview": (128_000, 32_000),
        "gemini-3-pro-image-preview": (65_000, 32_000),
    }
    by_id = {item.provider_model_id: item for item in catalog.models}
    assert set(by_id) == set(expected)
    for model_id, (max_input, max_output) in expected.items():
        record = by_id[model_id]
        assert record.max_input_tokens.status is FactStatus.KNOWN
        assert record.max_input_tokens.value == max_input
        assert record.max_output_tokens.status is FactStatus.KNOWN
        assert record.max_output_tokens.value == max_output
        assert record.context_window.status is FactStatus.NOT_VERIFIED
        assert record.context_window.value is None


def test_gemini_3_in_out_are_not_a_combined_context_size() -> None:
    record = _record("google:gemini-3.1-flash-lite")
    assert record.max_input_tokens.value == 1_000_000
    assert record.max_output_tokens.value == 64_000
    assert record.context_window.status is FactStatus.NOT_VERIFIED
    assert record.context_window.value is None


def test_gemini_3_in_out_with_text_modalities_derives_combined_context() -> None:
    catalog = catalog_from_google_sources(
        (
            _source(SOURCE_ID_GEMINI_3, "gemini-3.md", GEMINI_3_URL),
            _model_page(
                "models/gemini-3-in-out-agree.md",
                model_page_url("gemini-3-in-out-agree"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(
        item for item in catalog.models if item.provider_model_id == "gemini-3.1-flash-lite"
    )
    assert record.max_input_tokens.value == 1_000_000
    assert record.max_output_tokens.value == 64_000
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.tokens == 1_064_000
    assert record.context_window.value.kind is ContextWindowKind.COMBINED
    assert record.context_window.provenance.source_id == SOURCE_ID_DERIVED_CONTEXT_WINDOW


def test_derived_context_agrees_with_explicit_combined_value() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-derived-agree.md",
                model_page_url("gemini-derived-agree"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.tokens == 1_114_112
    assert record.context_window.provenance.source_id == (
        f"{SOURCE_ID_MODEL_PAGE_PREFIX}:gemini-derived-agree"
    )
    assert record.max_input_tokens.value == 1_048_576
    assert record.max_output_tokens.value == 65_536


def test_explicit_context_conflicts_with_derived_sum() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-derived-conflict.md",
                model_page_url("gemini-derived-conflict"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.context_window.status is FactStatus.CONFLICT
    assert {claim.value.tokens for claim in record.context_window.claims} == {
        1_000_000,
        1_114_112,
    }
    sources = {claim.provenance.source_id for claim in record.context_window.claims}
    assert f"{SOURCE_ID_MODEL_PAGE_PREFIX}:gemini-derived-conflict" in sources
    assert SOURCE_ID_DERIVED_CONTEXT_WINDOW in sources


def test_missing_output_does_not_derive_context() -> None:
    record = _record("google:imagen-4.0-generate-001")
    assert record.max_input_tokens.status is FactStatus.KNOWN
    assert record.max_output_tokens.status is not FactStatus.KNOWN
    assert record.context_window.status is FactStatus.UNKNOWN


def test_missing_input_does_not_derive_context() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-output-only.md",
                model_page_url("gemini-output-only"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.max_output_tokens.value == 65_536
    assert record.max_input_tokens.status is FactStatus.UNKNOWN
    assert record.context_window.status is FactStatus.UNKNOWN


def test_image_generation_input_output_does_not_derive_context() -> None:
    record = _record("google:gemini-3.1-flash-image")
    assert record.max_input_tokens.status is FactStatus.KNOWN
    assert record.max_output_tokens.status is FactStatus.KNOWN
    assert record.output_modalities.value == (Modality.IMAGE,)
    assert record.context_window.status is FactStatus.UNKNOWN


def test_image_and_text_output_does_not_derive_context() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page(
                "models/gemini-image-text-output.md",
                model_page_url("gemini-image-text-output"),
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.max_input_tokens.value == 65_536
    assert record.max_output_tokens.value == 32_768
    assert record.output_modalities.value == (Modality.IMAGE, Modality.TEXT)
    assert record.context_window.status is FactStatus.UNKNOWN


def test_gemini_3_thinking_table_is_not_a_context_window_source() -> None:
    with pytest.raises(GoogleParseError, match="missing a context-window table"):
        parse_google_sources(
            (_source(SOURCE_ID_GEMINI_3, "gemini-3-no-context-table.md", GEMINI_3_URL),)
        )


def test_conflicting_token_limits_are_preserved() -> None:
    catalog = catalog_from_google_sources(
        (
            _model_page("models/conflict-window-a.md", model_page_url("conflict-window-a")),
            _model_page("models/conflict-window-b.md", model_page_url("conflict-window-b")),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.max_input_tokens.status is FactStatus.CONFLICT
    assert {claim.value for claim in record.max_input_tokens.claims} == {1000, 2000}


def test_missing_model_code_fails_closed() -> None:
    with pytest.raises(GoogleParseError, match="Model code"):
        parse_google_sources((_model_page("models/missing-id.md", model_page_url("missing-id")),))


def test_malformed_index_fails_closed() -> None:
    with pytest.raises(GoogleParseError, match="did not list any Gemini API models"):
        parse_google_sources((_source(SOURCE_ID_MODELS_INDEX, "malformed.md", MODELS_INDEX_URL),))


def test_provenance_and_content_hash() -> None:
    record = _record("google:gemini-3.8-flash")
    assert record.max_input_tokens.provenance is not None
    assert record.max_input_tokens.provenance.provider == "google"
    assert record.max_input_tokens.provenance.source_id.startswith("google-docs-model-page:")
    assert record.max_input_tokens.provenance.source_kind is SourceKind.PROVIDER_DOCS
    assert record.max_input_tokens.provenance.confidence is Confidence.HIGH
    body = (FIXTURES / "models/gemini-3.8-flash.md").read_bytes()
    assert record.max_input_tokens.provenance.content_hash == content_hash_for_bytes(body)
    assert record.max_input_tokens.provenance.content_hash == (
        "sha256:" + hashlib.sha256(body).hexdigest()
    )
    assert record.lifecycle.provenance.source_id == SOURCE_ID_MODELS_INDEX
    deprecations = _record("google:gemini-old-preview")
    assert deprecations.replacement.provenance.source_id == SOURCE_ID_DEPRECATIONS
    assert deprecations.replacement.provenance.source_kind is SourceKind.PROVIDER_DEPRECATIONS


def test_derived_context_provenance_does_not_invent_a_content_hash() -> None:
    record = _record("google:gemini-3.8-flash")
    derived = record.context_window.provenance
    page = record.max_input_tokens.provenance
    assert derived is not None
    assert page is not None
    assert derived.source_id == SOURCE_ID_DERIVED_CONTEXT_WINDOW
    assert derived.source_url == TOKENS_URL
    assert derived.content_hash is None
    assert page.content_hash is not None
    assert record.max_output_tokens.provenance is not None
    assert record.max_output_tokens.provenance.content_hash == page.content_hash
    assert SOURCE_ID_DERIVED_CONTEXT_WINDOW in record.observed_sources
    assert page.source_id in record.observed_sources
    retrieved = {item.source_id for item in _catalog().freshness.source_retrieved_at}
    assert SOURCE_ID_DERIVED_CONTEXT_WINDOW not in retrieved
    assert page.source_id in retrieved


def test_freshness_and_round_trip() -> None:
    catalog = _catalog()
    assert catalog.freshness.generated_at == GENERATED_AT
    assert catalog.freshness.oldest_verified_at == VERIFIED_AT
    source_ids = {item.source_id for item in catalog.freshness.source_retrieved_at}
    assert SOURCE_ID_MODELS_INDEX in source_ids
    assert SOURCE_ID_DEPRECATIONS in source_ids
    assert SOURCE_ID_GEMINI_3 in source_ids
    restored = catalog_from_json(catalog_to_json(catalog))
    assert restored.to_dict() == catalog.to_dict()


def test_empty_sources_fail_closed() -> None:
    with pytest.raises(GoogleParseError, match="missing"):
        parse_google_sources(())
