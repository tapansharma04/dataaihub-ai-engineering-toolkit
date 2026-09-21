"""Fixture-based context-window audit across serving sources. No network."""

from __future__ import annotations

import test_model_anthropic_adapter as anthropic_tests
import test_model_fireworks_adapter as fireworks_tests
import test_model_google_adapter as google_tests
import test_model_openai_adapter as openai_tests
import test_model_together_adapter as together_tests
from context_window_audit import (
    CONFLICT,
    KNOWN,
    NOT_VERIFIED,
    UNKNOWN_INPUT_OUTPUT_ONLY,
    UNKNOWN_NO_EVIDENCE,
    audit_context_windows,
    classify_context_window,
)
from samyak.model.catalog import (
    CatalogFreshness,
    CatalogOverlay,
    FreshnessStatus,
    build_catalog,
)
from samyak.model.facts import ContextWindowKind, FactStatus


def test_audit_counts_fixture_catalogs_by_status() -> None:
    catalogs = {
        "openai": openai_tests._catalog(),
        "anthropic": anthropic_tests._catalog(),
        "google": google_tests._catalog(),
        "fireworks": fireworks_tests._catalog(),
        "together": together_tests._catalog(),
    }
    for provider_id, catalog in catalogs.items():
        audits = audit_context_windows(catalog, provider_id=provider_id)
        assert len(audits) == 1
        audit = audits[0]
        assert audit.provider_id == provider_id
        assert audit.total == len(catalog.models)
        assert audit.known + audit.unknown + audit.not_verified + audit.conflict + audit.other == (
            audit.total
        )
        assert audit.known == sum(
            1 for item in catalog.models if item.context_window.status is FactStatus.KNOWN
        )


def test_known_context_with_unknown_kind_is_still_known() -> None:
    record = openai_tests._record("openai:gpt-5.6-sol")
    assert record.context_window.status is FactStatus.KNOWN
    assert record.context_window.value.kind is ContextWindowKind.UNKNOWN
    assert classify_context_window(record) == KNOWN


def test_google_input_output_only_is_classified_without_filling_context() -> None:
    record = google_tests._record("google:gemini-3.1-flash-image")
    assert record.max_input_tokens.status is FactStatus.KNOWN
    assert record.max_output_tokens.status is FactStatus.KNOWN
    assert record.context_window.status is FactStatus.UNKNOWN
    assert classify_context_window(record) == UNKNOWN_INPUT_OUTPUT_ONLY
    omni = google_tests.catalog_from_google_sources(
        (
            google_tests._model_page(
                "models/gemini-omni-flash.md", google_tests.model_page_url("gemini-omni-flash")
            ),
        ),
        generated_at=google_tests.GENERATED_AT,
        verified_at=google_tests.VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    ).models[0]
    assert omni.provider_model_id == "gemini-omni-1.1-flash"
    assert classify_context_window(omni) == KNOWN
    assert omni.context_window.value.tokens == 1_048_576


def test_google_text_generative_input_output_derives_combined_context() -> None:
    record = google_tests._record("google:gemini-3.8-flash")
    assert classify_context_window(record) == KNOWN
    assert record.context_window.value.tokens == 1_114_112
    assert record.context_window.value.kind is ContextWindowKind.COMBINED
    assert (
        record.context_window.provenance.source_id == google_tests.SOURCE_ID_DERIVED_CONTEXT_WINDOW
    )


def test_google_gemini_3_guide_context_is_audited_as_known() -> None:
    catalog = google_tests._catalog()
    audit = audit_context_windows(catalog, provider_id="google")[0]
    by_id = {row.samyak_id: row for row in audit.rows}
    flash = by_id["google:gemini-3.8-flash"]
    assert flash.classification == KNOWN
    assert flash.tokens == 1_114_112
    assert flash.kind == ContextWindowKind.COMBINED.value
    assert flash.provenance_source_id == google_tests.SOURCE_ID_DERIVED_CONTEXT_WINDOW
    image = by_id["google:gemini-3.1-flash-image"]
    assert image.classification == UNKNOWN_INPUT_OUTPUT_ONLY
    assert image.tokens is None
    lite = by_id["google:gemini-3.1-flash-lite"]
    assert lite.classification == NOT_VERIFIED
    assert lite.tokens is None


def test_together_dash_and_missing_column_and_deprecations_only() -> None:
    dash = together_tests._record("together:Qwen/Qwen3.7-Max")
    image = together_tests._record("together:black-forest-labs/FLUX.1.1-pro")
    retired = together_tests._record("together:nvidia/Nemotron-3-ultra-550b-a55b")
    known = together_tests._record("together:Qwen/Qwen3.8-Flash")
    assert classify_context_window(dash) == UNKNOWN_NO_EVIDENCE
    assert classify_context_window(image) == UNKNOWN_NO_EVIDENCE
    assert classify_context_window(retired) == NOT_VERIFIED
    assert classify_context_window(known) == KNOWN
    assert known.context_window.value.tokens == 1_000_000


def test_fireworks_explicit_versus_missing_context() -> None:
    embedding = fireworks_tests._record("fireworks:fireworks/qwen3-embedding-8b")
    bert = fireworks_tests._record("fireworks:BAAI/bge-small-en-v1.5")
    router = fireworks_tests._record("fireworks:accounts/fireworks/routers/glm-5p2-fast")
    assert classify_context_window(embedding) == KNOWN
    assert embedding.context_window.provenance.source_id == fireworks_tests.SOURCE_ID_EMBEDDINGS
    assert classify_context_window(bert) == UNKNOWN_NO_EVIDENCE
    assert classify_context_window(router) == UNKNOWN_NO_EVIDENCE


def test_no_cross_provider_context_contamination() -> None:
    openai = openai_tests._record("openai:gpt-5.6-sol")
    together = together_tests._record("together:openai/gpt-oss-120b")
    google = google_tests._record("google:gemini-3.8-flash")
    merged = build_catalog(
        models=tuple(sorted((openai, together, google), key=lambda item: item.samyak_id)),
        generated_at=openai_tests.GENERATED_AT,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=openai_tests.GENERATED_AT,
            overlay=CatalogOverlay.BUNDLED,
            oldest_verified_at=openai_tests.VERIFIED_AT,
        ),
    )
    audits = {item.provider_id: item for item in audit_context_windows(merged)}
    assert audits["openai"].known == 1
    assert audits["together"].known == 1
    assert audits["google"].known == 1
    assert audits["google"].unknown == 0
    by_id = {item.samyak_id: item for item in merged.models}
    assert by_id["openai:gpt-5.6-sol"].context_window.value.tokens == 1_050_000
    assert by_id["together:openai/gpt-oss-120b"].context_window.value.tokens == 131072
    assert by_id["google:gemini-3.8-flash"].context_window.value.tokens == 1_114_112


def test_google_context_conflict_is_audited_as_conflict() -> None:
    catalog = google_tests.catalog_from_google_sources(
        (
            google_tests._model_page(
                "models/conflict-context-a.md", google_tests.model_page_url("conflict-context-a")
            ),
            google_tests._model_page(
                "models/conflict-context-b.md", google_tests.model_page_url("conflict-context-b")
            ),
        ),
        generated_at=google_tests.GENERATED_AT,
        verified_at=google_tests.VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    audit = audit_context_windows(catalog, provider_id="google")[0]
    assert audit.conflict == 1
    assert audit.non_known[0].classification == CONFLICT


def test_google_explicit_and_derived_conflict_is_audited() -> None:
    catalog = google_tests.catalog_from_google_sources(
        (
            google_tests._model_page(
                "models/gemini-derived-conflict.md",
                google_tests.model_page_url("gemini-derived-conflict"),
            ),
        ),
        generated_at=google_tests.GENERATED_AT,
        verified_at=google_tests.VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    audit = audit_context_windows(catalog, provider_id="google")[0]
    row = audit.rows[0]
    assert row.classification == CONFLICT
    assert google_tests.SOURCE_ID_DERIVED_CONTEXT_WINDOW in (row.provenance_source_id or "")


def test_anthropic_index_only_context_is_not_verified() -> None:
    record = anthropic_tests._record("anthropic:claude-opus-5")
    assert classify_context_window(record) == KNOWN
    catalog = anthropic_tests.catalog_from_anthropic_sources(
        (
            anthropic_tests._source(
                anthropic_tests.SOURCE_ID_MODELS_INDEX,
                "overview.md",
                anthropic_tests.MODELS_INDEX_URL,
            ),
            anthropic_tests._source(
                anthropic_tests.SOURCE_ID_DEPRECATIONS,
                "deprecations.md",
                anthropic_tests.DEPRECATIONS_URL,
            ),
        ),
        generated_at=anthropic_tests.GENERATED_AT,
        verified_at=anthropic_tests.VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    index_only = next(item for item in catalog.models if item.samyak_id == "anthropic:claude-2.0")
    assert classify_context_window(index_only) == NOT_VERIFIED
