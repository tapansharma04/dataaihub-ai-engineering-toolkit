"""Offline tests for the Fireworks AI Model Intelligence adapter."""

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
from samyak.model.identity import IdentityKind, parse_samyak_id
from samyak.model.providers.fireworks import (
    SOURCE_ID_CHANGELOG,
    SOURCE_ID_EMBEDDINGS,
    SOURCE_ID_SERVING_PATHS,
    SOURCE_ID_TEXT_MODELS,
    SOURCE_ID_TOOL_CALLING,
    SOURCE_ID_VISION_MODELS,
    FireworksParseError,
    captured_markdown,
    catalog_from_fireworks_sources,
    model_page_source_id_from_url,
    parse_fireworks_sources,
)
from samyak.model.providers.fireworks.sources import (
    CHANGELOG_URL,
    EMBEDDINGS_URL,
    KIMI_K2_URL,
    SERVING_PATHS_URL,
    TEXT_MODELS_URL,
    TOOL_CALLING_URL,
    VISION_MODELS_URL,
    content_hash_for_bytes,
    model_page_url,
)
from samyak.model.serialize import catalog_from_json, catalog_to_json

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "fireworks"
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
        _source(SOURCE_ID_EMBEDDINGS, "embeddings.md", EMBEDDINGS_URL),
        _source(SOURCE_ID_SERVING_PATHS, "serving-paths.md", SERVING_PATHS_URL),
        _source(SOURCE_ID_CHANGELOG, "changelog.md", CHANGELOG_URL),
        _source(SOURCE_ID_TEXT_MODELS, "text-models.md", TEXT_MODELS_URL),
        _source(SOURCE_ID_VISION_MODELS, "vision-models.md", VISION_MODELS_URL),
        _source(SOURCE_ID_TOOL_CALLING, "tool-calling.md", TOOL_CALLING_URL),
        _model_page("kimi-k2.md", KIMI_K2_URL),
    )


def _catalog(*, verified_at: str = VERIFIED_AT, sources=None):
    return catalog_from_fireworks_sources(
        sources if sources is not None else _representative_sources(),
        generated_at=GENERATED_AT,
        verified_at=verified_at,
        overlay=CatalogOverlay.BUNDLED,
    )


def _record(samyak_id: str, *, catalog=None):
    catalog = catalog if catalog is not None else _catalog()
    matches = [item for item in catalog.models if item.samyak_id == samyak_id]
    assert matches, f"missing {samyak_id}"
    return matches[0]


def test_fireworks_identity_preserves_serving_api_id() -> None:
    record = _record("fireworks:accounts/fireworks/models/deepseek-v3p1")
    assert record.provider_id == "fireworks"
    assert record.provider_model_id == "accounts/fireworks/models/deepseek-v3p1"
    assert record.identity.samyak_id == "fireworks:accounts/fireworks/models/deepseek-v3p1"
    parsed = parse_samyak_id(record.samyak_id)
    assert parsed.provider_id == "fireworks"
    assert parsed.provider_model_id == "accounts/fireworks/models/deepseek-v3p1"
    assert record.identity_kind is IdentityKind.CANONICAL


def test_short_fireworks_id_is_not_collapsed_into_accounts_path() -> None:
    short = _record("fireworks:fireworks/glm-5p2")
    long_id = _record("fireworks:accounts/fireworks/models/glm-5p2")
    assert short.provider_model_id == "fireworks/glm-5p2"
    assert long_id.provider_model_id == "accounts/fireworks/models/glm-5p2"
    assert short.samyak_id != long_id.samyak_id
    assert short.identity_kind is IdentityKind.CANONICAL
    assert long_id.identity_kind is IdentityKind.CANONICAL


def test_fast_router_id_is_not_an_alias_of_the_standard_id() -> None:
    fast = _record("fireworks:accounts/fireworks/routers/glm-5p2-fast")
    standard = _record("fireworks:accounts/fireworks/models/glm-5p2")
    assert fast.identity_kind is IdentityKind.CANONICAL
    assert standard.identity_kind is IdentityKind.CANONICAL
    assert fast.resolves_to.status is FactStatus.NOT_VERIFIED
    assert "accounts/fireworks/routers/glm-5p2-fast" not in (standard.aliases.value or ())
    assert "accounts/fireworks/models/glm-5p2" not in (fast.aliases.value or ())


def test_serverless_modes_parse_as_existing_serving_paths_source() -> None:
    source = _source(SOURCE_ID_SERVING_PATHS, "serving-paths.md", SERVING_PATHS_URL)
    assert SERVING_PATHS_URL == "https://docs.fireworks.ai/serverless/serverless-modes.md"
    assert "/serverless/serving-paths.md" not in SERVING_PATHS_URL
    observations = parse_fireworks_sources((source,))
    ids = {item.provider_model_id for item in observations}
    assert "accounts/fireworks/models/glm-5p2" in ids
    assert "accounts/fireworks/routers/glm-5p2-fast" in ids
    assert "accounts/fireworks/routers/kimi-k3-fast" in ids
    assert "accounts/fireworks/routers/glm-5p3-fast" in ids
    assert "accounts/fireworks/routers/glm-5p2-fast-us" in ids
    assert "Kimi K3 Fast" not in ids
    assert "GLM 5.2 Fast" not in ids
    assert "GLM 5.2 Fast (US)" not in ids
    assert all(item.source.source_id == SOURCE_ID_SERVING_PATHS for item in observations)
    assert all(
        item.source.source_url == "https://docs.fireworks.ai/serverless/serverless-modes.md"
        for item in observations
    )
    kimi_fast = next(
        item
        for item in observations
        if item.provider_model_id == "accounts/fireworks/routers/kimi-k3-fast"
    )
    assert kimi_fast.display_name == "Kimi K3 Fast"
    assert kimi_fast.documented_serverless is True
    assert kimi_fast.api_access is True
    assert kimi_fast.lifecycle is None


def test_new_fast_router_ids_are_canonical_not_aliases() -> None:
    kimi = _record("fireworks:accounts/fireworks/routers/kimi-k3-fast")
    glm53 = _record("fireworks:accounts/fireworks/routers/glm-5p3-fast")
    glm_us = _record("fireworks:accounts/fireworks/routers/glm-5p2-fast-us")
    glm52 = _record("fireworks:accounts/fireworks/models/glm-5p2")
    for record in (kimi, glm53, glm_us):
        assert record.identity_kind is IdentityKind.CANONICAL
        assert record.resolves_to.status is FactStatus.NOT_VERIFIED
        assert record.lifecycle.value is LifecycleState.ACTIVE
        assert record.api_access.value is True
        assert record.api_access.provenance is not None
        assert record.api_access.provenance.source_id == SOURCE_ID_SERVING_PATHS
        assert (
            record.api_access.provenance.source_url
            == "https://docs.fireworks.ai/serverless/serverless-modes.md"
        )
    assert kimi.display_name.value == "Kimi K3 Fast"
    assert kimi.provider_model_id == "accounts/fireworks/routers/kimi-k3-fast"
    assert glm52.lifecycle.value is LifecycleState.ACTIVE


def test_display_name_is_not_the_api_id() -> None:
    record = _record("fireworks:fireworks/qwen3-embedding-8b")
    assert record.display_name.value == "Qwen3 Embedding 8B"
    assert record.provider_model_id == "fireworks/qwen3-embedding-8b"


def test_huggingface_embedder_id_stays_a_fireworks_offering() -> None:
    record = _record("fireworks:BAAI/bge-small-en-v1.5")
    assert record.provider_id == "fireworks"
    assert record.provider_model_id == "BAAI/bge-small-en-v1.5"
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.api_access.value is True


def test_legacy_bert_does_not_inherit_embedding_family_context() -> None:
    qwen = _record("fireworks:fireworks/qwen3-embedding-8b")
    bert = _record("fireworks:BAAI/bge-small-en-v1.5")
    nomic = _record("fireworks:nomic-ai/nomic-embed-text-v1.5")
    assert qwen.context_window.value.tokens == 40_000
    assert bert.context_window.status is FactStatus.UNKNOWN
    assert bert.context_window.value is None
    assert nomic.context_window.status is FactStatus.UNKNOWN
    assert nomic.context_window.value is None


def test_serving_path_and_example_ids_without_context_stay_unknown() -> None:
    router = _record("fireworks:accounts/fireworks/routers/glm-5p2-fast")
    example = _record("fireworks:fireworks/glm-5p2")
    kimi = _record("fireworks:accounts/fireworks/models/kimi-k2-instruct")
    assert router.context_window.status is FactStatus.UNKNOWN
    assert example.context_window.status is FactStatus.UNKNOWN
    assert kimi.context_window.status is FactStatus.UNKNOWN


def test_third_party_origin_does_not_create_origin_provider_records() -> None:
    catalog = _catalog()
    providers = {item.provider_id for item in catalog.models}
    assert providers == {"fireworks"}
    ids = {item.samyak_id for item in catalog.models}
    assert "qwen:fireworks/qwen3-embedding-8b" not in ids
    assert "deepseek:accounts/fireworks/models/deepseek-v3p1" not in ids
    assert "baai:BAAI/bge-small-en-v1.5" not in ids
    assert "meta:BAAI/bge-small-en-v1.5" not in ids
    assert "voyage:fireworks/voyage-4" not in ids


def test_legacy_bert_category_is_not_lifecycle_legacy() -> None:
    record = _record("fireworks:nomic-ai/nomic-embed-text-v1.5")
    assert record.lifecycle.value is not LifecycleState.LEGACY
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.family.status is FactStatus.NOT_VERIFIED


def test_retired_embedder_lifecycle_and_replacement() -> None:
    record = _record("fireworks:sentence-transformers/all-MiniLM-L6-v2")
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.api_access.value is False
    assert record.replacement.value[0].samyak_id == "fireworks:BAAI/bge-small-en-v1.5"
    assert record.replacement.value[0].provider_id == "fireworks"


def test_dedicated_only_voyage_does_not_claim_api_access() -> None:
    record = _record("fireworks:fireworks/voyage-4")
    assert record.api_access.status is not FactStatus.KNOWN or record.api_access.value is not True
    assert record.lifecycle.status is FactStatus.UNKNOWN
    assert record.context_window.value.tokens == 40_000
    assert record.context_window.value.kind is ContextWindowKind.UNKNOWN
    assert record.max_input_tokens.status is FactStatus.UNKNOWN


def test_dedicated_qwen_embedding_does_not_inherit_serverless_access() -> None:
    serverless = _record("fireworks:fireworks/qwen3-embedding-8b")
    dedicated = _record("fireworks:fireworks/qwen3-embedding-4b")
    assert serverless.api_access.value is True
    assert serverless.lifecycle.value is LifecycleState.ACTIVE
    assert dedicated.api_access.status is not FactStatus.KNOWN or (
        dedicated.api_access.value is not True
    )
    assert dedicated.context_window.value.tokens == 40_000


def test_embeddings_do_not_invent_vector_output_modality() -> None:
    record = _record("fireworks:fireworks/qwen3-embedding-8b")
    assert record.input_modalities.value == (Modality.TEXT,)
    assert record.output_modalities.status is FactStatus.UNKNOWN
    assert record.tool_calling.status is FactStatus.UNKNOWN


def test_vision_example_sets_input_modalities_for_that_id_only() -> None:
    vision = _record("fireworks:accounts/fireworks/models/kimi-k2p5")
    text = _record("fireworks:accounts/fireworks/models/deepseek-v3p1")
    assert vision.input_modalities.value == (Modality.TEXT, Modality.IMAGE)
    assert text.input_modalities.value == (Modality.TEXT,)
    assert vision.api_access.value is True
    assert text.api_access.value is True


def test_tool_calling_is_not_copied_onto_family_sibling() -> None:
    dated = _record("fireworks:accounts/fireworks/models/kimi-k2-instruct-0905")
    family = _record("fireworks:accounts/fireworks/models/kimi-k2-instruct")
    assert dated.tool_calling.value is True
    assert family.tool_calling.status is FactStatus.UNKNOWN
    assert dated.provider_id == "fireworks"
    assert family.provider_id == "fireworks"


def test_explicit_alias_is_a_separate_fireworks_record() -> None:
    alias = _record("fireworks:accounts/fireworks/models/kimi-k2-instruct-latest")
    target = _record("fireworks:accounts/fireworks/models/kimi-k2-instruct")
    assert alias.identity_kind is IdentityKind.ALIAS
    assert alias.resolves_to.value.samyak_id == (
        "fireworks:accounts/fireworks/models/kimi-k2-instruct"
    )
    assert target.identity_kind is IdentityKind.CANONICAL
    assert "accounts/fireworks/models/kimi-k2-instruct-latest" in (target.aliases.value or ())


def test_display_name_deprecation_is_not_an_identity() -> None:
    ids = {item.provider_model_id for item in _catalog().models}
    assert "DeepSeek V4 Flash" not in ids
    assert "Kimi K2 Instruct" not in ids


def test_customer_deployment_paths_are_ignored() -> None:
    ids = {item.provider_model_id for item in _catalog().models}
    assert not any("/deployments/" in item for item in ids)
    assert not any("ACCOUNT_ID" in item for item in ids)


def test_current_changelog_event_wins_over_past() -> None:
    record = _record("fireworks:accounts/fireworks/models/event-test")
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert record.replacement.value[0].samyak_id == "fireworks:accounts/fireworks/models/glm-5p2"
    assert record.api_access.status is FactStatus.NOT_VERIFIED


def test_lifecycle_ages_to_retired_after_decommission_date() -> None:
    retired = _record("fireworks:accounts/fireworks/models/aging-test")
    assert retired.lifecycle.value is LifecycleState.RETIRED
    assert retired.retirement_at.value == "2026-01-01"
    earlier = _catalog(verified_at="2025-12-15T00:00:00+00:00")
    pending = _record("fireworks:accounts/fireworks/models/aging-test", catalog=earlier)
    assert pending.lifecycle.value is LifecycleState.DEPRECATED
    assert pending.retirement_at.value == "2026-01-01"


def test_retired_after_decommission_clears_serverless_api_access() -> None:
    serving_body = (
        (FIXTURES / "serving-paths.md")
        .read_text(encoding="utf-8")
        .replace(
            "| GLM 5.2 Fast | `accounts/fireworks/routers/glm-5p2-fast` |",
            "| Aging | `accounts/fireworks/models/aging-test` |\n"
            "| GLM 5.2 Fast | `accounts/fireworks/routers/glm-5p2-fast` |",
        )
    )
    sources = (
        _source(SOURCE_ID_EMBEDDINGS, "embeddings.md", EMBEDDINGS_URL),
        captured_markdown(
            source_id=SOURCE_ID_SERVING_PATHS,
            source_url=SERVING_PATHS_URL,
            body=serving_body,
            retrieved_at=RETRIEVED_AT,
        ),
        _source(SOURCE_ID_CHANGELOG, "changelog.md", CHANGELOG_URL),
    )
    retired = catalog_from_fireworks_sources(
        sources,
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = next(
        item
        for item in retired.models
        if item.samyak_id == "fireworks:accounts/fireworks/models/aging-test"
    )
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.api_access.value is False
    pending_catalog = catalog_from_fireworks_sources(
        sources,
        generated_at=GENERATED_AT,
        verified_at="2025-12-15T00:00:00+00:00",
        overlay=CatalogOverlay.BUNDLED,
    )
    pending = next(
        item
        for item in pending_catalog.models
        if item.samyak_id == "fireworks:accounts/fireworks/models/aging-test"
    )
    assert pending.lifecycle.value is LifecycleState.DEPRECATED
    assert pending.api_access.value is True


def test_conflicting_modalities_are_preserved() -> None:
    catalog = catalog_from_fireworks_sources(
        (
            _source(SOURCE_ID_TEXT_MODELS, "conflict-a.md", TEXT_MODELS_URL),
            _source(SOURCE_ID_VISION_MODELS, "conflict-b.md", VISION_MODELS_URL),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.provider_model_id == "accounts/fireworks/models/conflict-test"
    assert record.input_modalities.status is FactStatus.CONFLICT
    values = {claim.value for claim in record.input_modalities.claims}
    assert values == {(Modality.TEXT,), (Modality.TEXT, Modality.IMAGE)}


def test_malformed_embeddings_fail_closed() -> None:
    with pytest.raises(FireworksParseError, match="embeddings"):
        parse_fireworks_sources((_source(SOURCE_ID_EMBEDDINGS, "malformed.md", EMBEDDINGS_URL),))


def test_missing_model_page_id_fails_closed() -> None:
    with pytest.raises(FireworksParseError, match="serving model id"):
        parse_fireworks_sources((_model_page("missing-id.md", model_page_url("kimi-k2")),))


def test_empty_sources_fail_closed() -> None:
    with pytest.raises(FireworksParseError, match="missing"):
        parse_fireworks_sources(())


def test_empty_changelog_does_not_fail_when_other_sources_list_ids() -> None:
    catalog = catalog_from_fireworks_sources(
        (
            _source(SOURCE_ID_EMBEDDINGS, "embeddings.md", EMBEDDINGS_URL),
            _source(SOURCE_ID_SERVING_PATHS, "serving-paths.md", SERVING_PATHS_URL),
            _source(SOURCE_ID_CHANGELOG, "changelog-empty.md", CHANGELOG_URL),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    assert any(item.provider_model_id == "fireworks/qwen3-embedding-8b" for item in catalog.models)


def test_provenance_and_content_hash() -> None:
    record = _record("fireworks:fireworks/qwen3-embedding-8b")
    assert record.context_window.provenance is not None
    assert record.context_window.provenance.provider == "fireworks"
    assert record.context_window.provenance.source_id == SOURCE_ID_EMBEDDINGS
    assert record.context_window.provenance.source_kind is SourceKind.PROVIDER_DOCS
    assert record.context_window.provenance.confidence is Confidence.HIGH
    body = (FIXTURES / "embeddings.md").read_bytes()
    assert record.context_window.provenance.content_hash == content_hash_for_bytes(body)
    assert record.context_window.provenance.content_hash == (
        "sha256:" + hashlib.sha256(body).hexdigest()
    )
    changelog = _record("fireworks:accounts/fireworks/models/event-test")
    assert changelog.replacement.provenance.source_id == SOURCE_ID_CHANGELOG
    assert changelog.replacement.provenance.source_kind is SourceKind.PROVIDER_CHANGELOG
    assert changelog.replacement.provenance.provider == "fireworks"


def test_freshness_and_round_trip() -> None:
    catalog = _catalog()
    assert catalog.freshness.generated_at == GENERATED_AT
    assert catalog.freshness.oldest_verified_at == VERIFIED_AT
    source_ids = {item.source_id for item in catalog.freshness.source_retrieved_at}
    assert SOURCE_ID_EMBEDDINGS in source_ids
    assert SOURCE_ID_SERVING_PATHS in source_ids
    assert SOURCE_ID_CHANGELOG in source_ids
    restored = catalog_from_json(catalog_to_json(catalog))
    assert restored.to_dict() == catalog.to_dict()


def test_family_is_unmodeled() -> None:
    record = _record("fireworks:accounts/fireworks/models/kimi-k2-instruct")
    assert record.family.status is FactStatus.NOT_VERIFIED
