"""Offline tests for the Together AI Model Intelligence adapter."""

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
from samyak.model.providers.together import (
    SOURCE_ID_CHANGELOG,
    SOURCE_ID_DEPRECATIONS,
    SOURCE_ID_MODEL_PAGE_PREFIX,
    SOURCE_ID_SERVERLESS,
    TogetherParseError,
    captured_markdown,
    catalog_from_together_sources,
    parse_together_sources,
)
from samyak.model.providers.together.sources import (
    CHANGELOG_URL,
    DEPRECATIONS_URL,
    SERVERLESS_URL,
    content_hash_for_bytes,
    model_page_source_id,
    model_page_url,
    source_type_for_id,
)
from samyak.model.serialize import catalog_from_json, catalog_to_json

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "together"
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


def _representative_sources():
    return (
        _source(SOURCE_ID_SERVERLESS, "serverless.md", SERVERLESS_URL),
        _source(SOURCE_ID_DEPRECATIONS, "deprecations.md", DEPRECATIONS_URL),
        _source(SOURCE_ID_CHANGELOG, "changelog.md", CHANGELOG_URL),
        _model_page("glm-5.2-quickstart"),
    )


def _model_page(slug: str, relative: str | None = None):
    filename = relative if relative is not None else f"{slug}.md"
    return _source(model_page_source_id(slug), filename, model_page_url(slug))


def _catalog(*, verified_at: str = VERIFIED_AT, sources=None):
    return catalog_from_together_sources(
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


def test_together_identity_preserves_api_model_string() -> None:
    record = _record("together:openai/gpt-oss-120b")
    assert record.provider_id == "together"
    assert record.provider_model_id == "openai/gpt-oss-120b"
    assert record.identity.samyak_id == "together:openai/gpt-oss-120b"
    parsed = parse_samyak_id(record.samyak_id)
    assert parsed.provider_id == "together"
    assert parsed.provider_model_id == "openai/gpt-oss-120b"
    assert record.identity_kind is IdentityKind.CANONICAL


def test_organization_prefix_and_slash_are_preserved() -> None:
    qwen = _record("together:Qwen/Qwen3.5-9B")
    llama = _record("together:meta-llama/Llama-3.3-70B-Instruct-Turbo")
    gemma = _record("together:google/gemma-4-31B-it")
    assert qwen.provider_model_id == "Qwen/Qwen3.5-9B"
    assert llama.provider_model_id == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    assert gemma.provider_model_id == "google/gemma-4-31B-it"


def test_organization_does_not_create_origin_provider_records() -> None:
    catalog = _catalog()
    providers = {item.provider_id for item in catalog.models}
    assert providers == {"together"}
    ids = {item.samyak_id for item in catalog.models}
    assert "openai:gpt-oss-120b" not in ids
    assert "openai:openai/gpt-oss-120b" not in ids
    assert "google:gemma-4-31B-it" not in ids
    assert "qwen:Qwen/Qwen3.5-9B" not in ids
    assert "meta:meta-llama/Llama-3.3-70B-Instruct-Turbo" not in ids


def test_no_cross_provider_aliasing() -> None:
    record = _record("together:openai/gpt-oss-120b")
    assert record.aliases.status is FactStatus.NOT_VERIFIED
    assert record.resolves_to.status is FactStatus.NOT_VERIFIED
    assert record.family.status is FactStatus.NOT_VERIFIED


def test_serverless_model_is_included() -> None:
    record = _record("together:openai/gpt-oss-120b")
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.api_access.value is True
    assert record.display_name.value == "GPT-OSS 120B"


def test_dedicated_only_and_customer_models_are_excluded() -> None:
    catalog = _catalog()
    ids = {item.samyak_id for item in catalog.models}
    assert "together:togethercomputer/dedicated-only-model" not in ids
    assert "together:customer/private-finetune" not in ids
    assert "together:togethercomputer/dedicated-changelog-only" not in ids
    assert "together:togethercomputer/changelog-launch-only" not in ids


def test_rerank_dedicated_only_model_is_excluded() -> None:
    catalog = _catalog()
    ids = {item.provider_model_id for item in catalog.models}
    assert "mixedbread-ai/mxbai-rerank-large-v2" not in ids
    assert "together:mixedbread-ai/mxbai-rerank-large-v2" not in {
        item.samyak_id for item in catalog.models
    }


def test_explicit_function_calling_and_structured_outputs() -> None:
    record = _record("together:openai/gpt-oss-120b")
    assert record.tool_calling.value is True
    assert record.structured_output.value is True
    assert record.tool_calling.provenance.source_id == SOURCE_ID_SERVERLESS


def test_missing_capability_stays_unknown() -> None:
    record = _record("together:google/gemma-4-31B-it")
    assert record.tool_calling.status is FactStatus.UNKNOWN
    assert record.structured_output.status is FactStatus.UNKNOWN


def test_no_sibling_capability_inheritance() -> None:
    supported = _record("together:sibling-org/sibling-a")
    sparse = _record("together:sibling-org/sibling-b")
    assert supported.tool_calling.value is True
    assert supported.structured_output.value is True
    assert sparse.tool_calling.status is FactStatus.UNKNOWN
    assert sparse.structured_output.status is FactStatus.UNKNOWN


def test_explicit_context_length_is_not_max_output() -> None:
    record = _record("together:openai/gpt-oss-120b")
    assert record.context_window.value.tokens == 131072
    assert record.context_window.value.kind is ContextWindowKind.UNKNOWN
    assert record.max_input_tokens.status is FactStatus.UNKNOWN
    assert record.max_output_tokens.status is FactStatus.UNKNOWN


def test_dash_context_length_stays_unknown() -> None:
    record = _record("together:Qwen/Qwen3.7-Max")
    assert record.context_window.status is FactStatus.UNKNOWN
    assert record.context_window.value is None
    assert record.max_input_tokens.status is FactStatus.UNKNOWN
    assert record.max_output_tokens.status is FactStatus.UNKNOWN


def test_comma_formatted_serverless_context_is_captured() -> None:
    record = _record("together:Qwen/Qwen-comma")
    assert record.context_window.value.tokens == 131072


def test_suffix_formatted_serverless_context_is_captured() -> None:
    record = _record("together:Qwen/Qwen-ksuffix")
    assert record.context_window.value.tokens == 128_000
    assert record.max_input_tokens.status is FactStatus.UNKNOWN
    assert record.max_output_tokens.status is FactStatus.UNKNOWN


def test_image_table_without_context_column_stays_unknown() -> None:
    record = _record("together:black-forest-labs/FLUX.1.1-pro")
    assert record.context_window.status is FactStatus.UNKNOWN
    assert record.context_window.value is None


def test_separate_together_model_page_is_outside_source_scope() -> None:
    page = FIXTURES / "model-page-qwen37.md"
    assert "Qwen/Qwen3.7-Max" in page.read_text(encoding="utf-8")
    assert "262144" in page.read_text(encoding="utf-8")
    with pytest.raises(TogetherParseError, match="not a known documentation source"):
        source_type_for_id(f"{SOURCE_ID_MODEL_PAGE_PREFIX}:qwen37")
    record = _record("together:Qwen/Qwen3.7-Max")
    assert record.context_window.status is FactStatus.UNKNOWN
    assert record.context_window.value is None


def test_glm_quickstart_establishes_output_without_inferring_input() -> None:
    record = _record("together:zai-org/GLM-5.2")
    assert record.context_window.value.tokens == 512_000
    assert record.context_window.value.kind is ContextWindowKind.UNKNOWN
    assert record.max_output_tokens.value == 128_000
    assert record.max_input_tokens.status is FactStatus.UNKNOWN
    assert record.max_input_tokens.value is None
    assert record.max_output_tokens.provenance.source_id == model_page_source_id(
        "glm-5.2-quickstart"
    )
    assert record.context_window.provenance.source_id == SOURCE_ID_SERVERLESS


def test_kimi_k3_quickstart_captures_explicit_input_not_output_ceiling() -> None:
    catalog = catalog_from_together_sources(
        _representative_sources() + (_model_page("kimi-k3-quickstart"),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = _record("together:moonshotai/Kimi-K3", catalog=catalog)
    assert record.context_window.value.tokens == 1_050_000
    assert record.max_input_tokens.value == 1_050_000
    assert record.max_output_tokens.status is FactStatus.UNKNOWN
    assert record.max_output_tokens.value is None


def test_deepseek_quickstart_captures_input_and_ignores_recommendation() -> None:
    catalog = catalog_from_together_sources(
        _representative_sources() + (_model_page("deepseek-v4-quickstart"),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = _record("together:deepseek-ai/DeepSeek-V4-Pro-0813", catalog=catalog)
    assert record.context_window.value.tokens == 1_000_000
    assert record.max_input_tokens.value == 1_000_000
    assert record.max_output_tokens.status is FactStatus.UNKNOWN
    assert 384_000 not in {
        record.context_window.value.tokens,
        record.max_input_tokens.value,
    }


def test_gpt_oss_quickstart_does_not_use_recommended_max_tokens() -> None:
    catalog = catalog_from_together_sources(
        (_model_page("gpt-oss"),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.provider_model_id == "openai/gpt-oss-120b"
    assert record.context_window.value.tokens == 128_000
    assert record.max_output_tokens.status is FactStatus.UNKNOWN
    ids = {item.provider_model_id for item in catalog.models}
    assert "openai/gpt-oss-20b" not in ids


def test_gpt_oss_serverless_and_quickstart_context_conflict() -> None:
    catalog = catalog_from_together_sources(
        (
            _source(SOURCE_ID_SERVERLESS, "serverless.md", SERVERLESS_URL),
            _model_page("gpt-oss"),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = _record("together:openai/gpt-oss-120b", catalog=catalog)
    assert record.context_window.status is FactStatus.CONFLICT
    tokens = {claim.value.tokens for claim in record.context_window.claims}
    assert tokens == {131072, 128_000}


def test_kimi_k26_quickstart_context_without_serverless_listing() -> None:
    catalog = catalog_from_together_sources(
        _representative_sources() + (_model_page("kimi-k2.6-quickstart"),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = _record("together:moonshotai/Kimi-K2.6", catalog=catalog)
    assert record.context_window.value.tokens == 256_000
    assert record.max_input_tokens.status is FactStatus.UNKNOWN
    assert record.max_output_tokens.status is FactStatus.UNKNOWN
    assert record.lifecycle.status is FactStatus.UNKNOWN


def test_serverless_and_model_page_context_conflict_is_preserved() -> None:
    catalog = catalog_from_together_sources(
        (
            _source(SOURCE_ID_SERVERLESS, "serverless.md", SERVERLESS_URL),
            _model_page("glm-5.2-quickstart", "conflict-page.md"),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = _record("together:zai-org/GLM-5.2", catalog=catalog)
    assert record.context_window.status is FactStatus.CONFLICT
    tokens = {claim.value.tokens for claim in record.context_window.claims}
    assert tokens == {512_000, 262_000}
    assert record.max_output_tokens.value == 128_000


def test_deprecations_only_model_context_is_not_verified() -> None:
    record = _record("together:nvidia/Nemotron-3-ultra-550b-a55b")
    assert record.context_window.status is FactStatus.NOT_VERIFIED
    assert record.context_window.value is None


def test_deprecations_only_unstated_dates_remain_unknown() -> None:
    catalog = _catalog(
        sources=(_source(SOURCE_ID_DEPRECATIONS, "deprecations.md", DEPRECATIONS_URL),)
    )
    record = _record("together:nvidia/Nemotron-3-ultra-550b-a55b", catalog=catalog)
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.retirement_at.status is FactStatus.KNOWN
    assert record.retirement_at.value == "2026-08-27"
    assert record.deprecated_at.status is FactStatus.UNKNOWN
    redirect = _record("together:deepseek-ai/DeepSeek-V3", catalog=catalog)
    assert redirect.replacement.status is FactStatus.KNOWN
    assert redirect.deprecated_at.status is FactStatus.UNKNOWN
    assert redirect.retirement_at.status is FactStatus.UNKNOWN
    assert redirect.lifecycle.status is FactStatus.UNKNOWN


def test_chat_vision_image_video_audio_embedding_moderation_modalities() -> None:
    chat = _record("together:openai/gpt-oss-120b")
    assert chat.input_modalities.value == (Modality.TEXT,)
    assert chat.output_modalities.value == (Modality.TEXT,)
    vision = _record("together:Qwen/Qwen3.5-9B")
    assert vision.input_modalities.value == (Modality.TEXT, Modality.IMAGE)
    assert vision.output_modalities.value == (Modality.TEXT,)
    image = _record("together:black-forest-labs/FLUX.1.1-pro")
    assert image.input_modalities.status is FactStatus.UNKNOWN
    assert image.output_modalities.value == (Modality.IMAGE,)
    video = _record("together:openai/sora-2")
    assert video.output_modalities.value == (Modality.VIDEO,)
    tts = _record("together:canopylabs/orpheus-3b-0.1-ft")
    assert tts.input_modalities.value == (Modality.TEXT,)
    assert tts.output_modalities.value == (Modality.AUDIO,)
    stt = _record("together:openai/whisper-large-v3")
    assert stt.input_modalities.value == (Modality.AUDIO,)
    assert stt.output_modalities.value == (Modality.TEXT,)
    embedding = _record("together:BAAI/bge-base-en-v1.5")
    assert embedding.input_modalities.value == (Modality.TEXT,)
    assert embedding.output_modalities.status is FactStatus.UNKNOWN
    moderation = _record("together:togethercomputer/together-safety-policy")
    assert moderation.input_modalities.value == (Modality.TEXT,)


def test_embedding_context_without_invented_dimension_field() -> None:
    record = _record("together:BAAI/bge-base-en-v1.5")
    assert record.context_window.value.tokens == 512
    assert not hasattr(record, "embedding_dimension")


def test_redirect_is_replacement_not_retirement_or_alias() -> None:
    observations = parse_together_sources(_representative_sources())
    redirects = [
        item for item in observations if item.provider_model_id == "deepseek-ai/DeepSeek-V3"
    ]
    assert redirects
    assert all(item.documented_serverless is False for item in redirects)
    assert all(item.lifecycle is None for item in redirects)
    assert all(item.api_access is None for item in redirects)
    assert any(item.replacements == ("deepseek-ai/DeepSeek-V3.1",) for item in redirects)

    record = _record("together:deepseek-ai/DeepSeek-V3")
    assert record.identity_kind is IdentityKind.CANONICAL
    assert record.lifecycle.status is FactStatus.UNKNOWN
    assert record.api_access.status is FactStatus.NOT_VERIFIED
    assert record.resolves_to.status is FactStatus.NOT_VERIFIED
    assert record.aliases.status is FactStatus.NOT_VERIFIED
    assert record.replacement.value[0].samyak_id == "together:deepseek-ai/DeepSeek-V3.1"
    assert record.replacement.value[0].provider_id == "together"


def test_redirect_target_does_not_inherit_source_lifecycle() -> None:
    source = _record("together:deepseek-ai/DeepSeek-V3")
    target = _record("together:deepseek-ai/DeepSeek-V3.1")
    assert source.lifecycle.status is FactStatus.UNKNOWN
    assert target.lifecycle.value is LifecycleState.ACTIVE
    assert target.api_access.value is True
    assert target.replacement.status is not FactStatus.KNOWN


def test_current_serverless_listing_establishes_active() -> None:
    record = _record("together:openai/gpt-oss-120b")
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.api_access.value is True


def test_unslashed_redirect_display_name_is_not_an_identity() -> None:
    catalog = _catalog()
    ids = {item.provider_model_id for item in catalog.models}
    assert "DeepSeek-V3" not in ids
    assert "DeepSeek-V3.1" not in ids


def test_scheduled_deprecation_before_removal_is_deprecated() -> None:
    record = _record("together:openai/gpt-oss-20b")
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert record.retirement_at.value == "2026-09-14"
    assert record.api_access.value is True
    assert record.replacement.value[0].samyak_id == "together:Qwen/Qwen3.5-9B"


def test_changelog_does_not_override_deprecations_replacement() -> None:
    record = _record("together:openai/gpt-oss-20b")
    assert record.replacement.value[0].provider_model_id == "Qwen/Qwen3.5-9B"
    assert record.replacement.provenance.source_id == SOURCE_ID_DEPRECATIONS


def test_retired_after_removal_date() -> None:
    later = _catalog(verified_at="2026-09-14T00:00:00+00:00")
    record = _record("together:openai/gpt-oss-20b", catalog=later)
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.api_access.value is False


def test_no_retirement_date_model_is_not_aged_to_retired() -> None:
    record = _record("together:openai/gpt-oss-120b")
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.api_access.value is True
    later = _catalog(verified_at="2027-01-01T00:00:00+00:00")
    aged = _record("together:openai/gpt-oss-120b", catalog=later)
    assert aged.lifecycle.value is LifecycleState.ACTIVE
    assert aged.api_access.value is True


def test_historical_deprecation_is_retired() -> None:
    record = _record("together:nvidia/Nemotron-3-ultra-550b-a55b")
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.api_access.value is False
    assert record.tool_calling.status is FactStatus.NOT_VERIFIED


def test_aging_and_future_removal_depend_on_verified_at() -> None:
    retired = _record("together:togethercomputer/aging-test")
    assert retired.lifecycle.value is LifecycleState.RETIRED
    earlier = _catalog(verified_at="2025-12-01T00:00:00+00:00")
    pending = _record("together:togethercomputer/aging-test", catalog=earlier)
    assert pending.lifecycle.value is LifecycleState.DEPRECATED
    future = _record("together:togethercomputer/future-test")
    assert future.lifecycle.value is LifecycleState.DEPRECATED
    assert future.retirement_at.value == "2027-01-01"
    assert future.api_access.value is True


def test_current_deprecation_event_wins_over_past() -> None:
    record = _record("together:togethercomputer/event-test")
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert record.replacement.value[0].samyak_id == "together:Qwen/Qwen3.5-9B"


def test_ambiguous_lifecycle_replacements_are_conflicts() -> None:
    record = _record("together:togethercomputer/ambiguous-test")
    assert record.replacement.status is FactStatus.CONFLICT
    values = {claim.value[0].provider_model_id for claim in record.replacement.claims}
    assert values == {"openai/gpt-oss-120b", "Qwen/Qwen3.5-9B"}


def test_conflicting_context_lengths_are_preserved() -> None:
    catalog = catalog_from_together_sources(
        (_source(SOURCE_ID_SERVERLESS, "conflict-context.md", SERVERLESS_URL),),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    record = catalog.models[0]
    assert record.provider_model_id == "togethercomputer/conflict-context"
    assert record.context_window.status is FactStatus.CONFLICT
    tokens = {claim.value.tokens for claim in record.context_window.claims}
    assert tokens == {1000, 2000}
    assert record.input_modalities.value == (Modality.TEXT, Modality.IMAGE)


def test_malformed_serverless_fails_closed() -> None:
    with pytest.raises(TogetherParseError, match="serverless"):
        parse_together_sources((_source(SOURCE_ID_SERVERLESS, "malformed.md", SERVERLESS_URL),))


def test_missing_model_id_fails_closed() -> None:
    with pytest.raises(TogetherParseError, match="did not list any serving model ids"):
        parse_together_sources((_source(SOURCE_ID_SERVERLESS, "missing-id.md", SERVERLESS_URL),))


def test_empty_sources_fail_closed() -> None:
    with pytest.raises(TogetherParseError, match="missing"):
        parse_together_sources(())


def test_provenance_and_content_hash() -> None:
    record = _record("together:openai/gpt-oss-120b")
    assert record.context_window.provenance is not None
    assert record.context_window.provenance.provider == "together"
    assert record.context_window.provenance.source_id == SOURCE_ID_SERVERLESS
    assert record.context_window.provenance.source_kind is SourceKind.PROVIDER_DOCS
    assert record.context_window.provenance.confidence is Confidence.HIGH
    body = (FIXTURES / "serverless.md").read_bytes()
    assert record.context_window.provenance.content_hash == content_hash_for_bytes(body)
    assert record.context_window.provenance.content_hash == (
        "sha256:" + hashlib.sha256(body).hexdigest()
    )
    redirect = _record("together:deepseek-ai/DeepSeek-V3")
    assert redirect.replacement.provenance.source_id == SOURCE_ID_DEPRECATIONS
    assert redirect.replacement.provenance.source_kind is SourceKind.PROVIDER_DEPRECATIONS


def test_freshness_and_round_trip() -> None:
    catalog = _catalog()
    assert catalog.freshness.generated_at == GENERATED_AT
    assert catalog.freshness.oldest_verified_at == VERIFIED_AT
    source_ids = {item.source_id for item in catalog.freshness.source_retrieved_at}
    assert SOURCE_ID_SERVERLESS in source_ids
    assert SOURCE_ID_DEPRECATIONS in source_ids
    assert SOURCE_ID_CHANGELOG in source_ids
    assert model_page_source_id("glm-5.2-quickstart") in source_ids
    restored = catalog_from_json(catalog_to_json(catalog))
    assert restored.to_dict() == catalog.to_dict()
