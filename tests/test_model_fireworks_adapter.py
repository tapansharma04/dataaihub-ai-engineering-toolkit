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
from samyak.model.providers.fireworks.observations import (
    FireworksLifecycle,
    FireworksLifecycleScope,
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
from samyak.model.records import EvidenceDisposition
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


_TAB_EMBEDDINGS_HASH = "7e7e9439a7e93e2edf04a62cfe56b7dbc0734f13c49614e532853e599e114619"
_TAB_ROWS = {
    "fireworks/qwen3-embedding-8b": ("Qwen3 Embedding 8B", 40_000, True),
    "fireworks/qwen3-embedding-4b": ("Qwen3 Embedding 4B", 40_000, False),
    "fireworks/qwen3-embedding-0p6b": ("Qwen3 Embedding 0.6B", 32_000, False),
    "fireworks/voyage-4-large": ("Voyage 4 Large", 40_000, False),
    "fireworks/voyage-4": ("Voyage 4", 40_000, False),
    "fireworks/voyage-4-lite": ("Voyage 4 Lite", 40_000, False),
    "fireworks/voyage-4-nano": ("Voyage 4 Nano", 40_000, False),
    "fireworks/voyage-multimodal-3-5": ("Voyage Multimodal 3.5", 32_000, False),
    "fireworks/qwen3-reranker-8b": ("Qwen3 Reranker 8B", 40_000, True),
    "fireworks/qwen3-reranker-4b": ("Qwen3 Reranker 4B", 40_000, False),
    "fireworks/qwen3-reranker-0p6b": ("Qwen3 Reranker 0.6B", 40_000, False),
    "fireworks/voyage-rerank-2-5": ("Voyage ReRank 2.5", 32_000, False),
}
_EMBEDDING_IDS = frozenset(
    {
        "fireworks/qwen3-embedding-8b",
        "fireworks/qwen3-embedding-4b",
        "fireworks/qwen3-embedding-0p6b",
        "fireworks/voyage-4-large",
        "fireworks/voyage-4",
        "fireworks/voyage-4-lite",
        "fireworks/voyage-4-nano",
        "fireworks/voyage-multimodal-3-5",
    }
)
_RERANK_IDS = frozenset(
    {
        "fireworks/qwen3-reranker-8b",
        "fireworks/qwen3-reranker-4b",
        "fireworks/qwen3-reranker-0p6b",
        "fireworks/voyage-rerank-2-5",
    }
)


def test_ordinary_embeddings_fixture_still_reads_unquoted_tables() -> None:
    observations = parse_fireworks_sources(
        (_source(SOURCE_ID_EMBEDDINGS, "embeddings.md", EMBEDDINGS_URL),)
    )
    ids = {item.provider_model_id for item in observations}
    assert "fireworks/qwen3-embedding-8b" in ids
    assert "fireworks/qwen3-embedding-4b" in ids
    assert "fireworks/voyage-4" in ids
    assert "fireworks/voyage-rerank-2-5" in ids
    assert "fireworks/qwen3-reranker-8b" in ids
    record = _record("fireworks:fireworks/qwen3-embedding-8b")
    assert record.display_name.value == "Qwen3 Embedding 8B"
    assert record.context_window.value.tokens == 40_000


def test_mintlify_tab_tables_restore_blockquoted_embedding_rows() -> None:
    body = (FIXTURES / "embeddings-mintlify-tabs.md").read_bytes()
    assert hashlib.sha256(body).hexdigest() == _TAB_EMBEDDINGS_HASH
    assert len(body) == 22212
    catalog = catalog_from_fireworks_sources(
        (
            captured_markdown(
                source_id=SOURCE_ID_EMBEDDINGS,
                source_url=EMBEDDINGS_URL,
                body=body.decode("utf-8"),
                retrieved_at=RETRIEVED_AT,
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    by_id = {record.provider_model_id: record for record in catalog.models}
    assert len(by_id) == len(catalog.models)
    assert len({model_id.casefold() for model_id in by_id}) == len(by_id)
    assert _EMBEDDING_IDS.isdisjoint(_RERANK_IDS)
    assert set(by_id) >= _EMBEDDING_IDS | _RERANK_IDS
    for model_id, (display_name, tokens, serverless) in _TAB_ROWS.items():
        record = by_id[model_id]
        assert record.provider_model_id == model_id
        assert record.display_name.status is FactStatus.KNOWN
        assert record.display_name.value == display_name
        assert record.context_window.status is FactStatus.KNOWN
        assert record.context_window.value.tokens == tokens
        assert record.context_window.value.kind is ContextWindowKind.UNKNOWN
        assert record.input_modalities.value == (Modality.TEXT,)
        assert record.output_modalities.status is not FactStatus.KNOWN
        if serverless:
            assert record.api_access.value is True
        else:
            assert record.api_access.status is not FactStatus.KNOWN or (
                record.api_access.value is not True
            )
        assert model_id in _EMBEDDING_IDS or model_id in _RERANK_IDS
    assert by_id["fireworks/voyage-4"].display_name.value == "Voyage 4"
    assert by_id["fireworks/voyage-rerank-2-5"].display_name.value == "Voyage ReRank 2.5"
    assert "thenlper/gte-base" not in by_id
    assert "nomic-ai/nomic-embed-text-v1.5" not in by_id
    assert "WhereIsAI/UAE-Large-V1" not in by_id
    assert "mixedbread-ai/mxbai-embed-large-v1" not in by_id


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


def test_changelog_lifecycle_links_use_only_supported_fireworks_model_urls() -> None:
    source = captured_markdown(
        source_id=SOURCE_ID_CHANGELOG,
        source_url=CHANGELOG_URL,
        retrieved_at=RETRIEVED_AT,
        body="""# Changelog

## Current

[Target](https://app.fireworks.ai/models/fireworks/target) and
[Second target](https://app.fireworks.ai/models/fireworks/second-target) and
`accounts/fireworks/models/target` are deprecated from serverless. Migrate to
[Replacement](https://app.fireworks.ai/models/fireworks/replacement).

[Retired](https://app.fireworks.ai/models/fireworks/retired) is no longer enabled
on January 1, 2026.

[Other publisher](https://app.fireworks.ai/models/acme/other),
[Wrong path](https://app.fireworks.ai/deployments/fireworks/wrong),
[Wrong host](https://example.com/models/fireworks/wrong), and
[Query](https://app.fireworks.ai/models/fireworks/query?preview=1) are deprecated
from serverless.

Display Name is deprecated from serverless.

[Outside context](https://app.fireworks.ai/models/fireworks/outside) is newly available.
""",
    )
    observations = parse_fireworks_sources((source,))
    by_id = {item.provider_model_id: item for item in observations}

    assert set(by_id) == {
        "accounts/fireworks/models/target",
        "accounts/fireworks/models/second-target",
        "accounts/fireworks/models/retired",
    }
    target = by_id["accounts/fireworks/models/target"]
    assert target.lifecycle is FireworksLifecycle.DEPRECATED
    assert target.replacements == ("accounts/fireworks/models/replacement",)
    assert target.source.source_id == SOURCE_ID_CHANGELOG
    assert target.source.source_kind is SourceKind.PROVIDER_CHANGELOG
    assert by_id["accounts/fireworks/models/second-target"].replacements == (
        "accounts/fireworks/models/replacement",
    )
    retired = by_id["accounts/fireworks/models/retired"]
    assert retired.lifecycle is FireworksLifecycle.RETIRED
    assert retired.retirement_at == "2026-01-01"


def test_changelog_event_blocks_keep_targets_dates_and_replacements_separate() -> None:
    source = captured_markdown(
        source_id=SOURCE_ID_CHANGELOG,
        source_url=CHANGELOG_URL,
        retrieved_at=RETRIEVED_AT,
        body="""# Changelog

# Upcoming Serverless deprecation

The affected models will be decommissioned on December 1, 2026.

## Affected models

* [Model A](https://app.fireworks.ai/models/fireworks/model-a)
* [Model B](https://app.fireworks.ai/models/fireworks/model-b)

## Recommended migrations

* Migrate from [Model A](https://app.fireworks.ai/models/fireworks/model-a) to [Model C](https://app.fireworks.ai/models/fireworks/model-c).
* [Model B](https://app.fireworks.ai/models/fireworks/model-b) — migrate to [Model C](https://app.fireworks.ai/models/fireworks/model-c).

## New models on Serverless

* [Model D](https://app.fireworks.ai/models/fireworks/model-d) is now available.

# Historical removal

[Model E](https://app.fireworks.ai/models/fireworks/model-e) is no longer enabled
on January 2, 2025.
""",
    )
    observations = parse_fireworks_sources((source,))
    by_id = {item.provider_model_id: item for item in observations}

    assert set(by_id) == {
        "accounts/fireworks/models/model-a",
        "accounts/fireworks/models/model-b",
        "accounts/fireworks/models/model-e",
    }
    for model_id in ("accounts/fireworks/models/model-a", "accounts/fireworks/models/model-b"):
        item = by_id[model_id]
        assert item.lifecycle is FireworksLifecycle.DEPRECATED
        assert item.retirement_at == "2026-12-01"
        assert item.replacements == ("accounts/fireworks/models/model-c",)
    assert by_id["accounts/fireworks/models/model-e"].lifecycle is FireworksLifecycle.RETIRED
    assert by_id["accounts/fireworks/models/model-e"].retirement_at == "2025-01-02"


def test_changelog_migration_bullets_require_strict_identities_and_preserve_direction() -> None:
    source = captured_markdown(
        source_id=SOURCE_ID_CHANGELOG,
        source_url=CHANGELOG_URL,
        retrieved_at=RETRIEVED_AT,
        body="""# Changelog

# Serverless deprecation: MiniMax M2.7, GPT OSS 20B, Kimi K2.6 Turbo/Fast

The following models are deprecated from serverless effective August 27, 2026.

## Recommended migrations

* **[MiniMax M2.7](https://app.fireworks.ai/models/fireworks/minimax-m2p7)**
  — migrate to **[MiniMax M3](https://app.fireworks.ai/models/fireworks/minimax-m3)**
* **[GPT OSS 20B](https://app.fireworks.ai/models/fireworks/gpt-oss-20b)**
  — migrate to **[GPT OSS 120B](https://app.fireworks.ai/models/fireworks/gpt-oss-120b)**
  or **[Qwen3 8B](https://app.fireworks.ai/models/fireworks/qwen3-8b)**
* **[Kimi K2.6 Turbo / Fast](https://app.fireworks.ai/models/fireworks/kimi-k2p6)**
  — migrate to **[Kimi K2.6](https://app.fireworks.ai/models/fireworks/kimi-k2p6)**
* Migrate from `accounts/fireworks/models/deepseek-v4-pro` to [DeepSeek V4 Pro (0813)](https://app.fireworks.ai/models/fireworks/deepseek-v4-pro-0813)
* Migrate new workloads to [Replacement only](https://app.fireworks.ai/models/fireworks/replacement-only)
* [Ambiguous A](https://app.fireworks.ai/models/fireworks/ambiguous-a) and
  [Ambiguous B](https://app.fireworks.ai/models/fireworks/ambiguous-b) have migration guidance.

## New models on Serverless

* [New model](https://app.fireworks.ai/models/fireworks/new-model) is now available.

# Historical removal

[Old model](https://app.fireworks.ai/models/fireworks/old-model) is no longer available.
Use [New replacement](https://app.fireworks.ai/models/fireworks/new-replacement).

# Future removal

[Future model](https://app.fireworks.ai/models/fireworks/future-model) is scheduled for removal
on December 1, 2026.
""",
    )
    observations = parse_fireworks_sources((source,))
    by_id = {item.provider_model_id: item for item in observations}

    assert set(by_id) == {
        "accounts/fireworks/models/minimax-m2p7",
        "accounts/fireworks/models/gpt-oss-20b",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/old-model",
        "accounts/fireworks/models/future-model",
    }
    assert "Turbo/Fast" not in by_id
    assert "accounts/fireworks/models/replacement-only" not in by_id
    assert "accounts/fireworks/models/ambiguous-a" not in by_id
    assert "accounts/fireworks/models/new-model" not in by_id
    assert by_id["accounts/fireworks/models/minimax-m2p7"].replacements == (
        "accounts/fireworks/models/minimax-m3",
    )
    assert by_id["accounts/fireworks/models/gpt-oss-20b"].replacements == (
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/qwen3-8b",
    )
    assert by_id["accounts/fireworks/models/deepseek-v4-pro"].replacements == (
        "accounts/fireworks/models/deepseek-v4-pro-0813",
    )
    old = by_id["accounts/fireworks/models/old-model"]
    assert old.lifecycle is FireworksLifecycle.RETIRED
    assert old.replacements == ("accounts/fireworks/models/new-replacement",)
    future = by_id["accounts/fireworks/models/future-model"]
    assert future.lifecycle is FireworksLifecycle.DEPRECATED
    assert future.retirement_at == "2026-12-01"

    catalog = catalog_from_fireworks_sources(
        (source,),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    assert {record.samyak_id for record in catalog.models} == {
        "fireworks:accounts/fireworks/models/minimax-m2p7",
        "fireworks:accounts/fireworks/models/gpt-oss-20b",
        "fireworks:accounts/fireworks/models/deepseek-v4-pro",
        "fireworks:accounts/fireworks/models/old-model",
        "fireworks:accounts/fireworks/models/future-model",
    }
    assert all("Turbo/Fast" not in record.samyak_id for record in catalog.models)


@pytest.mark.parametrize(
    ("heading", "expected"),
    (
        ("Recommended migrations", True),
        ("**Recommended migrations**", True),
        ("__Recommended migrations__", True),
        ("*Recommended migrations*", True),
        ("_Recommended migrations_", True),
        ("  **Recommended migrations**  ", True),
        ("**Required action**", True),
        ("**New models**", False),
    ),
)
def test_changelog_emphasized_headings_keep_migration_bullets_scoped(
    heading: str, expected: bool
) -> None:
    source = captured_markdown(
        source_id=SOURCE_ID_CHANGELOG,
        source_url=CHANGELOG_URL,
        retrieved_at=RETRIEVED_AT,
        body=f"""# Changelog

# Upcoming deprecation

The affected models will be decommissioned on December 1, 2026.

## {heading}

* [Target](https://app.fireworks.ai/models/fireworks/target) — migrate to
  [Replacement](https://app.fireworks.ai/models/fireworks/replacement).

# Independent removal

[Historical](https://app.fireworks.ai/models/fireworks/historical) is no longer available
on January 2, 2025.
""",
    )
    observations = parse_fireworks_sources((source,))
    by_id = {item.provider_model_id: item for item in observations}

    assert by_id["accounts/fireworks/models/historical"].retirement_at == "2025-01-02"
    if expected:
        target = by_id["accounts/fireworks/models/target"]
        assert target.replacements == ("accounts/fireworks/models/replacement",)
        assert target.retirement_at == "2026-12-01"
    else:
        assert "accounts/fireworks/models/target" not in by_id
        assert "accounts/fireworks/models/replacement" not in by_id


def test_changelog_adjacent_migration_paragraph_requires_one_immediate_target() -> None:
    source = captured_markdown(
        source_id=SOURCE_ID_CHANGELOG,
        source_url=CHANGELOG_URL,
        retrieved_at=RETRIEVED_AT,
        body="""# Changelog

# Upcoming Serverless Training deprecation: DeepSeek V4 Flash (0731)

On November 2, 2026, [DeepSeek V4 Flash (0731)](https://app.fireworks.ai/models/fireworks/deepseek-v4-flash-0731)
will be removed from the shared Serverless Training pool.

If you train DeepSeek V4 Flash (0731), migrate to
[DeepSeek V4.1 Flash 1017](https://app.fireworks.ai/models/fireworks/deepseek-v4p1-flash-1017)
before November 2, 2026.

# Unrelated paragraph

[Unrelated target](https://app.fireworks.ai/models/fireworks/unrelated-target) is deprecated
from serverless.

This paragraph is unrelated.

Migrate to [Unrelated replacement](https://app.fireworks.ai/models/fireworks/unrelated-replacement).

# Multiple targets

[First](https://app.fireworks.ai/models/fireworks/first) and
[Second](https://app.fireworks.ai/models/fireworks/second) are deprecated from serverless.

Migrate to [Ambiguous replacement](https://app.fireworks.ai/models/fireworks/ambiguous-replacement).

# New models

[New model](https://app.fireworks.ai/models/fireworks/new-model) is now available.

Use [New replacement](https://app.fireworks.ai/models/fireworks/new-replacement) for migration.
""",
    )
    observations = parse_fireworks_sources((source,))
    by_id = {item.provider_model_id: item for item in observations}

    training = by_id["accounts/fireworks/models/deepseek-v4-flash-0731"]
    assert training.replacements == ("accounts/fireworks/models/deepseek-v4p1-flash-1017",)
    assert training.retirement_at == "2026-11-02"
    assert training.source.source_id == SOURCE_ID_CHANGELOG
    assert by_id["accounts/fireworks/models/unrelated-target"].replacements is None
    assert "accounts/fireworks/models/first" not in by_id
    assert "accounts/fireworks/models/second" not in by_id
    assert "accounts/fireworks/models/unrelated-replacement" not in by_id
    assert "accounts/fireworks/models/ambiguous-replacement" not in by_id
    assert "accounts/fireworks/models/new-model" not in by_id
    assert "accounts/fireworks/models/new-replacement" not in by_id


def test_changelog_workload_scoped_adjacent_migration_requires_matching_scope() -> None:
    source = captured_markdown(
        source_id=SOURCE_ID_CHANGELOG,
        source_url=CHANGELOG_URL,
        retrieved_at=RETRIEVED_AT,
        body="""# Changelog

# Serverless Training deprecation

[DeepSeek V4 Flash (0731)](https://app.fireworks.ai/models/fireworks/deepseek-v4-flash-0731)
is removed from the shared Serverless Training pool effective November 2, 2026.

Migrate Serverless Training workloads to
[DeepSeek V4.1 Flash 1017](https://app.fireworks.ai/models/fireworks/deepseek-v4p1-flash-1017).

# Inference deprecation

[Inference target](https://app.fireworks.ai/models/fireworks/inference-target)
is deprecated from serverless effective December 1, 2026.

Migrate inference workloads to
[Inference replacement](https://app.fireworks.ai/models/fireworks/inference-replacement).

# Scope mismatch

[Training target](https://app.fireworks.ai/models/fireworks/training-target)
is deprecated from Serverless Training effective December 2, 2026.

Migrate inference workloads to
[Mismatched replacement](https://app.fireworks.ai/models/fireworks/mismatched-replacement).

# Unsupported destination

[Unsupported target](https://app.fireworks.ai/models/fireworks/unsupported-target)
is deprecated from Serverless Training effective December 3, 2026.

Migrate Serverless Training workloads to [Unsupported](https://example.com/model).

# Self mapping

[Self target](https://app.fireworks.ai/models/fireworks/self-target)
is deprecated from Serverless Training effective December 4, 2026.

Migrate Serverless Training workloads to
[Self target](https://app.fireworks.ai/models/fireworks/self-target).
""",
    )

    observations = parse_fireworks_sources((source,))
    by_id = {item.provider_model_id: item for item in observations}

    training = by_id["accounts/fireworks/models/deepseek-v4-flash-0731"]
    assert training.replacements == ("accounts/fireworks/models/deepseek-v4p1-flash-1017",)
    assert training.retirement_at == "2026-11-02"
    assert training.source.source_id == SOURCE_ID_CHANGELOG
    assert by_id["accounts/fireworks/models/inference-target"].replacements == (
        "accounts/fireworks/models/inference-replacement",
    )
    assert by_id["accounts/fireworks/models/training-target"].replacements is None
    assert by_id["accounts/fireworks/models/unsupported-target"].replacements is None
    assert by_id["accounts/fireworks/models/self-target"].replacements is None
    assert "accounts/fireworks/models/mismatched-replacement" not in by_id
    assert "accounts/fireworks/models/inference-replacement" not in {
        item.provider_model_id for item in observations if item.lifecycle is not None
    }


def test_scoped_changelog_lifecycle_evidence_does_not_change_global_facts() -> None:
    source = captured_markdown(
        source_id=SOURCE_ID_CHANGELOG,
        source_url=CHANGELOG_URL,
        retrieved_at=RETRIEVED_AT,
        body="""# Changelog

# Serverless Training removal

[Scoped training](https://app.fireworks.ai/models/fireworks/scoped-training) is removed
from the Serverless Training pool on January 2, 2025.

Migrate Serverless Training workloads to
[Scoped replacement](https://app.fireworks.ai/models/fireworks/scoped-replacement).

# Endpoint removal

[Scoped endpoint](https://app.fireworks.ai/models/fireworks/scoped-endpoint) is no longer available
on US endpoints on January 3, 2025.

# Ambiguous removal

[Ambiguous](https://app.fireworks.ai/models/fireworks/ambiguous) is no longer available
on January 4, 2025.

# Global retirement

[Global](https://app.fireworks.ai/models/fireworks/global) is no longer available on the
Fireworks platform on January 5, 2025. Migrate to
[Global replacement](https://app.fireworks.ai/models/fireworks/global-replacement).

# Global retirement after scoped history

[Shared](https://app.fireworks.ai/models/fireworks/shared) is no longer available on
Serverless on January 6, 2025.

[Shared](https://app.fireworks.ai/models/fireworks/shared) is no longer available on the
Fireworks platform on January 7, 2025.
""",
    )

    observations = parse_fireworks_sources((source,))
    by_id = {item.provider_model_id: item for item in observations}
    assert by_id["accounts/fireworks/models/scoped-training"].lifecycle_scope is (
        FireworksLifecycleScope.DEPLOYMENT_SCOPED
    )
    assert by_id["accounts/fireworks/models/scoped-training"].replacements == (
        "accounts/fireworks/models/scoped-replacement",
    )
    assert by_id["accounts/fireworks/models/ambiguous"].lifecycle_scope is (
        FireworksLifecycleScope.AMBIGUOUS
    )
    assert by_id["accounts/fireworks/models/global"].lifecycle_scope is (
        FireworksLifecycleScope.GLOBAL
    )

    catalog = catalog_from_fireworks_sources(
        (source,),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    records = {item.provider_model_id: item for item in catalog.models}
    for model_id in (
        "accounts/fireworks/models/scoped-training",
        "accounts/fireworks/models/scoped-endpoint",
        "accounts/fireworks/models/ambiguous",
    ):
        record = records[model_id]
        assert record.lifecycle.status is FactStatus.UNKNOWN
        assert record.retirement_at.status is FactStatus.NOT_VERIFIED
        assert record.replacement.status is FactStatus.NOT_VERIFIED
        assert record.api_access.status is FactStatus.NOT_VERIFIED
        assert record.evidence_disposition is EvidenceDisposition.HISTORICAL_REFERENCE_ONLY

    global_record = records["accounts/fireworks/models/global"]
    assert global_record.lifecycle.value is LifecycleState.RETIRED
    assert global_record.retirement_at.value == "2025-01-05"
    assert global_record.replacement.value[0].provider_model_id == (
        "accounts/fireworks/models/global-replacement"
    )
    shared = records["accounts/fireworks/models/shared"]
    assert shared.lifecycle.value is LifecycleState.RETIRED
    assert shared.retirement_at.value == "2025-01-07"
    assert global_record.evidence_disposition is EvidenceDisposition.HISTORICAL_REFERENCE_ONLY


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
