"""Local overlay update and CLI tests. No live network."""

from __future__ import annotations

import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from samyak.cli import main
from samyak.model.catalog import (
    CatalogFreshness,
    CatalogNotice,
    CatalogOverlay,
    FreshnessStatus,
    build_catalog,
)
from samyak.model.errors import CatalogValidationError
from samyak.model.facts import (
    Confidence,
    Fact,
    FactStatus,
    LifecycleState,
    Provenance,
    SourceKind,
)
from samyak.model.http import FixtureHop, FixtureTransport
from samyak.model.identity import ModelIdentity
from samyak.model.lifecycle import LifecycleChangeType
from samyak.model.providers.anthropic.fetch import anthropic_docs_policy
from samyak.model.providers.anthropic.sources import (
    DEPRECATIONS_URL as ANTHROPIC_DEPRECATIONS_URL,
)
from samyak.model.providers.anthropic.sources import (
    MODELS_INDEX_URL as ANTHROPIC_INDEX_URL,
)
from samyak.model.providers.anthropic.sources import model_page_url as anthropic_model_page_url
from samyak.model.providers.fireworks.fetch import fireworks_docs_policy
from samyak.model.providers.fireworks.sources import (
    CHANGELOG_URL as FIREWORKS_CHANGELOG_URL,
)
from samyak.model.providers.fireworks.sources import (
    EMBEDDINGS_URL as FIREWORKS_EMBEDDINGS_URL,
)
from samyak.model.providers.fireworks.sources import (
    KIMI_K2_URL as FIREWORKS_KIMI_URL,
)
from samyak.model.providers.fireworks.sources import (
    SERVING_PATHS_URL as FIREWORKS_SERVING_URL,
)
from samyak.model.providers.fireworks.sources import (
    TEXT_MODELS_URL as FIREWORKS_TEXT_URL,
)
from samyak.model.providers.fireworks.sources import (
    TOOL_CALLING_URL as FIREWORKS_TOOL_URL,
)
from samyak.model.providers.fireworks.sources import (
    VISION_MODELS_URL as FIREWORKS_VISION_URL,
)
from samyak.model.providers.google.fetch import google_docs_policy
from samyak.model.providers.google.sources import (
    DEPRECATIONS_URL as GOOGLE_DEPRECATIONS_URL,
)
from samyak.model.providers.google.sources import (
    GEMINI_3_URL as GOOGLE_GEMINI_3_URL,
)
from samyak.model.providers.google.sources import (
    MODELS_INDEX_URL as GOOGLE_INDEX_URL,
)
from samyak.model.providers.google.sources import model_page_url as google_model_page_url
from samyak.model.providers.openai.fetch import (
    OpenAIRefreshResult,
    openai_docs_policy,
    refresh_openai_catalog,
)
from samyak.model.providers.openai.sources import (
    DEPRECATIONS_URL,
    MODELS_INDEX_URL,
    model_page_url,
)
from samyak.model.providers.together.fetch import together_docs_policy
from samyak.model.providers.together.sources import (
    CHANGELOG_URL as TOGETHER_CHANGELOG_URL,
)
from samyak.model.providers.together.sources import (
    DEPRECATIONS_URL as TOGETHER_DEPRECATIONS_URL,
)
from samyak.model.providers.together.sources import (
    SERVERLESS_URL as TOGETHER_SERVERLESS_URL,
)
from samyak.model.providers.together.sources import (
    optional_source_descriptors as together_optional_source_descriptors,
)
from samyak.model.records import ModelRecord
from samyak.model.serialize import catalog_from_json
from samyak.model.store import FileCatalogStore
from samyak.model.update import (
    _retain_records_for_failed_sources,
    update_anthropic_catalog,
    update_fireworks_catalog,
    update_google_catalog,
    update_openai_catalog,
    update_together_catalog,
)

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "openai"
ANTHROPIC_FIXTURES = Path(__file__).parent / "fixtures" / "models" / "anthropic"
GOOGLE_FIXTURES = Path(__file__).parent / "fixtures" / "models" / "google"
FIREWORKS_FIXTURES = Path(__file__).parent / "fixtures" / "models" / "fireworks"
TOGETHER_FIXTURES = Path(__file__).parent / "fixtures" / "models" / "together"
STAMP = "2026-09-10T18:00:00+00:00"
INDEX_ONLY_BODY = b"# Index Only\n\nModel ID: `index-only`\n"


@pytest.fixture(autouse=True)
def deny_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Model Intelligence update tests must not use the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("urllib.request.urlopen", blocked)


def _md(relative: str, **kwargs: object) -> FixtureHop:
    return FixtureHop(body=(FIXTURES / relative).read_bytes(), **kwargs)  # type: ignore[arg-type]


def _transport(
    pages: dict[str, FixtureHop | tuple[FixtureHop, ...]],
    *,
    clock: object | None = None,
) -> FixtureTransport:
    return FixtureTransport(
        pages,
        policy=openai_docs_policy(),
        clock=clock if callable(clock) else (lambda: STAMP),
    )


def _official_pages(**overrides: FixtureHop) -> dict[str, FixtureHop]:
    pages = {
        MODELS_INDEX_URL: _md("models.md"),
        DEPRECATIONS_URL: _md("deprecations.md"),
        model_page_url("gpt-5.6-sol"): _md("models/gpt-5.6-sol.md"),
        model_page_url("gpt-5.6"): _md("models/gpt-5.6.md"),
        model_page_url("gpt-4.5-preview"): _md("models/gpt-4.5-preview.md"),
        model_page_url("sparse-test"): _md("models/sparse-test.md"),
        model_page_url("index-only"): FixtureHop(body=INDEX_ONLY_BODY),
    }
    pages.update(overrides)
    return pages


def _clock() -> str:
    return STAMP


def test_complete_successful_update(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert result.ok is True
    assert result.committed is True
    assert result.partial is False
    assert result.error is None
    assert result.catalog is not None
    assert result.catalog.freshness.overlay is CatalogOverlay.USER_CACHE
    assert result.catalog.generated_at == STAMP
    assert result.provider_model_count == len(result.catalog.models)
    assert result.provider_model_count == result.catalog_model_count
    assert result.provider_model_count > 0
    loaded = store.load()
    assert loaded.to_dict() == result.catalog.to_dict()
    assert loaded.generated_at == STAMP
    round_trip = catalog_from_json(store.catalog_path.read_text(encoding="utf-8"))
    assert round_trip.to_dict() == loaded.to_dict()
    assert not (tmp_path / "cache" / "runs").exists()
    assert result.lifecycle_changes == ()
    payload = json.loads(store.catalog_path.read_text(encoding="utf-8"))
    assert payload["catalog_schema_version"] == 1
    assert "lifecycle_changes" not in payload


def test_partial_model_page_failures_commit_with_notices(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    pages = _official_pages()
    del pages[model_page_url("index-only")]
    result = update_openai_catalog(
        transport=_transport(pages),
        store=store,
        clock=_clock,
    )
    assert result.ok is True
    assert result.committed is True
    assert result.partial is True
    assert result.catalog is not None
    codes = {notice.code for notice in result.catalog.notices}
    assert "PARTIAL_MODEL_PAGES" in codes
    assert "SOURCE_FETCH_FAILED:openai-docs-model-page:index-only" in codes
    loaded = store.load()
    assert loaded.notices == result.catalog.notices
    assert any(item.provider_model_id == "index-only" for item in loaded.models)


def test_partial_refresh_does_not_remove_index_established_model(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert first.committed is True
    assert first.partial is False
    assert first.catalog is not None
    assert any(item.provider_model_id == "index-only" for item in first.catalog.models)

    pages = _official_pages()
    del pages[model_page_url("index-only")]
    second = update_openai_catalog(
        transport=_transport(pages),
        store=store,
        clock=_clock,
    )
    assert second.ok is True
    assert second.committed is True
    assert second.partial is True
    assert second.catalog is not None
    assert any(item.provider_model_id == "index-only" for item in second.catalog.models)
    removed = [
        item for item in second.lifecycle_changes if item.change_type is LifecycleChangeType.REMOVED
    ]
    assert "openai:index-only" not in {item.samyak_id for item in removed}
    assert removed == []


def test_informational_notices_do_not_mark_update_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    notice = CatalogNotice(code="INFO_AS_OF", message="Snapshot is labeled as-of")
    baseline = refresh_openai_catalog(
        _transport(_official_pages()),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert baseline.ok is True
    assert baseline.partial is False
    assert baseline.catalog is not None
    catalog = build_catalog(
        models=baseline.catalog.models,
        generated_at=baseline.catalog.generated_at,
        freshness=baseline.catalog.freshness,
        notices=(notice,),
        version=baseline.catalog.version,
    )

    def fake_refresh(*args: object, **kwargs: object) -> OpenAIRefreshResult:
        return OpenAIRefreshResult(
            ok=True,
            catalog=catalog,
            error=None,
            notices=(notice,),
            partial=False,
        )

    monkeypatch.setattr("samyak.model.update.refresh_openai_catalog", fake_refresh)
    result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert result.ok is True
    assert result.committed is True
    assert result.partial is False
    assert result.notices == (notice,)
    loaded = store.load()
    assert loaded.notices == (notice,)


def test_required_index_failure_preserves_previous(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert first.committed is True
    previous = store.catalog_path.read_bytes()
    pages = _official_pages()
    pages[MODELS_INDEX_URL] = FixtureHop(status=500, body=b"error")
    result = update_openai_catalog(
        transport=_transport(pages),
        store=store,
        clock=lambda: "2026-09-10T19:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert result.partial is False
    assert result.catalog is None
    assert result.error is not None
    assert "500" in result.error
    assert store.catalog_path.read_bytes() == previous


def test_required_deprecations_failure_preserves_previous(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    previous = store.catalog_path.read_bytes()
    pages = _official_pages()
    pages[DEPRECATIONS_URL] = FixtureHop(status=404, body=b"missing")
    result = update_openai_catalog(
        transport=_transport(pages),
        store=store,
        clock=lambda: "2026-09-10T19:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert "404" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous


def test_parse_failure_preserves_previous(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    previous = store.catalog_path.read_bytes()
    pages = _official_pages()
    pages[MODELS_INDEX_URL] = _md("malformed.md")
    result = update_openai_catalog(
        transport=_transport(pages),
        store=store,
        clock=lambda: "2026-09-10T19:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert store.catalog_path.read_bytes() == previous


def test_validation_failure_preserves_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    previous = store.catalog_path.read_bytes()

    def boom(*args: object, **kwargs: object) -> None:
        raise CatalogValidationError("duplicate canonical identity")

    monkeypatch.setattr(
        "samyak.model.providers.openai.fetch.catalog_from_openai_sources",
        boom,
    )
    result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=lambda: "2026-09-10T19:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert "duplicate canonical identity" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous


def test_first_update_creates_catalog_without_prior_file(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    assert store.exists() is False
    result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert result.previous_existed is False
    assert store.exists() is True
    assert result.catalog_path == store.catalog_path
    assert result.lifecycle_changes == ()


def test_identical_second_update_has_no_lifecycle_changes(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert first.committed is True
    second = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert second.committed is True
    assert second.previous_existed is True
    assert second.lifecycle_changes == ()


def test_update_reports_lifecycle_change_against_previous_overlay(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert first.catalog is not None
    target = "openai:gpt-4.5-preview"
    rewritten_models = []
    for record in first.catalog.models:
        if record.samyak_id != target:
            rewritten_models.append(record)
            continue
        rewritten_models.append(
            replace(
                record,
                lifecycle=Fact(
                    status=FactStatus.KNOWN,
                    value=LifecycleState.ACTIVE,
                    provenance=record.lifecycle.provenance,
                ),
            )
        )
    store.save(
        build_catalog(
            models=rewritten_models,
            generated_at=first.catalog.generated_at,
            freshness=first.catalog.freshness,
            notices=first.catalog.notices,
            version=first.catalog.version,
        )
    )
    second = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert second.committed is True
    changed = [
        item
        for item in second.lifecycle_changes
        if item.samyak_id == target and item.change_type is LifecycleChangeType.LIFECYCLE_CHANGED
    ]
    assert len(changed) == 1
    assert changed[0].previous is not None
    assert changed[0].current is not None
    assert changed[0].previous.lifecycle.value is LifecycleState.ACTIVE
    assert changed[0].current.lifecycle.value is LifecycleState.DEPRECATED


def _patch_cli_transport(
    monkeypatch: pytest.MonkeyPatch,
    pages: dict[str, FixtureHop],
) -> None:
    monkeypatch.setattr(
        "samyak.model.update.openai_https_transport",
        lambda clock=None: _transport(pages, clock=clock),
    )
    monkeypatch.setattr("samyak.model.update.utc_now_iso", lambda: STAMP)


def test_cli_model_update_openai_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], samyak_cache_dir: Path
) -> None:
    _patch_cli_transport(monkeypatch, _official_pages())
    code = main(["model", "update", "openai"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Updated OpenAI model catalog." in captured.out
    assert "Status: complete" in captured.out
    assert "Models:" in captured.out
    assert "Lifecycle changes" not in captured.out
    assert str(samyak_cache_dir / "models" / "catalog.json") in captured.out
    store = FileCatalogStore()
    assert store.exists() is True
    catalog = store.load()
    assert catalog.generated_at == STAMP
    assert catalog.freshness.overlay is CatalogOverlay.USER_CACHE


def test_cli_model_update_partial_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pages = _official_pages()
    del pages[model_page_url("index-only")]
    _patch_cli_transport(monkeypatch, pages)
    code = main(["model", "update", "openai"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Status: partial" in captured.out
    assert "Some model pages could not be retrieved." in captured.out
    assert "Notices:" in captured.out
    store = FileCatalogStore()
    assert store.load().notices


def test_cli_required_source_failure_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_cli_transport(monkeypatch, _official_pages())
    assert main(["model", "update", "openai"]) == 0
    previous = FileCatalogStore().catalog_path.read_bytes()
    pages = _official_pages()
    pages[MODELS_INDEX_URL] = FixtureHop(status=500, body=b"error")
    _patch_cli_transport(monkeypatch, pages)
    code = main(["model", "update", "openai"])
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "was not changed" in captured.err
    assert FileCatalogStore().catalog_path.read_bytes() == previous


def test_cli_rejects_url_option() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["model", "update", "openai", "--url", "https://evil.example/docs"])
    assert exc.value.code == 2


def test_cli_does_not_use_api_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-must-not-be-used")
    _patch_cli_transport(monkeypatch, _official_pages())
    code = main(["model", "update", "openai"])
    captured = capsys.readouterr()
    assert code == 0
    catalog_text = FileCatalogStore().catalog_path.read_text(encoding="utf-8")
    assert "sk-secret-must-not-be-used" not in catalog_text
    assert "sk-secret-must-not-be-used" not in captured.out
    assert "sk-secret-must-not-be-used" not in captured.err


def test_cli_unknown_provider(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["model", "update", "groq"])
    captured = capsys.readouterr()
    assert code == 2
    assert "unknown provider" in captured.err
    assert "openai" in captured.err
    assert "anthropic" in captured.err
    assert "google" in captured.err
    assert "fireworks" in captured.err
    assert "together" in captured.err


def test_cli_model_without_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["model"])
    captured = capsys.readouterr()
    assert code == 2
    assert "update" in captured.out.lower()


def test_cli_model_update_has_no_url_or_credential_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["model", "update", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--url" not in out
    assert "--api-key" not in out
    assert "--token" not in out
    assert "OPENAI_API_KEY" not in out
    assert "openai" in out.lower()
    assert "anthropic" in out.lower()
    assert "google" in out.lower()
    assert "fireworks" in out.lower()
    assert "together" in out.lower()
    assert "API key" in out or "api key" in out.lower()


def _anthropic_md(relative: str, **kwargs: object) -> FixtureHop:
    return FixtureHop(body=(ANTHROPIC_FIXTURES / relative).read_bytes(), **kwargs)  # type: ignore[arg-type]


def _anthropic_transport(
    pages: dict[str, FixtureHop],
    *,
    clock: object | None = None,
) -> FixtureTransport:
    return FixtureTransport(
        pages,
        policy=anthropic_docs_policy(),
        clock=clock if callable(clock) else (lambda: STAMP),
    )


def _anthropic_pages(**overrides: FixtureHop) -> dict[str, FixtureHop]:
    pages = {
        ANTHROPIC_INDEX_URL: _anthropic_md("overview.md"),
        ANTHROPIC_DEPRECATIONS_URL: _anthropic_md("deprecations.md"),
        anthropic_model_page_url("opus-5"): _anthropic_md("models/opus-5.md"),
        anthropic_model_page_url("haiku-4-5"): _anthropic_md("models/haiku-4-5.md"),
        anthropic_model_page_url("sonnet-5"): _anthropic_md("models/sonnet-5.md"),
        anthropic_model_page_url("fable-5-1"): _anthropic_md("models/fable-5-1.md"),
        anthropic_model_page_url("index-only"): _anthropic_md("models/index-only.md"),
        anthropic_model_page_url("sparse-test"): _anthropic_md("models/sparse-test.md"),
        anthropic_model_page_url("no-api-access"): _anthropic_md("models/no-api-access.md"),
    }
    pages.update(overrides)
    return pages


def test_anthropic_update_does_not_remove_openai(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    openai_result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert openai_result.committed is True
    openai_ids = {item.samyak_id for item in openai_result.catalog.models}
    anthropic_result = update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=lambda: "2026-09-11T13:00:00+00:00",
    )
    assert anthropic_result.committed is True
    assert anthropic_result.lifecycle_changes == ()
    loaded = store.load()
    providers = {item.id for item in loaded.providers}
    assert providers == {"anthropic", "openai"}
    ids = {item.samyak_id for item in loaded.models}
    assert openai_ids <= ids
    assert any(item.startswith("anthropic:") for item in ids)
    assert loaded.generated_at == "2026-09-11T13:00:00+00:00"


def test_failed_anthropic_update_preserves_openai(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    previous = store.catalog_path.read_bytes()
    pages = _anthropic_pages()
    pages[ANTHROPIC_INDEX_URL] = FixtureHop(status=500, body=b"error")
    result = update_anthropic_catalog(
        transport=_anthropic_transport(pages),
        store=store,
        clock=lambda: "2026-09-11T13:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert store.catalog_path.read_bytes() == previous
    loaded = catalog_from_json(previous.decode("utf-8"))
    assert {item.id for item in loaded.providers} == {"openai"}


def test_openai_refresh_does_not_drop_anthropic(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=_clock,
    )
    loaded = store.load()
    anthropic_ids = {item.samyak_id for item in loaded.models if item.provider_id == "anthropic"}
    update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=lambda: "2026-09-11T13:00:00+00:00",
    )
    loaded = store.load()
    remaining = {item.samyak_id for item in loaded.models if item.provider_id == "anthropic"}
    assert remaining == anthropic_ids
    assert any(item.provider_id == "openai" for item in loaded.models)


def test_cli_model_update_anthropic_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], samyak_cache_dir: Path
) -> None:
    monkeypatch.setattr(
        "samyak.model.update.anthropic_https_transport",
        lambda clock=None: _anthropic_transport(_anthropic_pages(), clock=clock),
    )
    monkeypatch.setattr("samyak.model.update.utc_now_iso", lambda: STAMP)
    code = main(["model", "update", "anthropic"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Updated Anthropic model catalog." in captured.out
    assert "Status: complete" in captured.out
    store = FileCatalogStore()
    catalog = store.load()
    anthropic_count = sum(1 for item in catalog.models if item.provider_id == "anthropic")
    assert _reported_model_count(captured.out) == anthropic_count
    assert any(item.provider_id == "anthropic" for item in catalog.models)


def test_mixed_catalog_update_counts_are_provider_scoped(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    openai_first = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert openai_first.committed is True
    openai_count = openai_first.provider_model_count
    assert openai_count == openai_first.catalog_model_count

    anthropic_result = update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=lambda: "2026-09-11T13:00:00+00:00",
    )
    assert anthropic_result.committed is True
    assert anthropic_result.catalog is not None
    anthropic_count = sum(
        1 for item in anthropic_result.catalog.models if item.provider_id == "anthropic"
    )
    assert anthropic_result.provider_model_count == anthropic_count
    assert anthropic_result.catalog_model_count == openai_count + anthropic_count
    assert anthropic_result.catalog_model_count > anthropic_result.provider_model_count

    openai_again = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=lambda: "2026-09-11T14:00:00+00:00",
    )
    assert openai_again.committed is True
    assert openai_again.provider_model_count == openai_count
    assert openai_again.catalog_model_count == openai_count + anthropic_count


def test_cli_mixed_catalog_reports_updated_provider_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], samyak_cache_dir: Path
) -> None:
    _patch_cli_transport(monkeypatch, _official_pages())
    monkeypatch.setattr(
        "samyak.model.update.anthropic_https_transport",
        lambda clock=None: _anthropic_transport(_anthropic_pages(), clock=clock),
    )
    assert main(["model", "update", "openai"]) == 0
    openai_out = capsys.readouterr().out
    store = FileCatalogStore()
    openai_count = sum(1 for item in store.load().models if item.provider_id == "openai")
    assert _reported_model_count(openai_out) == openai_count

    assert main(["model", "update", "anthropic"]) == 0
    anthropic_out = capsys.readouterr().out
    catalog = store.load()
    anthropic_count = sum(1 for item in catalog.models if item.provider_id == "anthropic")
    total = len(catalog.models)
    assert anthropic_count > 0
    assert total == openai_count + anthropic_count
    assert _reported_model_count(anthropic_out) == anthropic_count

    assert main(["model", "update", "openai"]) == 0
    openai_again = capsys.readouterr().out
    assert _reported_model_count(openai_again) == openai_count
    assert len(store.load().models) == total


def _reported_model_count(output: str) -> int:
    for line in output.splitlines():
        if line.startswith("Models: "):
            return int(line.removeprefix("Models: "))
    raise AssertionError(f"CLI output is missing a Models line:\n{output}")


def _write_corrupt_catalog(store: FileCatalogStore) -> bytes:
    store.catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"{not-json"
    store.catalog_path.write_bytes(payload)
    return payload


def _write_non_utf8_catalog(store: FileCatalogStore) -> bytes:
    store.catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\xff\xfecatalog"
    store.catalog_path.write_bytes(payload)
    return payload


def _write_unsupported_schema(store: FileCatalogStore) -> bytes:
    first = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert first.committed is True
    payload = json.loads(store.catalog_path.read_text(encoding="utf-8"))
    payload["catalog_schema_version"] = 99
    raw = json.dumps(payload).encode("utf-8")
    store.catalog_path.write_bytes(raw)
    return raw


def test_corrupt_catalog_openai_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_corrupt_catalog(store)
    result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert result.lifecycle_changes == ()
    assert "corrupt" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_corrupt_catalog_anthropic_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_corrupt_catalog(store)
    result = update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=_clock,
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "corrupt" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_non_utf8_catalog_openai_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_non_utf8_catalog(store)
    result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "corrupt" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_non_utf8_catalog_anthropic_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_non_utf8_catalog(store)
    result = update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=_clock,
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "corrupt" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_unsupported_schema_openai_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_unsupported_schema(store)
    result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=lambda: "2026-09-11T15:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "unsupported catalog_schema_version" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_unsupported_schema_anthropic_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_unsupported_schema(store)
    result = update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=lambda: "2026-09-11T15:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "unsupported catalog_schema_version" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_cli_corrupt_catalog_exits_nonzero_without_write(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], samyak_cache_dir: Path
) -> None:
    store = FileCatalogStore()
    previous = _write_corrupt_catalog(store)
    _patch_cli_transport(monkeypatch, _official_pages())
    code = main(["model", "update", "openai"])
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "was not changed" in captured.err
    assert store.catalog_path.read_bytes() == previous


def test_cli_unsupported_schema_exits_nonzero_without_write(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], samyak_cache_dir: Path
) -> None:
    store = FileCatalogStore()
    previous = _write_unsupported_schema(store)
    monkeypatch.setattr(
        "samyak.model.update.anthropic_https_transport",
        lambda clock=None: _anthropic_transport(_anthropic_pages(), clock=clock),
    )
    assert store.catalog_path.read_bytes() == previous


def _google_md(relative: str, **kwargs: object) -> FixtureHop:
    return FixtureHop(body=(GOOGLE_FIXTURES / relative).read_bytes(), **kwargs)  # type: ignore[arg-type]


def _google_transport(
    pages: dict[str, FixtureHop],
    *,
    clock: object | None = None,
) -> FixtureTransport:
    return FixtureTransport(
        pages,
        policy=google_docs_policy(),
        clock=clock if callable(clock) else (lambda: STAMP),
    )


def _google_pages(**overrides: FixtureHop) -> dict[str, FixtureHop]:
    pages = {
        GOOGLE_INDEX_URL: _google_md("models.md"),
        GOOGLE_DEPRECATIONS_URL: _google_md("deprecations.md"),
        GOOGLE_GEMINI_3_URL: _google_md("gemini-3.md"),
        google_model_page_url("gemini-3.8-flash"): _google_md("models/gemini-3.8-flash.md"),
        google_model_page_url("gemini-3.1-flash-image"): _google_md(
            "models/gemini-3.1-flash-image.md"
        ),
        google_model_page_url("gemini-3.1-pro-preview"): _google_md(
            "models/gemini-3.1-pro-preview.md"
        ),
        google_model_page_url("gemini-3.5-transcribe"): _google_md(
            "models/gemini-3.5-transcribe.md"
        ),
        google_model_page_url("index-only"): _google_md("models/index-only.md"),
        google_model_page_url("sparse-test"): _google_md("models/sparse-test.md"),
        google_model_page_url("gemini-stable-dated"): _google_md("models/gemini-stable-dated.md"),
        google_model_page_url("gemini-alias-target"): _google_md("models/gemini-alias-target.md"),
        google_model_page_url("imagen"): _google_md("models/imagen.md"),
        google_model_page_url("gemini-2.0-flash"): _google_md("models/gemini-2.0-flash.md"),
    }
    pages.update(overrides)
    return pages


def test_google_update_does_not_remove_openai_or_anthropic(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    openai_result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    anthropic_result = update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=lambda: "2026-09-11T13:00:00+00:00",
    )
    assert openai_result.committed is True
    assert anthropic_result.committed is True
    openai_ids = {item.samyak_id for item in openai_result.catalog.models}
    anthropic_ids = {
        item.samyak_id
        for item in anthropic_result.catalog.models
        if item.provider_id == "anthropic"
    }
    google_result = update_google_catalog(
        transport=_google_transport(_google_pages()),
        store=store,
        clock=lambda: "2026-09-12T12:00:00+00:00",
    )
    assert google_result.committed is True
    loaded = store.load()
    providers = {item.id for item in loaded.providers}
    assert providers == {"anthropic", "google", "openai"}
    ids = {item.samyak_id for item in loaded.models}
    assert openai_ids <= ids
    assert anthropic_ids <= ids
    assert any(item.startswith("google:") for item in ids)
    assert loaded.generated_at == "2026-09-12T12:00:00+00:00"
    google_count = sum(1 for item in loaded.models if item.provider_id == "google")
    assert google_result.provider_model_count == google_count
    assert google_result.catalog_model_count == len(loaded.models)
    assert google_result.catalog_model_count > google_result.provider_model_count


def test_google_update_does_not_delete_openai_or_anthropic_on_second_refresh(
    tmp_path: Path,
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=_clock,
    )
    update_google_catalog(
        transport=_google_transport(_google_pages()),
        store=store,
        clock=_clock,
    )
    before = store.load()
    openai_ids = {item.samyak_id for item in before.models if item.provider_id == "openai"}
    anthropic_ids = {item.samyak_id for item in before.models if item.provider_id == "anthropic"}
    update_google_catalog(
        transport=_google_transport(_google_pages()),
        store=store,
        clock=lambda: "2026-09-12T13:00:00+00:00",
    )
    loaded = store.load()
    assert {item.samyak_id for item in loaded.models if item.provider_id == "openai"} == openai_ids
    assert {
        item.samyak_id for item in loaded.models if item.provider_id == "anthropic"
    } == anthropic_ids


def test_failed_google_update_preserves_live_catalog(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=_clock,
    )
    previous = store.catalog_path.read_bytes()
    pages = _google_pages()
    pages[GOOGLE_INDEX_URL] = FixtureHop(status=500, body=b"error")
    result = update_google_catalog(
        transport=_google_transport(pages),
        store=store,
        clock=lambda: "2026-09-12T13:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert store.catalog_path.read_bytes() == previous
    loaded = catalog_from_json(previous.decode("utf-8"))
    assert {item.id for item in loaded.providers} == {"anthropic", "openai"}


def test_google_refresh_keeps_openai_partial_notices(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    pages = _official_pages()
    del pages[model_page_url("index-only")]
    openai_result = update_openai_catalog(
        transport=_transport(pages),
        store=store,
        clock=_clock,
    )
    assert openai_result.partial is True
    openai_codes = {notice.code for notice in openai_result.catalog.notices}
    assert "PARTIAL_MODEL_PAGES" in openai_codes
    google_result = update_google_catalog(
        transport=_google_transport(_google_pages()),
        store=store,
        clock=lambda: "2026-09-12T12:00:00+00:00",
    )
    assert google_result.committed is True
    assert google_result.partial is False
    loaded = store.load()
    codes = {notice.code for notice in loaded.notices}
    assert "PARTIAL_MODEL_PAGES" in codes
    assert "PARTIAL_MODEL_PAGES:google" not in codes
    assert any(item.provider_id == "openai" for item in loaded.models)
    assert any(item.provider_id == "google" for item in loaded.models)


def test_google_partial_does_not_replace_openai_partial_code(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    openai_pages = _official_pages()
    del openai_pages[model_page_url("index-only")]
    update_openai_catalog(
        transport=_transport(openai_pages),
        store=store,
        clock=_clock,
    )
    google_pages = _google_pages()
    del google_pages[google_model_page_url("index-only")]
    result = update_google_catalog(
        transport=_google_transport(google_pages),
        store=store,
        clock=lambda: "2026-09-12T12:00:00+00:00",
    )
    assert result.committed is True
    assert result.partial is True
    codes = {notice.code for notice in result.catalog.notices}
    assert "PARTIAL_MODEL_PAGES" in codes
    assert "PARTIAL_MODEL_PAGES:google" in codes
    assert "SOURCE_FETCH_FAILED:openai-docs-model-page:index-only" in codes
    assert "SOURCE_FETCH_FAILED:google-docs-model-page:index-only" in codes


def test_corrupt_catalog_google_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_corrupt_catalog(store)
    result = update_google_catalog(
        transport=_google_transport(_google_pages()),
        store=store,
        clock=_clock,
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "corrupt" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_unsupported_schema_google_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_unsupported_schema(store)
    result = update_google_catalog(
        transport=_google_transport(_google_pages()),
        store=store,
        clock=lambda: "2026-09-12T15:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "unsupported catalog_schema_version" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_cli_model_update_google_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], samyak_cache_dir: Path
) -> None:
    monkeypatch.setattr(
        "samyak.model.update.google_https_transport",
        lambda clock=None: _google_transport(_google_pages(), clock=clock),
    )
    monkeypatch.setattr("samyak.model.update.utc_now_iso", lambda: STAMP)
    code = main(["model", "update", "google"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Updated Google model catalog." in captured.out
    assert "Status: complete" in captured.out
    store = FileCatalogStore()
    catalog = store.load()
    google_count = sum(1 for item in catalog.models if item.provider_id == "google")
    assert _reported_model_count(captured.out) == google_count
    assert any(item.provider_id == "google" for item in catalog.models)


def test_cli_does_not_use_google_api_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret-must-not-be-used")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret-must-not-be-used")
    monkeypatch.setattr(
        "samyak.model.update.google_https_transport",
        lambda clock=None: _google_transport(_google_pages(), clock=clock),
    )
    monkeypatch.setattr("samyak.model.update.utc_now_iso", lambda: STAMP)
    code = main(["model", "update", "google"])
    captured = capsys.readouterr()
    assert code == 0
    catalog_text = FileCatalogStore().catalog_path.read_text(encoding="utf-8")
    assert "google-secret-must-not-be-used" not in catalog_text
    assert "gemini-secret-must-not-be-used" not in catalog_text
    assert "google-secret-must-not-be-used" not in captured.out
    assert "gemini-secret-must-not-be-used" not in captured.out


def _fireworks_md(relative: str, **kwargs: object) -> FixtureHop:
    return FixtureHop(body=(FIREWORKS_FIXTURES / relative).read_bytes(), **kwargs)  # type: ignore[arg-type]


def _fireworks_transport(
    pages: dict[str, FixtureHop],
    *,
    clock: object | None = None,
) -> FixtureTransport:
    return FixtureTransport(
        pages,
        policy=fireworks_docs_policy(),
        clock=clock if callable(clock) else (lambda: STAMP),
    )


def _fireworks_pages(**overrides: FixtureHop) -> dict[str, FixtureHop]:
    pages = {
        FIREWORKS_EMBEDDINGS_URL: _fireworks_md("embeddings.md"),
        FIREWORKS_SERVING_URL: _fireworks_md("serving-paths.md"),
        FIREWORKS_CHANGELOG_URL: _fireworks_md("changelog.md"),
        FIREWORKS_TEXT_URL: _fireworks_md("text-models.md"),
        FIREWORKS_VISION_URL: _fireworks_md("vision-models.md"),
        FIREWORKS_TOOL_URL: _fireworks_md("tool-calling.md"),
        FIREWORKS_KIMI_URL: _fireworks_md("kimi-k2.md"),
    }
    pages.update(overrides)
    return pages


def test_fireworks_update_preserves_openai_anthropic_and_google(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    openai_result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    anthropic_result = update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=lambda: "2026-09-11T13:00:00+00:00",
    )
    google_result = update_google_catalog(
        transport=_google_transport(_google_pages()),
        store=store,
        clock=lambda: "2026-09-12T12:00:00+00:00",
    )
    assert openai_result.committed is True
    assert anthropic_result.committed is True
    assert google_result.committed is True
    openai_ids = {item.samyak_id for item in openai_result.catalog.models}
    anthropic_ids = {
        item.samyak_id
        for item in anthropic_result.catalog.models
        if item.provider_id == "anthropic"
    }
    google_ids = {
        item.samyak_id for item in google_result.catalog.models if item.provider_id == "google"
    }
    fireworks_result = update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    assert fireworks_result.committed is True
    assert fireworks_result.partial is False
    loaded = store.load()
    providers = {item.id for item in loaded.providers}
    assert providers == {"anthropic", "fireworks", "google", "openai"}
    ids = {item.samyak_id for item in loaded.models}
    assert openai_ids <= ids
    assert anthropic_ids <= ids
    assert google_ids <= ids
    assert any(item.startswith("fireworks:") for item in ids)
    fireworks_count = sum(1 for item in loaded.models if item.provider_id == "fireworks")
    assert fireworks_result.provider_model_count == fireworks_count
    assert fireworks_result.catalog_model_count == len(loaded.models)
    assert fireworks_result.catalog_model_count > fireworks_result.provider_model_count
    codes = {notice.code for notice in loaded.notices}
    assert "DOCUMENTED_SUBSET:fireworks" in codes


def test_fireworks_second_refresh_does_not_delete_other_providers(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(transport=_transport(_official_pages()), store=store, clock=_clock)
    update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()), store=store, clock=_clock
    )
    update_google_catalog(transport=_google_transport(_google_pages()), store=store, clock=_clock)
    update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()), store=store, clock=_clock
    )
    before = store.load()
    openai_ids = {item.samyak_id for item in before.models if item.provider_id == "openai"}
    anthropic_ids = {item.samyak_id for item in before.models if item.provider_id == "anthropic"}
    google_ids = {item.samyak_id for item in before.models if item.provider_id == "google"}
    update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=lambda: "2026-09-12T15:00:00+00:00",
    )
    loaded = store.load()
    assert {item.samyak_id for item in loaded.models if item.provider_id == "openai"} == openai_ids
    assert {
        item.samyak_id for item in loaded.models if item.provider_id == "anthropic"
    } == anthropic_ids
    assert {item.samyak_id for item in loaded.models if item.provider_id == "google"} == google_ids


def test_failed_fireworks_update_preserves_live_catalog(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(transport=_transport(_official_pages()), store=store, clock=_clock)
    update_google_catalog(
        transport=_google_transport(_google_pages()),
        store=store,
        clock=lambda: "2026-09-12T12:00:00+00:00",
    )
    previous = store.catalog_path.read_bytes()
    pages = _fireworks_pages()
    pages[FIREWORKS_EMBEDDINGS_URL] = FixtureHop(status=500, body=b"error")
    result = update_fireworks_catalog(
        transport=_fireworks_transport(pages),
        store=store,
        clock=lambda: "2026-09-12T16:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert store.catalog_path.read_bytes() == previous
    loaded = catalog_from_json(previous.decode("utf-8"))
    assert {item.id for item in loaded.providers} == {"google", "openai"}


def test_fireworks_refresh_keeps_other_provider_notices(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    pages = _official_pages()
    del pages[model_page_url("index-only")]
    openai_result = update_openai_catalog(
        transport=_transport(pages),
        store=store,
        clock=_clock,
    )
    assert openai_result.partial is True
    fireworks_result = update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    assert fireworks_result.committed is True
    assert fireworks_result.partial is False
    loaded = store.load()
    codes = {notice.code for notice in loaded.notices}
    assert "PARTIAL_MODEL_PAGES" in codes
    assert "DOCUMENTED_SUBSET:fireworks" in codes
    assert "PARTIAL_MODEL_PAGES:fireworks" not in codes
    assert any(item.provider_id == "openai" for item in loaded.models)
    assert any(item.provider_id == "fireworks" for item in loaded.models)


def test_fireworks_partial_does_not_replace_openai_partial_code(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    openai_pages = _official_pages()
    del openai_pages[model_page_url("index-only")]
    update_openai_catalog(
        transport=_transport(openai_pages),
        store=store,
        clock=_clock,
    )
    fireworks_pages = _fireworks_pages()
    del fireworks_pages[FIREWORKS_KIMI_URL]
    result = update_fireworks_catalog(
        transport=_fireworks_transport(fireworks_pages),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    assert result.committed is True
    assert result.partial is True
    codes = {notice.code for notice in result.catalog.notices}
    assert "PARTIAL_MODEL_PAGES" in codes
    assert "PARTIAL_MODEL_PAGES:fireworks" in codes
    assert "DOCUMENTED_SUBSET:fireworks" in codes
    assert "SOURCE_FETCH_FAILED:openai-docs-model-page:index-only" in codes
    assert "SOURCE_FETCH_FAILED:fireworks-docs-model-page:kimi-k2" in codes


_KIMI_INSTRUCT_ID = "fireworks:accounts/fireworks/models/kimi-k2-instruct"
_KIMI_ALIAS_ID = "fireworks:accounts/fireworks/models/kimi-k2-instruct-latest"
_SERVING_WITHOUT_NEW_FAST = (
    b"# Serverless Modes\n\n"
    b"Fast is not a different model.\n\n"
    b"| Model | `model` ID |\n"
    b"| --- | --- |\n"
    b"| GLM 5.2 Fast | `accounts/fireworks/routers/glm-5p2-fast` |\n"
)
_SERVING_WITHOUT_GLM_FAST = (
    b"# Serverless Modes\n\n"
    b"Fast is not a different model.\n\n"
    b"| Model | `model` ID |\n"
    b"| --- | --- |\n"
    b"| Kimi K3 Fast | `accounts/fireworks/routers/kimi-k3-fast` |\n"
)
_KIMI_PAGE_WITHOUT_INSTRUCT = (
    b'# Kimi K2 family\n\n```python\nmodel="accounts/fireworks/models/kimi-k2p5"\n```\n'
)


def _fireworks_record(catalog, samyak_id: str):
    matches = [item for item in catalog.models if item.samyak_id == samyak_id]
    assert matches, f"missing {samyak_id}"
    return matches[0]


def test_fireworks_optional_page_failure_preserves_page_established_model(
    tmp_path: Path,
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=_clock,
    )
    assert first.committed is True
    assert first.partial is False
    previous_instruct = _fireworks_record(first.catalog, _KIMI_INSTRUCT_ID)
    previous_alias = _fireworks_record(first.catalog, _KIMI_ALIAS_ID)
    pages = _fireworks_pages()
    del pages[FIREWORKS_KIMI_URL]
    second = update_fireworks_catalog(
        transport=_fireworks_transport(pages),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    assert second.ok is True
    assert second.committed is True
    assert second.partial is True
    assert second.catalog is not None
    codes = {notice.code for notice in second.catalog.notices}
    assert "PARTIAL_MODEL_PAGES:fireworks" in codes
    assert "SOURCE_FETCH_FAILED:fireworks-docs-model-page:kimi-k2" in codes
    retained = _fireworks_record(second.catalog, _KIMI_INSTRUCT_ID)
    assert retained == previous_instruct
    assert _fireworks_record(second.catalog, _KIMI_ALIAS_ID) == previous_alias
    removed = {
        item.samyak_id
        for item in second.lifecycle_changes
        if item.change_type is LifecycleChangeType.REMOVED
    }
    assert _KIMI_INSTRUCT_ID not in removed
    assert _KIMI_ALIAS_ID not in removed
    loaded = store.load()
    assert _fireworks_record(loaded, _KIMI_INSTRUCT_ID) == previous_instruct


def test_fireworks_partial_refresh_still_adds_new_serving_path_models(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    first_pages = _fireworks_pages(
        **{FIREWORKS_SERVING_URL: FixtureHop(body=_SERVING_WITHOUT_NEW_FAST)}
    )
    first = update_fireworks_catalog(
        transport=_fireworks_transport(first_pages),
        store=store,
        clock=_clock,
    )
    assert first.committed is True
    ids = {item.samyak_id for item in first.catalog.models}
    assert "fireworks:accounts/fireworks/routers/kimi-k3-fast" not in ids
    second_pages = _fireworks_pages()
    del second_pages[FIREWORKS_KIMI_URL]
    second = update_fireworks_catalog(
        transport=_fireworks_transport(second_pages),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    assert second.partial is True
    added = {
        item.samyak_id
        for item in second.lifecycle_changes
        if item.change_type is LifecycleChangeType.ADDED
    }
    assert "fireworks:accounts/fireworks/routers/kimi-k3-fast" in added
    assert "fireworks:accounts/fireworks/routers/glm-5p3-fast" in added
    assert "fireworks:accounts/fireworks/routers/glm-5p2-fast-us" in added
    removed = {
        item.samyak_id
        for item in second.lifecycle_changes
        if item.change_type is LifecycleChangeType.REMOVED
    }
    assert _KIMI_INSTRUCT_ID not in removed
    loaded_ids = {item.samyak_id for item in store.load().models}
    assert _KIMI_INSTRUCT_ID in loaded_ids
    assert "fireworks:accounts/fireworks/routers/kimi-k3-fast" in loaded_ids


def test_fireworks_successful_page_without_model_still_reports_removal(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=_clock,
    )
    assert first.committed is True
    second = update_fireworks_catalog(
        transport=_fireworks_transport(
            _fireworks_pages(**{FIREWORKS_KIMI_URL: FixtureHop(body=_KIMI_PAGE_WITHOUT_INSTRUCT)})
        ),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    assert second.committed is True
    assert second.partial is False
    removed = {
        item.samyak_id
        for item in second.lifecycle_changes
        if item.change_type is LifecycleChangeType.REMOVED
    }
    assert _KIMI_INSTRUCT_ID in removed
    loaded_ids = {item.samyak_id for item in store.load().models}
    assert _KIMI_INSTRUCT_ID not in loaded_ids


def test_fireworks_unrelated_optional_failure_does_not_suppress_removal(
    tmp_path: Path,
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=_clock,
    )
    assert first.committed is True
    pages = _fireworks_pages(**{FIREWORKS_SERVING_URL: FixtureHop(body=_SERVING_WITHOUT_GLM_FAST)})
    del pages[FIREWORKS_KIMI_URL]
    second = update_fireworks_catalog(
        transport=_fireworks_transport(pages),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    assert second.partial is True
    removed = {
        item.samyak_id
        for item in second.lifecycle_changes
        if item.change_type is LifecycleChangeType.REMOVED
    }
    assert "fireworks:accounts/fireworks/routers/glm-5p2-fast" in removed
    assert _KIMI_INSTRUCT_ID not in removed
    loaded_ids = {item.samyak_id for item in store.load().models}
    assert "fireworks:accounts/fireworks/routers/glm-5p2-fast" not in loaded_ids
    assert _KIMI_INSTRUCT_ID in loaded_ids


def _retention_provenance(source_id: str) -> Provenance:
    return Provenance(
        provider="fireworks",
        source_id=source_id,
        source_kind=SourceKind.PROVIDER_DOCS,
        source_url="https://docs.fireworks.ai/docs.md",
        content_hash="sha256:abc",
        retrieved_at=STAMP,
        observed_at=None,
        verified_at=STAMP,
        confidence=Confidence.HIGH,
    )


def _retention_fact(value: object, *, source_id: str) -> Fact:
    return Fact(status=FactStatus.KNOWN, value=value, provenance=_retention_provenance(source_id))


def _retention_model(provider_model_id: str, **fields: object) -> ModelRecord:
    return ModelRecord(
        identity=ModelIdentity(provider_id="fireworks", provider_model_id=provider_model_id),
        **fields,  # type: ignore[arg-type]
    )


def _retention_catalog(*models: ModelRecord, notices: tuple[CatalogNotice, ...] = ()):
    return build_catalog(
        models=models,
        generated_at=STAMP,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=STAMP,
            overlay=CatalogOverlay.USER_CACHE,
        ),
        notices=notices,
    )


def test_partial_refresh_retains_page_established_presence() -> None:
    page = "fireworks-docs-model-page:optional"
    previous_record = _retention_model(
        "page-established",
        lifecycle=_retention_fact(LifecycleState.ACTIVE, source_id=page),
        api_access=_retention_fact(True, source_id=page),
    )
    incoming_new = _retention_model(
        "inventory-new",
        lifecycle=_retention_fact(LifecycleState.ACTIVE, source_id="fireworks-docs-serving-paths"),
    )
    previous = _retention_catalog(previous_record)
    incoming = _retention_catalog(incoming_new)
    retained = _retain_records_for_failed_sources(
        previous,
        incoming,
        provider_id="fireworks",
        notices=(
            CatalogNotice(code=f"SOURCE_FETCH_FAILED:{page}", message="optional page failed"),
        ),
    )
    ids = {item.samyak_id for item in retained.models}
    assert "fireworks:page-established" in ids
    assert "fireworks:inventory-new" in ids
    kept = next(item for item in retained.models if item.samyak_id == "fireworks:page-established")
    assert kept == previous_record


def test_partial_refresh_does_not_retain_inventory_model_for_unrelated_page_fact() -> None:
    page = "fireworks-docs-model-page:optional"
    inventory = "fireworks-docs-serving-paths"
    previous_record = _retention_model(
        "inventory-established",
        lifecycle=_retention_fact(LifecycleState.ACTIVE, source_id=inventory),
        tool_calling=_retention_fact(True, source_id=page),
    )
    incoming_new = _retention_model(
        "inventory-new",
        lifecycle=_retention_fact(LifecycleState.ACTIVE, source_id=inventory),
    )
    previous = _retention_catalog(previous_record)
    incoming = _retention_catalog(incoming_new)
    result = _retain_records_for_failed_sources(
        previous,
        incoming,
        provider_id="fireworks",
        notices=(
            CatalogNotice(code=f"SOURCE_FETCH_FAILED:{page}", message="optional page failed"),
        ),
    )
    ids = {item.samyak_id for item in result.models}
    assert "fireworks:inventory-established" not in ids
    assert "fireworks:inventory-new" in ids


def test_partial_refresh_unrelated_optional_failure_does_not_retain_other_inventory() -> None:
    page_a = "fireworks-docs-model-page:model-a"
    inventory = "fireworks-docs-serving-paths"
    model_a = _retention_model(
        "page-a",
        lifecycle=_retention_fact(LifecycleState.ACTIVE, source_id=page_a),
    )
    model_b = _retention_model(
        "inventory-b",
        lifecycle=_retention_fact(LifecycleState.ACTIVE, source_id=inventory),
    )
    previous = _retention_catalog(model_a, model_b)
    incoming = _retention_catalog()
    result = _retain_records_for_failed_sources(
        previous,
        incoming,
        provider_id="fireworks",
        notices=(
            CatalogNotice(code=f"SOURCE_FETCH_FAILED:{page_a}", message="optional page failed"),
        ),
    )
    ids = {item.samyak_id for item in result.models}
    assert "fireworks:page-a" in ids
    assert "fireworks:inventory-b" not in ids


def test_failed_optional_page_does_not_retain_inventory_established_model(
    tmp_path: Path,
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=_clock,
    )
    assert first.committed is True
    glm_id = "fireworks:accounts/fireworks/routers/glm-5p2-fast"
    glm = _fireworks_record(first.catalog, glm_id)
    mutated = replace(
        glm,
        tool_calling=Fact(
            status=FactStatus.KNOWN,
            value=True,
            provenance=_retention_provenance("fireworks-docs-model-page:kimi-k2"),
        ),
    )
    rewritten = [
        mutated if record.samyak_id == glm_id else record for record in first.catalog.models
    ]
    store.save(
        build_catalog(
            models=rewritten,
            generated_at=first.catalog.generated_at,
            freshness=first.catalog.freshness,
            notices=first.catalog.notices,
            version=first.catalog.version,
        )
    )
    pages = _fireworks_pages(**{FIREWORKS_SERVING_URL: FixtureHop(body=_SERVING_WITHOUT_GLM_FAST)})
    del pages[FIREWORKS_KIMI_URL]
    second = update_fireworks_catalog(
        transport=_fireworks_transport(pages),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    assert second.partial is True
    removed = {
        item.samyak_id
        for item in second.lifecycle_changes
        if item.change_type is LifecycleChangeType.REMOVED
    }
    assert glm_id in removed
    assert _KIMI_INSTRUCT_ID not in removed
    loaded_ids = {item.samyak_id for item in store.load().models}
    assert glm_id not in loaded_ids
    assert _KIMI_INSTRUCT_ID in loaded_ids


def test_corrupt_catalog_fireworks_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_corrupt_catalog(store)
    result = update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=_clock,
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "corrupt" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_unsupported_schema_fireworks_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_unsupported_schema(store)
    result = update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=lambda: "2026-09-12T17:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "unsupported catalog_schema_version" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_same_underlying_model_stays_separate_across_serving_sources(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(transport=_transport(_official_pages()), store=store, clock=_clock)
    update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    loaded = store.load()
    openai_ids = {item.samyak_id for item in loaded.models if item.provider_id == "openai"}
    fireworks_ids = {item.samyak_id for item in loaded.models if item.provider_id == "fireworks"}
    assert openai_ids
    assert fireworks_ids
    assert openai_ids.isdisjoint(fireworks_ids)
    fireworks_qwen = next(
        item for item in loaded.models if item.provider_model_id == "fireworks/qwen3-embedding-8b"
    )
    assert fireworks_qwen.provider_id == "fireworks"
    assert fireworks_qwen.samyak_id == "fireworks:fireworks/qwen3-embedding-8b"


def test_cli_model_update_fireworks_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], samyak_cache_dir: Path
) -> None:
    monkeypatch.setattr(
        "samyak.model.update.fireworks_https_transport",
        lambda clock=None: _fireworks_transport(_fireworks_pages(), clock=clock),
    )
    monkeypatch.setattr("samyak.model.update.utc_now_iso", lambda: STAMP)
    code = main(["model", "update", "fireworks"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Updated Fireworks model catalog." in captured.out
    assert "Status: complete" in captured.out
    assert "Notices:" in captured.out
    assert "Some model pages could not be retrieved." not in captured.out
    store = FileCatalogStore()
    catalog = store.load()
    fireworks_count = sum(1 for item in catalog.models if item.provider_id == "fireworks")
    assert _reported_model_count(captured.out) == fireworks_count
    assert any(item.provider_id == "fireworks" for item in catalog.models)
    assert "DOCUMENTED_SUBSET:fireworks" in {notice.code for notice in catalog.notices}


def test_cli_does_not_use_fireworks_api_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fireworks-secret-must-not-be-used")
    monkeypatch.setattr(
        "samyak.model.update.fireworks_https_transport",
        lambda clock=None: _fireworks_transport(_fireworks_pages(), clock=clock),
    )
    monkeypatch.setattr("samyak.model.update.utc_now_iso", lambda: STAMP)
    code = main(["model", "update", "fireworks"])
    captured = capsys.readouterr()
    assert code == 0
    catalog_text = FileCatalogStore().catalog_path.read_text(encoding="utf-8")
    assert "fireworks-secret-must-not-be-used" not in catalog_text
    assert "fireworks-secret-must-not-be-used" not in captured.out
    assert "fireworks-secret-must-not-be-used" not in captured.err


def _together_md(relative: str, **kwargs: object) -> FixtureHop:
    return FixtureHop(body=(TOGETHER_FIXTURES / relative).read_bytes(), **kwargs)  # type: ignore[arg-type]


def _together_transport(
    pages: dict[str, FixtureHop],
    *,
    clock: object | None = None,
) -> FixtureTransport:
    return FixtureTransport(
        pages,
        policy=together_docs_policy(),
        clock=clock if callable(clock) else (lambda: STAMP),
    )


def _together_pages(**overrides: FixtureHop) -> dict[str, FixtureHop]:
    pages = {
        TOGETHER_SERVERLESS_URL: _together_md("serverless.md"),
        TOGETHER_DEPRECATIONS_URL: _together_md("deprecations.md"),
        TOGETHER_CHANGELOG_URL: _together_md("changelog.md"),
    }
    for descriptor in together_optional_source_descriptors():
        if descriptor.url == TOGETHER_CHANGELOG_URL:
            continue
        slug = descriptor.url.rsplit("/", 1)[-1].removesuffix(".md")
        pages[descriptor.url] = _together_md(f"{slug}.md")
    pages.update(overrides)
    return pages


def test_together_update_preserves_other_serving_sources(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    openai_result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=store,
        clock=_clock,
    )
    anthropic_result = update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()),
        store=store,
        clock=lambda: "2026-09-11T13:00:00+00:00",
    )
    google_result = update_google_catalog(
        transport=_google_transport(_google_pages()),
        store=store,
        clock=lambda: "2026-09-12T12:00:00+00:00",
    )
    fireworks_result = update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    together_result = update_together_catalog(
        transport=_together_transport(_together_pages()),
        store=store,
        clock=lambda: "2026-09-12T16:00:00+00:00",
    )
    assert openai_result.committed is True
    assert anthropic_result.committed is True
    assert google_result.committed is True
    assert fireworks_result.committed is True
    assert together_result.committed is True
    assert together_result.partial is False
    loaded = store.load()
    providers = {item.id for item in loaded.providers}
    assert providers == {"anthropic", "fireworks", "google", "openai", "together"}
    ids = {item.samyak_id for item in loaded.models}
    assert {item.samyak_id for item in openai_result.catalog.models} <= ids
    assert any(item.startswith("together:") for item in ids)
    together_count = sum(1 for item in loaded.models if item.provider_id == "together")
    assert together_result.provider_model_count == together_count
    assert together_result.catalog_model_count == len(loaded.models)
    assert together_result.catalog_model_count > together_result.provider_model_count
    openai_gpt = [
        item for item in loaded.models if item.samyak_id == "together:openai/gpt-oss-120b"
    ]
    assert openai_gpt
    assert openai_gpt[0].provider_id == "together"
    assert "openai:gpt-oss-120b" not in ids


def test_together_second_refresh_does_not_delete_other_providers(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(transport=_transport(_official_pages()), store=store, clock=_clock)
    update_anthropic_catalog(
        transport=_anthropic_transport(_anthropic_pages()), store=store, clock=_clock
    )
    update_google_catalog(transport=_google_transport(_google_pages()), store=store, clock=_clock)
    update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()), store=store, clock=_clock
    )
    update_together_catalog(
        transport=_together_transport(_together_pages()), store=store, clock=_clock
    )
    before = store.load()
    openai_ids = {item.samyak_id for item in before.models if item.provider_id == "openai"}
    anthropic_ids = {item.samyak_id for item in before.models if item.provider_id == "anthropic"}
    google_ids = {item.samyak_id for item in before.models if item.provider_id == "google"}
    fireworks_ids = {item.samyak_id for item in before.models if item.provider_id == "fireworks"}
    update_together_catalog(
        transport=_together_transport(_together_pages()),
        store=store,
        clock=lambda: "2026-09-12T17:00:00+00:00",
    )
    loaded = store.load()
    assert {item.samyak_id for item in loaded.models if item.provider_id == "openai"} == openai_ids
    assert {
        item.samyak_id for item in loaded.models if item.provider_id == "anthropic"
    } == anthropic_ids
    assert {item.samyak_id for item in loaded.models if item.provider_id == "google"} == google_ids
    assert {
        item.samyak_id for item in loaded.models if item.provider_id == "fireworks"
    } == fireworks_ids


def test_failed_together_update_preserves_live_catalog(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(transport=_transport(_official_pages()), store=store, clock=_clock)
    update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    previous = store.catalog_path.read_bytes()
    pages = _together_pages()
    pages[TOGETHER_SERVERLESS_URL] = FixtureHop(status=500, body=b"error")
    result = update_together_catalog(
        transport=_together_transport(pages),
        store=store,
        clock=lambda: "2026-09-12T18:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert store.catalog_path.read_bytes() == previous
    loaded = catalog_from_json(previous.decode("utf-8"))
    assert {item.id for item in loaded.providers} == {"fireworks", "openai"}


def test_together_refresh_keeps_other_provider_notices(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    pages = _official_pages()
    del pages[model_page_url("index-only")]
    openai_result = update_openai_catalog(
        transport=_transport(pages),
        store=store,
        clock=_clock,
    )
    assert openai_result.partial is True
    together_result = update_together_catalog(
        transport=_together_transport(_together_pages()),
        store=store,
        clock=lambda: "2026-09-12T16:00:00+00:00",
    )
    assert together_result.committed is True
    assert together_result.partial is False
    loaded = store.load()
    codes = {notice.code for notice in loaded.notices}
    assert "PARTIAL_MODEL_PAGES" in codes
    assert "PARTIAL_MODEL_PAGES:together" not in codes
    assert any(item.provider_id == "openai" for item in loaded.models)
    assert any(item.provider_id == "together" for item in loaded.models)


def test_together_partial_does_not_replace_openai_partial_code(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    openai_pages = _official_pages()
    del openai_pages[model_page_url("index-only")]
    update_openai_catalog(
        transport=_transport(openai_pages),
        store=store,
        clock=_clock,
    )
    together_pages = _together_pages()
    del together_pages[TOGETHER_CHANGELOG_URL]
    result = update_together_catalog(
        transport=_together_transport(together_pages),
        store=store,
        clock=lambda: "2026-09-12T16:00:00+00:00",
    )
    assert result.committed is True
    assert result.partial is True
    codes = {notice.code for notice in result.catalog.notices}
    assert "PARTIAL_MODEL_PAGES" in codes
    assert "PARTIAL_MODEL_PAGES:together" in codes
    assert "SOURCE_FETCH_FAILED:openai-docs-model-page:index-only" in codes
    assert "SOURCE_FETCH_FAILED:together-docs-changelog" in codes


def test_corrupt_catalog_together_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_corrupt_catalog(store)
    result = update_together_catalog(
        transport=_together_transport(_together_pages()),
        store=store,
        clock=_clock,
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "corrupt" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_unsupported_schema_together_update_does_not_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    previous = _write_unsupported_schema(store)
    result = update_together_catalog(
        transport=_together_transport(_together_pages()),
        store=store,
        clock=lambda: "2026-09-12T19:00:00+00:00",
    )
    assert result.ok is False
    assert result.committed is False
    assert result.previous_existed is True
    assert result.catalog is None
    assert "unsupported catalog_schema_version" in (result.error or "")
    assert store.catalog_path.read_bytes() == previous
    assert not store.backup_path.exists()


def test_same_underlying_model_stays_separate_from_together(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(transport=_transport(_official_pages()), store=store, clock=_clock)
    update_together_catalog(
        transport=_together_transport(_together_pages()),
        store=store,
        clock=lambda: "2026-09-12T16:00:00+00:00",
    )
    loaded = store.load()
    openai_ids = {item.samyak_id for item in loaded.models if item.provider_id == "openai"}
    together_ids = {item.samyak_id for item in loaded.models if item.provider_id == "together"}
    assert openai_ids
    assert together_ids
    assert openai_ids.isdisjoint(together_ids)
    together_openai = next(
        item for item in loaded.models if item.provider_model_id == "openai/gpt-oss-120b"
    )
    assert together_openai.provider_id == "together"
    assert together_openai.samyak_id == "together:openai/gpt-oss-120b"


def test_cli_model_update_together_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], samyak_cache_dir: Path
) -> None:
    monkeypatch.setattr(
        "samyak.model.update.together_https_transport",
        lambda clock=None: _together_transport(_together_pages(), clock=clock),
    )
    monkeypatch.setattr("samyak.model.update.utc_now_iso", lambda: STAMP)
    code = main(["model", "update", "together"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Updated Together model catalog." in captured.out
    assert "Status: complete" in captured.out
    assert "Some model pages could not be retrieved." not in captured.out
    store = FileCatalogStore()
    catalog = store.load()
    together_count = sum(1 for item in catalog.models if item.provider_id == "together")
    assert _reported_model_count(captured.out) == together_count
    assert any(item.provider_id == "together" for item in catalog.models)


def test_cli_does_not_use_together_api_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "together-secret-must-not-be-used")
    monkeypatch.setattr(
        "samyak.model.update.together_https_transport",
        lambda clock=None: _together_transport(_together_pages(), clock=clock),
    )
    monkeypatch.setattr("samyak.model.update.utc_now_iso", lambda: STAMP)
    code = main(["model", "update", "together"])
    captured = capsys.readouterr()
    assert code == 0
    catalog_text = FileCatalogStore().catalog_path.read_text(encoding="utf-8")
    assert "together-secret-must-not-be-used" not in catalog_text
    assert "together-secret-must-not-be-used" not in captured.out
    assert "together-secret-must-not-be-used" not in captured.err
