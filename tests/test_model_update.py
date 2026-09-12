"""Local overlay update and CLI tests. No live network."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from samyak.cli import main
from samyak.model.catalog import CatalogNotice, CatalogOverlay, build_catalog
from samyak.model.errors import CatalogValidationError
from samyak.model.http import FixtureHop, FixtureTransport
from samyak.model.providers.anthropic.fetch import anthropic_docs_policy
from samyak.model.providers.anthropic.sources import (
    DEPRECATIONS_URL as ANTHROPIC_DEPRECATIONS_URL,
)
from samyak.model.providers.anthropic.sources import (
    MODELS_INDEX_URL as ANTHROPIC_INDEX_URL,
)
from samyak.model.providers.anthropic.sources import model_page_url as anthropic_model_page_url
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
from samyak.model.serialize import catalog_from_json
from samyak.model.store import FileCatalogStore
from samyak.model.update import update_anthropic_catalog, update_openai_catalog

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "openai"
ANTHROPIC_FIXTURES = Path(__file__).parent / "fixtures" / "models" / "anthropic"
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
    code = main(["model", "update", "google"])
    captured = capsys.readouterr()
    assert code == 2
    assert "unknown provider" in captured.err
    assert "openai" in captured.err
    assert "anthropic" in captured.err


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
    monkeypatch.setattr("samyak.model.update.utc_now_iso", lambda: STAMP)
    code = main(["model", "update", "anthropic"])
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "unsupported catalog_schema_version" in captured.err
    assert "was not changed" in captured.err
    assert store.catalog_path.read_bytes() == previous
