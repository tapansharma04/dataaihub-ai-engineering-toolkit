"""`samyak model update all` orchestration. No live network."""

from __future__ import annotations

import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from samyak.cli import main
from samyak.model.catalog import CATALOG_SCHEMA_VERSION, CatalogOverlay, build_catalog
from samyak.model.errors import CatalogValidationError
from samyak.model.facts import (
    Confidence,
    Fact,
    FactStatus,
    LifecycleState,
    Provenance,
    SourceKind,
)
from samyak.model.http import FixtureHop
from samyak.model.identity import ModelIdentity
from samyak.model.lifecycle import LifecycleChange, LifecycleChangeType
from samyak.model.records import ModelRecord
from samyak.model.store import FileCatalogStore
from samyak.model.update import (
    HISTORY_WRITE_FAILED_CODE,
    SUPPORTED_PROVIDERS,
    CatalogUpdateResult,
    update_all_catalogs,
    update_provider_catalog,
)
from test_model_update import (
    GOOGLE_INDEX_URL,
    STAMP,
    _anthropic_pages,
    _anthropic_transport,
    _clock,
    _fireworks_pages,
    _fireworks_transport,
    _google_pages,
    _google_transport,
    _official_pages,
    _together_pages,
    _together_transport,
    _transport,
)


@pytest.fixture(autouse=True)
def deny_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Model Intelligence update-all tests must not use the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("urllib.request.urlopen", blocked)


def _patch_all_transports(
    monkeypatch: pytest.MonkeyPatch,
    *,
    google_pages: dict[str, FixtureHop] | None = None,
) -> None:
    monkeypatch.setattr(
        "samyak.model.update.openai_https_transport",
        lambda clock=None: _transport(_official_pages(), clock=clock),
    )
    monkeypatch.setattr(
        "samyak.model.update.anthropic_https_transport",
        lambda clock=None: _anthropic_transport(_anthropic_pages(), clock=clock),
    )
    google = _google_pages() if google_pages is None else google_pages
    monkeypatch.setattr(
        "samyak.model.update.google_https_transport",
        lambda clock=None, pages=google: _google_transport(pages, clock=clock),
    )
    monkeypatch.setattr(
        "samyak.model.update.fireworks_https_transport",
        lambda clock=None: _fireworks_transport(_fireworks_pages(), clock=clock),
    )
    monkeypatch.setattr(
        "samyak.model.update.together_https_transport",
        lambda clock=None: _together_transport(_together_pages(), clock=clock),
    )
    monkeypatch.setattr("samyak.model.update.utc_now_iso", lambda: STAMP)


def _failed_google_pages() -> dict[str, FixtureHop]:
    pages = _google_pages()
    pages[GOOGLE_INDEX_URL] = FixtureHop(status=500, body=b"error")
    return pages


def _stub_result(
    provider_id: str,
    *,
    ok: bool = True,
    committed: bool = True,
    error: str | None = None,
    lifecycle_changes: tuple[LifecycleChange, ...] = (),
    catalog_path: Path | None = None,
) -> CatalogUpdateResult:
    return CatalogUpdateResult(
        provider_id=provider_id,
        ok=ok,
        committed=committed,
        partial=False,
        catalog=None,
        catalog_path=catalog_path or Path("/tmp/catalog.json"),
        previous_existed=True,
        error=error,
        notices=(),
        lifecycle_changes=lifecycle_changes,
    )


def _lifecycle_change(samyak_id: str = "openai:gpt-test") -> LifecycleChange:
    identity = ModelIdentity(provider_id="openai", provider_model_id="gpt-test")
    provenance = Provenance(
        provider="openai",
        source_id="openai-docs-deprecations",
        source_kind=SourceKind.PROVIDER_DEPRECATIONS,
        source_url="https://example.invalid/deprecations.md",
        content_hash="sha256:abc",
        retrieved_at=STAMP,
        observed_at=None,
        verified_at=STAMP,
        confidence=Confidence.HIGH,
    )
    previous = ModelRecord(
        identity=identity,
        lifecycle=Fact(
            status=FactStatus.KNOWN,
            value=LifecycleState.ACTIVE,
            provenance=provenance,
        ),
    )
    current = ModelRecord(
        identity=identity,
        lifecycle=Fact(
            status=FactStatus.KNOWN,
            value=LifecycleState.DEPRECATED,
            provenance=provenance,
        ),
    )
    return LifecycleChange(
        change_type=LifecycleChangeType.LIFECYCLE_CHANGED,
        samyak_id=samyak_id,
        previous=previous,
        current=current,
    )


def test_update_all_discovers_supported_providers_in_registry_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_update(provider_id: str, **kwargs: object) -> CatalogUpdateResult:
        calls.append(provider_id)
        return _stub_result(provider_id, catalog_path=tmp_path / "catalog.json")

    monkeypatch.setattr("samyak.model.update.update_provider_catalog", fake_update)
    results = update_all_catalogs(store=FileCatalogStore(root=tmp_path / "cache"), clock=_clock)
    assert calls == list(SUPPORTED_PROVIDERS)
    assert tuple(item.provider_id for item in results) == SUPPORTED_PROVIDERS
    assert set(SUPPORTED_PROVIDERS) == {"anthropic", "fireworks", "google", "openai", "together"}


def test_update_all_reuses_existing_provider_update_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    real = update_provider_catalog

    def wrapped(provider_id: str, **kwargs: object) -> CatalogUpdateResult:
        calls.append(provider_id)
        return real(provider_id, **kwargs)

    monkeypatch.setattr("samyak.model.update.update_provider_catalog", wrapped)
    _patch_all_transports(monkeypatch)
    store = FileCatalogStore(root=tmp_path / "cache")
    results = update_all_catalogs(store=store, clock=_clock)
    assert calls == list(SUPPORTED_PROVIDERS)
    assert all(item.ok and item.committed for item in results)


def test_update_all_succeeds_for_every_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_all_transports(monkeypatch)
    store = FileCatalogStore(root=tmp_path / "cache")
    results = update_all_catalogs(store=store, clock=_clock)
    assert tuple(item.provider_id for item in results) == SUPPORTED_PROVIDERS
    assert all(item.ok and item.committed for item in results)
    catalog = store.load()
    providers = {item.provider_id for item in catalog.models}
    assert providers == set(SUPPORTED_PROVIDERS)
    payload = json.loads(store.catalog_path.read_text(encoding="utf-8"))
    assert payload["catalog_schema_version"] == CATALOG_SCHEMA_VERSION == 1
    assert "lifecycle_changes" not in payload
    assert not (tmp_path / "cache" / "runs").exists()
    assert catalog.freshness.overlay is CatalogOverlay.USER_CACHE


def test_update_all_continues_after_one_provider_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_update(provider_id: str, **kwargs: object) -> CatalogUpdateResult:
        calls.append(provider_id)
        if provider_id == "google":
            return _stub_result(
                provider_id,
                ok=False,
                committed=False,
                error="google catalog refresh failed",
                catalog_path=tmp_path / "catalog.json",
            )
        return _stub_result(provider_id, catalog_path=tmp_path / "catalog.json")

    monkeypatch.setattr("samyak.model.update.update_provider_catalog", fake_update)
    results = update_all_catalogs(store=FileCatalogStore(root=tmp_path / "cache"), clock=_clock)
    assert calls == list(SUPPORTED_PROVIDERS)
    by_id = {item.provider_id: item for item in results}
    assert by_id["google"].ok is False
    assert by_id["google"].committed is False
    assert by_id["openai"].ok is True
    assert by_id["together"].ok is True
    assert by_id["google"].lifecycle_changes == ()


def test_failed_provider_does_not_overwrite_previous_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_all_transports(monkeypatch)
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_all_catalogs(store=store, clock=_clock)
    assert all(item.committed for item in first)
    before = store.load()
    google_ids = {item.samyak_id for item in before.models if item.provider_id == "google"}
    other_ids = {item.samyak_id for item in before.models if item.provider_id != "google"}
    assert google_ids

    _patch_all_transports(monkeypatch, google_pages=_failed_google_pages())
    second = update_all_catalogs(store=store, clock=_clock)
    by_id = {item.provider_id: item for item in second}
    assert by_id["google"].ok is False
    assert by_id["google"].committed is False
    assert by_id["google"].lifecycle_changes == ()
    assert by_id["openai"].committed is True
    after = store.load()
    assert {item.samyak_id for item in after.models if item.provider_id == "google"} == google_ids
    assert other_ids <= {item.samyak_id for item in after.models}


def test_successful_providers_persist_during_update_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_all_transports(monkeypatch)
    store = FileCatalogStore(root=tmp_path / "cache")
    results = update_all_catalogs(store=store, clock=_clock)
    loaded = store.load()
    for result in results:
        assert result.committed is True
        assert any(item.provider_id == result.provider_id for item in loaded.models)


def test_update_all_retains_per_provider_lifecycle_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_all_transports(monkeypatch)
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_all_catalogs(store=store, clock=_clock)
    assert all(item.lifecycle_changes == () for item in first)
    catalog = first[-1].catalog
    assert catalog is not None
    target = "openai:gpt-4.5-preview"
    rewritten = []
    for record in catalog.models:
        if record.samyak_id != target:
            rewritten.append(record)
            continue
        rewritten.append(
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
            models=rewritten,
            generated_at=catalog.generated_at,
            freshness=catalog.freshness,
            notices=catalog.notices,
            version=catalog.version,
        )
    )
    second = update_all_catalogs(store=store, clock=_clock)
    by_id = {item.provider_id: item for item in second}
    openai_changes = [
        item
        for item in by_id["openai"].lifecycle_changes
        if item.samyak_id == target and item.change_type is LifecycleChangeType.LIFECYCLE_CHANGED
    ]
    assert len(openai_changes) == 1
    assert all(item.lifecycle_changes == () for item in second if item.provider_id != "openai")


def test_first_update_all_does_not_invent_added_lifecycle_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_all_transports(monkeypatch)
    store = FileCatalogStore(root=tmp_path / "cache")
    assert store.exists() is False
    results = update_all_catalogs(store=store, clock=_clock)
    assert results[0].previous_existed is False
    for result in results:
        assert result.committed is True
        assert result.lifecycle_changes == ()
        assert all(
            item.change_type is not LifecycleChangeType.ADDED for item in result.lifecycle_changes
        )


def test_update_all_keeps_other_provider_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_all_transports(monkeypatch)
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_all_catalogs(store=store, clock=_clock)
    first_ids = {item.samyak_id for item in store.load().models}
    second = update_all_catalogs(store=store, clock=_clock)
    assert all(item.committed for item in first)
    assert all(item.committed for item in second)
    assert {item.samyak_id for item in store.load().models} == first_ids
    providers = {item.provider_id for item in store.load().models}
    assert providers == set(SUPPORTED_PROVIDERS)


def test_update_all_continues_after_post_commit_history_construction_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_all_transports(monkeypatch)
    store = FileCatalogStore(root=tmp_path / "cache")
    first = update_all_catalogs(store=store, clock=_clock)
    assert all(item.committed for item in first)
    catalog = store.load()
    target = "openai:gpt-4.5-preview"
    rewritten = [
        replace(
            record,
            lifecycle=Fact(
                status=FactStatus.KNOWN,
                value=LifecycleState.ACTIVE,
                provenance=record.lifecycle.provenance,
            ),
        )
        if record.samyak_id == target
        else record
        for record in catalog.models
    ]
    store.save(
        build_catalog(
            models=rewritten,
            generated_at=catalog.generated_at,
            freshness=catalog.freshness,
            notices=catalog.notices,
            version=catalog.version,
        )
    )

    def exploding_events(*_args: object, **_kwargs: object):
        raise CatalogValidationError("lifecycle history samyak_id does not match provider_id")

    monkeypatch.setattr("samyak.model.update.history_events_from_changes", exploding_events)
    second = update_all_catalogs(store=store, clock=_clock)
    assert [item.provider_id for item in second] == list(SUPPORTED_PROVIDERS)
    by_id = {item.provider_id: item for item in second}
    assert by_id["openai"].ok is True
    assert by_id["openai"].committed is True
    assert any(notice.code == HISTORY_WRITE_FAILED_CODE for notice in by_id["openai"].notices)
    assert by_id["together"].ok is True
    assert by_id["together"].committed is True
    loaded = store.load()
    current = next(item for item in loaded.models if item.samyak_id == target)
    assert current.lifecycle.value is not LifecycleState.ACTIVE


def test_cli_update_all_exits_nonzero_when_a_provider_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_all_transports(monkeypatch, google_pages=_failed_google_pages())
    code = main(["model", "update", "all"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Updating model catalogs" in captured.out
    assert "Google" in captured.out
    assert "Failed" in captured.out
    assert "OpenAI" in captured.out
    assert "Updated" in captured.out
    assert "1 provider failed" in captured.out
    assert "Provider announced" not in captured.out


def test_cli_update_all_exits_zero_when_all_providers_succeed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_all_transports(monkeypatch)
    code = main(["model", "update", "all"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Updating model catalogs" in captured.out
    for label in ("Anthropic", "Fireworks", "Google", "OpenAI", "Together"):
        assert label in captured.out
    assert "Failed" not in captured.out
    assert "5 providers updated" in captured.out
    assert "0 providers failed" in captured.out
    assert "0 lifecycle changes detected" in captured.out
    assert "Lifecycle changes\n" not in captured.out
    store = FileCatalogStore()
    providers = {item.provider_id for item in store.load().models}
    assert providers == set(SUPPORTED_PROVIDERS)


def test_cli_update_all_reports_retained_lifecycle_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_all(**kwargs: object) -> tuple[CatalogUpdateResult, ...]:
        return tuple(
            _stub_result(
                provider_id,
                lifecycle_changes=(_lifecycle_change(),) if provider_id == "openai" else (),
            )
            for provider_id in SUPPORTED_PROVIDERS
        )

    monkeypatch.setattr("samyak.cli.update_all_catalogs", fake_all)
    code = main(["model", "update", "all"])
    captured = capsys.readouterr()
    assert code == 0
    assert "OpenAI" in captured.out
    assert "Lifecycle changes: 1" in captured.out
    assert "openai:gpt-test" in captured.out
    assert "DEPRECATED" in captured.out
    assert "ACTIVE → DEPRECATED" in captured.out
    assert "1 lifecycle change detected" in captured.out


def test_cli_update_all_reports_progress_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_all_transports(monkeypatch)
    code = main(["model", "update", "all"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Updating Anthropic (1/5)" in captured.err
    assert "Updating Together (5/5)" in captured.err
    assert " done" in captured.err
    assert captured.err.count(".") >= 5
    assert "Updating model catalogs" in captured.out
    assert "Updating Anthropic (1/5)" not in captured.out


def test_cli_update_all_progress_continues_after_a_provider_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_all_transports(monkeypatch, google_pages=_failed_google_pages())
    code = main(["model", "update", "all"])
    captured = capsys.readouterr()
    assert code == 2
    assert "Updating Google (3/5)" in captured.err
    assert " failed" in captured.err
    assert "Updating OpenAI (4/5)" in captured.err
    assert "Updating Together (5/5)" in captured.err


def test_cli_update_help_includes_all(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["model", "update", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "all" in out
    for provider_id in SUPPORTED_PROVIDERS:
        assert provider_id in out


def test_cli_single_provider_update_remains_unchanged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_all_transports(monkeypatch)
    code = main(["model", "update", "openai"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Updated OpenAI model catalog." in captured.out
    assert "Updating model catalogs" not in captured.out
    assert "5 providers updated" not in captured.out
    store = FileCatalogStore()
    providers = {item.provider_id for item in store.load().models}
    assert providers == {"openai"}
