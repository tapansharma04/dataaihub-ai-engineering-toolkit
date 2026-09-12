"""Tests for Model Intelligence pages in the local viewer."""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from samyak.model.catalog import (
    CatalogFreshness,
    CatalogNotice,
    CatalogOverlay,
    FreshnessStatus,
    build_catalog,
)
from samyak.model.facts import (
    Confidence,
    ContextWindow,
    ContextWindowKind,
    Fact,
    FactClaim,
    FactStatus,
    LifecycleState,
    Modality,
    Provenance,
    SourceKind,
)
from samyak.model.identity import IdentityKind, ModelIdentity
from samyak.model.records import ModelRecord
from samyak.model.store import FileCatalogStore
from samyak.server.app import create_server, viewer_url
from samyak.store.filesystem import FileRunStore

GENERATED_AT = "2026-09-10T17:35:13+00:00"


def _provenance(**overrides: object) -> Provenance:
    payload: dict[str, object] = {
        "provider": "openai",
        "source_id": "openai-docs-model-page",
        "source_kind": SourceKind.PROVIDER_DOCS,
        "source_url": "https://example.invalid/models/gpt-test.md",
        "content_hash": "sha256:abc",
        "retrieved_at": "2026-09-10T17:30:00+00:00",
        "observed_at": None,
        "verified_at": "2026-09-10T17:35:13+00:00",
        "confidence": Confidence.HIGH,
    }
    payload.update(overrides)
    return Provenance(**payload)  # type: ignore[arg-type]


def _known(value: object, **overrides: object) -> Fact:
    return Fact(status=FactStatus.KNOWN, value=value, provenance=_provenance(**overrides))


def _unknown(**overrides: object) -> Fact:
    return Fact(status=FactStatus.UNKNOWN, provenance=_provenance(**overrides))


def _conflict(*values: object) -> Fact:
    claims = tuple(
        FactClaim(value=value, provenance=_provenance(source_id=f"source-{index}"))
        for index, value in enumerate(values)
    )
    return Fact(status=FactStatus.CONFLICT, claims=claims)


def _model(provider_model_id: str, **fields: object) -> ModelRecord:
    return ModelRecord(
        identity=ModelIdentity(provider_id="openai", provider_model_id=provider_model_id),
        identity_kind=IdentityKind.CANONICAL,
        **fields,  # type: ignore[arg-type]
    )


def _catalog(models: tuple[ModelRecord, ...], *, notices=()):
    return build_catalog(
        models=models,
        generated_at=GENERATED_AT,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=GENERATED_AT,
            overlay=CatalogOverlay.USER_CACHE,
            oldest_verified_at=GENERATED_AT,
        ),
        notices=notices,
    )


def _sample_catalog():
    dangling = ModelIdentity(provider_id="openai", provider_model_id="missing-successor")
    models = (
        _model(
            "gpt-active",
            display_name=_known("GPT Active"),
            aliases=_known(("active-alias",)),
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=_known(ContextWindow(tokens=400000, kind=ContextWindowKind.UNKNOWN)),
            max_input_tokens=_known(272000),
            max_output_tokens=_known(128000),
            tool_calling=_known(True),
            structured_output=_known(True),
            api_access=_known(True),
            input_modalities=_known((Modality.TEXT,)),
            output_modalities=_known((Modality.TEXT,)),
        ),
        _model(
            "gpt-retired",
            display_name=_known("GPT Retired"),
            lifecycle=_known(LifecycleState.RETIRED),
            replacement=_known((dangling,)),
            api_access=_known(False),
            tool_calling=_unknown(),
            structured_output=Fact(status=FactStatus.NOT_VERIFIED),
            family=Fact(status=FactStatus.NOT_APPLICABLE, provenance=_provenance()),
        ),
        _model(
            "gpt-conflict",
            lifecycle=_known(LifecycleState.DEPRECATED),
            tool_calling=_conflict(True, False),
            display_name=_known('<script>alert("x")</script>'),
        ),
    )
    notices = (
        CatalogNotice(
            code="PARTIAL_MODEL_PAGES",
            message="Some model pages could not be retrieved",
        ),
    )
    return _catalog(models, notices=notices)


def _start(run_root: Path):
    run_store = FileRunStore(root=run_root)
    catalog_store = FileCatalogStore(root=run_root)
    server = create_server(run_store, catalog_store, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, catalog_store


def _stop(server, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _get(url: str):
    return urlopen(url, timeout=5)


def test_workspace_is_home_and_lists_sections(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    FileCatalogStore(root=root).save(_sample_catalog())
    server, thread, _ = _start(root)
    try:
        with _get(viewer_url(server)) as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
        assert "Local workspace" in html
        assert "Corpus Intelligence" in html
        assert "Model Intelligence" in html
        assert "Run History" in html
        assert "Model Catalog" in html
        assert "3 models" in html
        assert "1 active" in html
        assert "1 deprecated" in html
        assert "1 retired" in html
        assert "As of Sep 10, 2026" in html
        assert "Incomplete" in html
        assert "Browse Model Catalog" in html
        assert 'href="/runs"' in html
        assert 'href="/models"' in html
        assert "<script>" not in html
    finally:
        _stop(server, thread)


def test_workspace_missing_catalog_is_onboarding(tmp_path: Path) -> None:
    server, thread, _ = _start(tmp_path / "cache")
    try:
        with _get(viewer_url(server)) as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
        assert "No local model catalog yet" in html
        assert "samyak model update openai" in html
        assert "Traceback" not in html
    finally:
        _stop(server, thread)


def test_models_renders_catalog_and_filters(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    FileCatalogStore(root=root).save(_sample_catalog())
    server, thread, _ = _start(root)
    try:
        base = viewer_url(server).rstrip("/")
        with _get(base + "/models") as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
        assert "Model Catalog" in html
        assert "Providers: openai: 3" in html
        assert "GPT Active" in html
        assert "openai:gpt-active" in html
        assert "400K" in html
        assert "As of" in html
        assert "user cache" in html
        assert "PARTIAL_MODEL_PAGES" in html
        assert "incomplete" in html.lower()
        assert "/models/openai:gpt-active" in html

        with _get(base + "/models?lifecycle=retired") as response:
            retired = response.read().decode("utf-8")
        assert "GPT Retired" in retired
        assert "GPT Active" not in retired

        with _get(base + "/models?q=active-alias") as response:
            search = response.read().decode("utf-8")
        assert "GPT Active" in search
        assert "GPT Retired" not in search

        with _get(base + "/models?q=no-such-model") as response:
            empty = response.read().decode("utf-8")
        assert "No models match this search" in empty
    finally:
        _stop(server, thread)


def test_empty_catalog_renders(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    FileCatalogStore(root=root).save(_catalog(()))
    server, thread, _ = _start(root)
    try:
        with _get(viewer_url(server).rstrip("/") + "/models") as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
        assert "This catalog contains no models" in html
        with _get(viewer_url(server)) as workspace:
            home = workspace.read().decode("utf-8")
        assert "0 models" in home
    finally:
        _stop(server, thread)


def test_missing_catalog_models_onboarding(tmp_path: Path) -> None:
    server, thread, _ = _start(tmp_path / "cache")
    try:
        with _get(viewer_url(server).rstrip("/") + "/models") as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
        assert "No local model catalog yet" in html
        assert "samyak model update openai" in html
    finally:
        _stop(server, thread)


def test_corrupt_catalog_returns_409(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    store = FileCatalogStore(root=root)
    store.catalog_path.parent.mkdir(parents=True)
    store.catalog_path.write_text("{not-json", encoding="utf-8")
    server, thread, _ = _start(root)
    try:
        with pytest.raises(HTTPError) as exc:
            urlopen(viewer_url(server).rstrip("/") + "/models", timeout=5)
        assert exc.value.code == 409
        body = exc.value.read().decode("utf-8")
        assert "could not be read" in body
        assert "Traceback" not in body
        assert str(root) not in body
    finally:
        _stop(server, thread)


def test_unsupported_schema_returns_409(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    store = FileCatalogStore(root=root)
    store.save(_catalog((_model("gpt-one"),)))
    payload = json.loads(store.catalog_path.read_text(encoding="utf-8"))
    payload["catalog_schema_version"] = 99
    store.catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    server, thread, _ = _start(root)
    try:
        with pytest.raises(HTTPError) as exc:
            urlopen(viewer_url(server).rstrip("/") + "/models", timeout=5)
        assert exc.value.code == 409
        body = exc.value.read().decode("utf-8")
        assert "unsupported" in body.lower()
        assert "Traceback" not in body
    finally:
        _stop(server, thread)


def test_model_detail_fact_statuses(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    FileCatalogStore(root=root).save(_sample_catalog())
    server, thread, _ = _start(root)
    try:
        base = viewer_url(server).rstrip("/")
        with _get(base + "/models/openai:gpt-active") as response:
            active = response.read().decode("utf-8")
            assert response.status == 200
        assert "GPT Active" in active
        assert "openai:gpt-active" in active
        assert "OpenAI" in active
        assert "canonical" in active
        assert "Yes" in active
        assert "400K" in active
        assert "Evidence" in active
        assert "sha256:abc" in active
        assert "https://example.invalid/models/gpt-test.md" in active
        assert 'href="https://example.invalid' not in active
        assert "<script>" not in active

        with _get(base + "/models/openai:gpt-retired") as response:
            retired = response.read().decode("utf-8")
        assert "No" in retired
        assert "Unknown" in retired
        assert "Not verified" in retired
        assert "Not applicable" in retired
        assert "openai:missing-successor" in retired
        assert "not in this catalog" in retired

        with _get(base + "/models/openai:gpt-conflict") as response:
            conflict = response.read().decode("utf-8")
        assert "Conflict" in conflict
        assert "did not resolve competing evidence" in conflict
        assert "&lt;script&gt;" in conflict
        assert "<script>" not in conflict
    finally:
        _stop(server, thread)


def test_unknown_model_returns_404(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    FileCatalogStore(root=root).save(_sample_catalog())
    server, thread, _ = _start(root)
    try:
        with pytest.raises(HTTPError) as exc:
            urlopen(viewer_url(server).rstrip("/") + "/models/openai:nope", timeout=5)
        assert exc.value.code == 404
        body = exc.value.read().decode("utf-8")
        assert "not found" in body.lower()
        assert "Traceback" not in body
    finally:
        _stop(server, thread)


def test_malicious_model_id_stays_in_catalog_lookup(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    FileCatalogStore(root=root).save(_sample_catalog())
    server, thread, _ = _start(root)
    try:
        host, port = server.server_address[:2]
        for suffix in (
            "/models/../etc/passwd",
            "/models/%2e%2e/etc/passwd",
            "/models/<script>alert(1)</script>",
        ):
            conn = http.client.HTTPConnection(host, port, timeout=5)
            try:
                conn.request("GET", suffix)
                response = conn.getresponse()
                body = response.read().decode("utf-8")
            finally:
                conn.close()
            assert response.status in {404, 400}
            assert "Traceback" not in body
            assert "<script>" not in body
    finally:
        _stop(server, thread)


def test_viewer_does_not_use_https_or_mutate_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    store = FileCatalogStore(root=root)
    store.save(_sample_catalog())
    before = store.catalog_path.read_bytes()
    mtime = store.catalog_path.stat().st_mtime_ns

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("viewer must not open HTTPS")

    monkeypatch.setattr(http.client.HTTPSConnection, "connect", boom)
    monkeypatch.setattr(FileCatalogStore, "save", no_catalog_write)
    server, thread, _ = _start(root)
    try:
        base = viewer_url(server).rstrip("/")
        for path in ("/", "/runs", "/models", "/models/openai:gpt-active"):
            with _get(base + path) as response:
                assert response.status == 200
                response.read()
    finally:
        _stop(server, thread)
    assert store.catalog_path.read_bytes() == before
    assert store.catalog_path.stat().st_mtime_ns == mtime


def no_catalog_write(*args: object, **kwargs: object) -> None:
    raise AssertionError("viewer must not write the catalog")


def test_multi_provider_catalog_lists_both_and_opens_anthropic_detail(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    anthropic = ModelRecord(
        identity=ModelIdentity(provider_id="anthropic", provider_model_id="claude-opus-5"),
        identity_kind=IdentityKind.CANONICAL,
        display_name=_known("Claude Opus 5", provider="anthropic"),
        lifecycle=_known(LifecycleState.ACTIVE, provider="anthropic"),
        context_window=_known(
            ContextWindow(tokens=1_000_000, kind=ContextWindowKind.UNKNOWN),
            provider="anthropic",
        ),
    )
    FileCatalogStore(root=root).save(_catalog(_sample_catalog().models + (anthropic,)))
    server, thread, _ = _start(root)
    try:
        base = viewer_url(server).rstrip("/")
        with _get(base + "/models") as response:
            html = response.read().decode("utf-8")
        assert "Providers: anthropic: 1, openai: 3" in html
        assert "anthropic:claude-opus-5" in html
        assert "openai:gpt-active" in html
        with _get(base + "/models/anthropic:claude-opus-5") as response:
            detail = response.read().decode("utf-8")
            assert response.status == 200
        assert "Claude Opus 5" in detail
        assert "anthropic:claude-opus-5" in detail
        assert "1M" in detail
        with _get(base) as workspace:
            home = workspace.read().decode("utf-8")
        assert "4 models" in home
        assert "samyak model update anthropic" in home
    finally:
        _stop(server, thread)
