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
from samyak.server.model_catalog import filter_models
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
        assert "Serving sources: openai: 3" in html
        assert "GPT Active" in html
        assert "openai:gpt-active" in html
        assert "400K" in html
        assert "272K" in html
        assert "128K" in html
        assert "<th>Context</th>" in html
        assert "<th>Max input</th>" in html
        assert "<th>Max output</th>" in html
        assert "As of" in html
        assert "user cache" in html
        assert "GPT Retired" in html
        assert (
            '<select name="provider" aria-label="Provider">'
            '<option value="all" selected>All providers</option>'
        ) in html
        assert (
            '<select name="lifecycle" aria-label="Lifecycle">'
            '<option value="all" selected>All lifecycle states</option>'
        ) in html
        assert 'name="provider"' in html
        assert "OpenAI" in html
        assert 'option value="openai"' in html
        assert "Catalog is partial" in html
        assert "Some provider documentation could not be retrieved." in html
        assert "<details" in html
        assert "PARTIAL_MODEL_PAGES" in html
        assert "This catalog is incomplete" in html
        assert html.index("Catalog is partial") < html.index("This catalog is incomplete")
        assert html.index("<summary>") < html.index("PARTIAL_MODEL_PAGES")
        assert "samyak model update openai" in html
        assert ">Apply</button>" in html

        with _get(base + "/models?lifecycle=retired") as response:
            retired = response.read().decode("utf-8")
        assert "GPT Retired" in retired
        assert "GPT Active" not in retired
        assert 'option value="retired" selected' in retired

        with _get(base + "/models?lifecycle=active") as response:
            active = response.read().decode("utf-8")
        assert "GPT Active" in active
        assert "GPT Retired" not in active
        assert 'option value="active" selected' in active

        with _get(base + "/models?q=active-alias") as response:
            search = response.read().decode("utf-8")
        assert "GPT Active" in search
        assert "GPT Retired" not in search

        with _get(base + "/models?q=no-such-model") as response:
            empty = response.read().decode("utf-8")
        assert "No models match these filters" in empty

        with _get(base + "/models?provider=openai") as response:
            openai_only = response.read().decode("utf-8")
        assert "GPT Active" in openai_only
        assert 'option value="openai" selected' in openai_only

        with _get(base + "/models?provider=unknown-source") as response:
            unknown_provider = response.read().decode("utf-8")
        assert "GPT Active" in unknown_provider
        assert 'option value="all" selected' in unknown_provider
    finally:
        _stop(server, thread)


def test_filter_models_unknown_provider_falls_back_to_all() -> None:
    catalog = _sample_catalog()
    matched = filter_models(catalog.models, provider="nope")
    assert {item.samyak_id for item in matched} == {item.samyak_id for item in catalog.models}
    openai_only = filter_models(catalog.models, provider="openai")
    assert openai_only == catalog.models


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
        assert "Serving source" in active
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


def test_context_window_status_is_not_confused_with_kind(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    models = (
        _model(
            "gpt-known-ctx",
            display_name=_known("Known Ctx"),
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=_known(ContextWindow(tokens=400000, kind=ContextWindowKind.UNKNOWN)),
        ),
        _model(
            "gpt-unknown-ctx",
            display_name=_known("Unknown Ctx"),
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=_unknown(),
        ),
        _model(
            "gpt-unverified-ctx",
            display_name=_known("Unverified Ctx"),
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=Fact(status=FactStatus.NOT_VERIFIED),
        ),
        _model(
            "gpt-conflict-ctx",
            display_name=_known("Conflict Ctx"),
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=_conflict(
                ContextWindow(tokens=1000, kind=ContextWindowKind.UNKNOWN),
                ContextWindow(tokens=2000, kind=ContextWindowKind.UNKNOWN),
            ),
        ),
    )
    FileCatalogStore(root=root).save(_catalog(models))
    server, thread, _ = _start(root)
    try:
        base = viewer_url(server).rstrip("/")
        with _get(base + "/models") as response:
            listing = response.read().decode("utf-8")
        assert "400K" in listing
        assert '<span class="fact-unknown">Unknown</span>' in listing
        assert '<span class="fact-not-verified">Not verified</span>' in listing
        assert '<span class="fact-conflict">Conflict</span>' in listing
        assert "<th>Max input</th>" in listing
        assert "<th>Max output</th>" in listing

        with _get(base + "/models/openai:gpt-known-ctx") as response:
            known = response.read().decode("utf-8")
        assert "400K" in known
        assert "(unknown)" not in known.lower()
        assert "<dt>Context window</dt><dd>400K</dd>" in known.replace("\n", "")

        with _get(base + "/models/openai:gpt-unknown-ctx") as response:
            unknown = response.read().decode("utf-8")
        assert '<span class="fact-unknown">Unknown</span>' in unknown
        assert "400K" not in unknown

        with _get(base + "/models/openai:gpt-unverified-ctx") as response:
            unverified = response.read().decode("utf-8")
        assert '<span class="fact-not-verified">Not verified</span>' in unverified

        with _get(base + "/models/openai:gpt-conflict-ctx") as response:
            conflict = response.read().decode("utf-8")
        assert "fact-conflict" in conflict
        assert "did not resolve competing evidence" in conflict
    finally:
        _stop(server, thread)


def test_catalog_table_shows_independent_token_limits(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    models = (
        _model(
            "gpt-known-all",
            display_name=_known("Known All"),
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=_known(ContextWindow(tokens=1_000_000, kind=ContextWindowKind.UNKNOWN)),
            max_input_tokens=_known(192000),
            max_output_tokens=_known(128000),
        ),
        _model(
            "gpt-unknown-ctx-known-io",
            display_name=_known("Unknown Ctx Known IO"),
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=_unknown(),
            max_input_tokens=_known(32000),
            max_output_tokens=_known(8000),
        ),
        _model(
            "gpt-unknown-all",
            display_name=_known("Unknown All"),
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=_unknown(),
            max_input_tokens=_unknown(),
            max_output_tokens=_unknown(),
        ),
        _model(
            "gpt-unverified-limits",
            display_name=_known("Unverified Limits"),
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=Fact(status=FactStatus.NOT_VERIFIED),
            max_input_tokens=Fact(status=FactStatus.NOT_VERIFIED),
            max_output_tokens=Fact(status=FactStatus.NOT_VERIFIED),
        ),
    )
    FileCatalogStore(root=root).save(_catalog(models))
    server, thread, _ = _start(root)
    try:
        base = viewer_url(server).rstrip("/")
        with _get(base + "/models") as response:
            listing = response.read().decode("utf-8")
        assert "<th>Context</th>" in listing
        assert "<th>Max input</th>" in listing
        assert "<th>Max output</th>" in listing
        known_row = _table_row(listing, "openai:gpt-known-all")
        assert "1M" in known_row
        assert "192K" in known_row
        assert "128K" in known_row
        assert "Unknown" not in known_row
        io_row = _table_row(listing, "openai:gpt-unknown-ctx-known-io")
        assert "Unknown" in io_row
        assert "32K" in io_row
        assert "8K" in io_row
        assert "1M" not in io_row
        unknown_row = _table_row(listing, "openai:gpt-unknown-all")
        assert unknown_row.count("Unknown") >= 3
        assert "1M" not in unknown_row
        assert "32K" not in unknown_row
        unverified_row = _table_row(listing, "openai:gpt-unverified-limits")
        assert unverified_row.count("Not verified") >= 3
        assert "1M" not in unverified_row

        with _get(base + "/models/openai:gpt-unknown-ctx-known-io") as response:
            detail = response.read().decode("utf-8").replace("\n", "")
        assert "<dt>Context window</dt><dd>" in detail
        assert "<dt>Max input tokens</dt><dd>32K</dd>" in detail
        assert "<dt>Max output tokens</dt><dd>8K</dd>" in detail
        assert "32K" in detail
        assert "8K" in detail
    finally:
        _stop(server, thread)


def _table_row(html: str, samyak_id: str) -> str:
    marker = f'<div class="muted id">{samyak_id}</div>'
    start = html.find(marker)
    assert start != -1, f"missing {samyak_id}"
    row_start = html.rfind("<tr>", 0, start)
    row_end = html.find("</tr>", start)
    assert row_start != -1 and row_end != -1
    return html[row_start:row_end]


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
        assert "Serving sources: anthropic: 1, openai: 3" in html
        assert "anthropic:claude-opus-5" in html
        assert "openai:gpt-active" in html
        assert "GPT Retired" in html
        assert (
            '<select name="provider" aria-label="Provider">'
            '<option value="all" selected>All providers</option>'
        ) in html
        assert (
            '<select name="lifecycle" aria-label="Lifecycle">'
            '<option value="all" selected>All lifecycle states</option>'
        ) in html
        assert 'option value="anthropic">Anthropic</option>' in html
        assert 'option value="openai">OpenAI</option>' in html
        with _get(base + "/models?provider=anthropic") as response:
            anthropic_only = response.read().decode("utf-8")
        assert "anthropic:claude-opus-5" in anthropic_only
        assert "openai:gpt-active" not in anthropic_only
        assert "GPT Active" not in anthropic_only
        with _get(base + "/models?provider=anthropic&lifecycle=retired") as response:
            combined = response.read().decode("utf-8")
        assert "No models match these filters" in combined
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
        assert "samyak model update fireworks" in home
        assert "samyak model update together" in home
    finally:
        _stop(server, thread)


def test_partial_catalog_notices_are_behind_details(tmp_path: Path) -> None:
    notices = (
        CatalogNotice(
            code="PARTIAL_MODEL_PAGES:fireworks",
            message=(
                "One or more Fireworks documentation pages could not be retrieved; "
                "facts from those pages are omitted"
            ),
        ),
        CatalogNotice(
            code="DOCUMENTED_SUBSET:fireworks",
            message=(
                "Fireworks coverage is the documented public serving IDs in official "
                "Markdown, not the complete Fireworks Model Library"
            ),
        ),
        CatalogNotice(
            code="SOURCE_FETCH_FAILED:fireworks-docs-model-page:kimi-k2",
            message=(
                "Fireworks documentation fireworks-docs-model-page:kimi-k2 could not be retrieved"
            ),
        ),
    )
    root = tmp_path / "cache"
    FileCatalogStore(root=root).save(_catalog(_sample_catalog().models, notices=notices))
    server, thread, _ = _start(root)
    try:
        with _get(viewer_url(server).rstrip("/") + "/models") as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
        assert "Catalog is partial" in html
        assert "Some provider documentation could not be retrieved." in html
        assert "This catalog is incomplete" in html
        assert "PARTIAL_MODEL_PAGES:fireworks" in html
        assert "DOCUMENTED_SUBSET:fireworks" in html
        assert "SOURCE_FETCH_FAILED:fireworks-docs-model-page:kimi-k2" in html
        assert html.index("Catalog is partial") < html.index("This catalog is incomplete")
        assert html.index("<summary>") < html.index("PARTIAL_MODEL_PAGES:fireworks")
        assert "samyak model update fireworks" in html
        assert "/models/openai:gpt-active" in html
    finally:
        _stop(server, thread)


def test_fireworks_serving_source_is_labeled_without_origin_columns(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    fireworks = ModelRecord(
        identity=ModelIdentity(
            provider_id="fireworks",
            provider_model_id="accounts/fireworks/models/deepseek-v3p1",
        ),
        identity_kind=IdentityKind.CANONICAL,
        display_name=_known("DeepSeek V3.1", provider="fireworks"),
        lifecycle=_known(LifecycleState.ACTIVE, provider="fireworks"),
    )
    FileCatalogStore(root=root).save(_catalog(_sample_catalog().models + (fireworks,)))
    server, thread, _ = _start(root)
    try:
        base = viewer_url(server).rstrip("/")
        with _get(base + "/models") as response:
            html = response.read().decode("utf-8")
        assert "Serving sources: fireworks: 1, openai: 3" in html
        assert "fireworks:accounts/fireworks/models/deepseek-v3p1" in html
        assert "Creator" not in html
        assert "Origin" not in html
        with _get(base + "/models/fireworks:accounts/fireworks/models/deepseek-v3p1") as response:
            detail = response.read().decode("utf-8")
            assert response.status == 200
        assert "Serving source" in detail
        assert "Fireworks" in detail
        assert "Creator" not in detail
        assert "Origin" not in detail
    finally:
        _stop(server, thread)


def test_together_serving_source_is_labeled_without_origin_columns(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    together = ModelRecord(
        identity=ModelIdentity(
            provider_id="together",
            provider_model_id="openai/gpt-oss-120b",
        ),
        identity_kind=IdentityKind.CANONICAL,
        display_name=_known("GPT-OSS 120B", provider="together"),
        lifecycle=_known(LifecycleState.ACTIVE, provider="together"),
    )
    FileCatalogStore(root=root).save(_catalog(_sample_catalog().models + (together,)))
    server, thread, _ = _start(root)
    try:
        base = viewer_url(server).rstrip("/")
        with _get(base + "/models") as response:
            html = response.read().decode("utf-8")
        assert "Serving sources: openai: 3, together: 1" in html
        assert "together:openai/gpt-oss-120b" in html
        assert "Creator" not in html
        assert "Origin" not in html
        assert "samyak model update together" in html
        with _get(base + "/models/together:openai/gpt-oss-120b") as response:
            detail = response.read().decode("utf-8")
            assert response.status == 200
        assert "Serving source" in detail
        assert "Together" in detail
        assert "Creator" not in detail
        assert "Origin" not in detail
    finally:
        _stop(server, thread)
