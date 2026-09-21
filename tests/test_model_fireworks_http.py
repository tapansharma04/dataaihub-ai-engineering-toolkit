"""HTTP capture tests for Fireworks AI Model Intelligence. No live network."""

from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest

from samyak.model.http import FixtureHop, FixtureTransport, TransportSecurityError
from samyak.model.providers.fireworks.fetch import (
    capture_fireworks_source,
    fireworks_docs_policy,
    refresh_fireworks_catalog,
)
from samyak.model.providers.fireworks.sources import (
    CHANGELOG_URL,
    EMBEDDINGS_URL,
    KIMI_K2_URL,
    SERVING_PATHS_URL,
    TEXT_MODELS_URL,
    TOOL_CALLING_URL,
    VISION_MODELS_URL,
    embeddings_descriptor,
    expected_path_for_source,
    model_page_url,
    required_source_descriptors,
    serving_paths_descriptor,
)

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "fireworks"
FIREWORKS_PROVIDER = Path(__file__).resolve().parents[1] / "src/samyak/model/providers/fireworks"
STAMP = "2026-09-12T12:00:00+00:00"


@pytest.fixture(autouse=True)
def deny_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Model Intelligence HTTP tests must not use the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("urllib.request.urlopen", blocked)


def _md(relative: str, **kwargs: object) -> FixtureHop:
    return FixtureHop(body=(FIXTURES / relative).read_bytes(), **kwargs)  # type: ignore[arg-type]


def _transport(pages: dict[str, FixtureHop], *, clock: object | None = None) -> FixtureTransport:
    return FixtureTransport(
        pages,
        policy=fireworks_docs_policy(),
        clock=clock if callable(clock) else (lambda: STAMP),
    )


def _official_pages(**overrides: FixtureHop) -> dict[str, FixtureHop]:
    pages = {
        EMBEDDINGS_URL: _md("embeddings.md"),
        SERVING_PATHS_URL: _md("serving-paths.md"),
        CHANGELOG_URL: _md("changelog.md"),
        TEXT_MODELS_URL: _md("text-models.md"),
        VISION_MODELS_URL: _md("vision-models.md"),
        TOOL_CALLING_URL: _md("tool-calling.md"),
        KIMI_K2_URL: _md("kimi-k2.md"),
    }
    pages.update(overrides)
    return pages


def test_parser_modules_do_not_import_http() -> None:
    for name in ("parse.py", "normalize.py", "observations.py", "sources.py"):
        path = FIREWORKS_PROVIDER / name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert "samyak.model.http" not in imported
        assert "urllib.request" not in imported
        assert "samyak.model.providers.fireworks.fetch" not in imported


def test_rejects_non_https() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="HTTPS"):
        transport.get("http://docs.fireworks.ai/guides/querying-embeddings-models.md")


def test_rejects_unapproved_host() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://evil.example/guides/querying-embeddings-models.md")


def test_rejects_library_app_and_api_hosts() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://fireworks.ai/models")
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://app.fireworks.ai/models")
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://api.fireworks.ai/inference/v1/models")


def test_rejects_unapproved_path() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation path"):
        transport.get("https://docs.fireworks.ai/api-reference/list-models")
    with pytest.raises(TransportSecurityError, match="approved documentation path"):
        transport.get("https://docs.fireworks.ai/pricing.md")


def test_rejects_query_and_fragment() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="query"):
        transport.get("https://docs.fireworks.ai/guides/querying-embeddings-models.md?key=1")
    with pytest.raises(TransportSecurityError, match="fragment"):
        transport.get("https://docs.fireworks.ai/guides/querying-embeddings-models.md#top")


def test_rejects_credentials_in_url() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="credentials"):
        transport.get("https://user:pass@docs.fireworks.ai/guides/querying-embeddings-models.md")


def test_capture_required_sources() -> None:
    transport = _transport(_official_pages())
    embeddings = capture_fireworks_source(transport, embeddings_descriptor())
    serving = capture_fireworks_source(transport, serving_paths_descriptor())
    assert embeddings.source_id == "fireworks-docs-embeddings"
    assert serving.source_id == "fireworks-docs-serving-paths"
    assert serving.source_url == SERVING_PATHS_URL
    assert serving.source_url == "https://docs.fireworks.ai/serverless/serverless-modes.md"
    assert embeddings.retrieved_at == STAMP


def test_refresh_documents_subset_without_marking_partial() -> None:
    result = refresh_fireworks_catalog(
        _transport(_official_pages()),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is False
    assert result.catalog is not None
    codes = {notice.code for notice in result.notices}
    assert "DOCUMENTED_SUBSET:fireworks" in codes
    assert "PARTIAL_MODEL_PAGES:fireworks" not in codes
    assert any(item.provider_id == "fireworks" for item in result.catalog.models)
    ids = {item.provider_model_id for item in result.catalog.models}
    assert "accounts/fireworks/routers/glm-5p2-fast" in ids
    assert "accounts/fireworks/routers/kimi-k3-fast" in ids
    serving_sources = [
        item
        for item in result.catalog.freshness.source_retrieved_at
        if item.source_id == "fireworks-docs-serving-paths"
    ]
    assert serving_sources
    glm_fast = next(
        item
        for item in result.catalog.models
        if item.provider_model_id == "accounts/fireworks/routers/glm-5p2-fast"
    )
    assert glm_fast.api_access.provenance is not None
    assert glm_fast.api_access.provenance.source_id == "fireworks-docs-serving-paths"
    assert glm_fast.api_access.provenance.source_url == SERVING_PATHS_URL


def test_refresh_partial_when_optional_page_missing() -> None:
    pages = _official_pages()
    del pages[KIMI_K2_URL]
    result = refresh_fireworks_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is True
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:fireworks" in codes
    assert "DOCUMENTED_SUBSET:fireworks" in codes
    assert "SOURCE_FETCH_FAILED:fireworks-docs-model-page:kimi-k2" in codes


def test_refresh_partial_when_optional_page_malformed() -> None:
    result = refresh_fireworks_catalog(
        _transport(_official_pages(**{KIMI_K2_URL: _md("missing-id.md")})),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is True
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:fireworks" in codes
    assert "SOURCE_PARSE_FAILED:fireworks-docs-model-page:kimi-k2" in codes


def test_refresh_fails_closed_on_required_source() -> None:
    pages = _official_pages()
    pages[EMBEDDINGS_URL] = FixtureHop(status=500, body=b"error")
    result = refresh_fireworks_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is False
    assert result.catalog is None
    assert "500" in (result.error or "")


def test_malformed_required_source_fails_refresh() -> None:
    pages = _official_pages()
    pages[EMBEDDINGS_URL] = _md("malformed.md")
    result = refresh_fireworks_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is False
    assert result.catalog is None


def test_official_sources_are_markdown_on_docs_host() -> None:
    assert EMBEDDINGS_URL == "https://docs.fireworks.ai/guides/querying-embeddings-models.md"
    assert SERVING_PATHS_URL == "https://docs.fireworks.ai/serverless/serverless-modes.md"
    assert SERVING_PATHS_URL != "https://docs.fireworks.ai/serverless/serving-paths.md"
    assert CHANGELOG_URL == "https://docs.fireworks.ai/updates/changelog.md"
    assert model_page_url("kimi-k2").endswith("/models/kimi-k2.md")
    for descriptor in required_source_descriptors():
        path = expected_path_for_source(descriptor)
        assert path.endswith(".md")
        assert not path.endswith(".md.txt")


def test_serving_paths_descriptor_requests_serverless_modes() -> None:
    descriptor = serving_paths_descriptor()
    assert descriptor.source_id == "fireworks-docs-serving-paths"
    assert descriptor.url == "https://docs.fireworks.ai/serverless/serverless-modes.md"
    assert expected_path_for_source(descriptor) == "/serverless/serverless-modes.md"
    assert "/serverless/serving-paths.md" not in descriptor.url
