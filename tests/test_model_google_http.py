"""HTTP capture tests for Google Gemini API Model Intelligence. No live network."""

from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest

from samyak.model.http import FixtureHop, FixtureTransport, TransportSecurityError
from samyak.model.providers.google.fetch import (
    capture_google_source,
    google_docs_policy,
    refresh_google_catalog,
)
from samyak.model.providers.google.sources import (
    DEPRECATIONS_URL,
    GEMINI_3_URL,
    MODELS_INDEX_URL,
    deprecations_descriptor,
    expected_path_for_source,
    gemini_3_descriptor,
    model_page_url,
    models_index_descriptor,
    required_source_descriptors,
)

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "google"
GOOGLE_PROVIDER = Path(__file__).resolve().parents[1] / "src/samyak/model/providers/google"
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
        policy=google_docs_policy(),
        clock=clock if callable(clock) else (lambda: STAMP),
    )


def _official_pages(**overrides: FixtureHop) -> dict[str, FixtureHop]:
    pages = {
        MODELS_INDEX_URL: _md("models.md"),
        DEPRECATIONS_URL: _md("deprecations.md"),
        GEMINI_3_URL: _md("gemini-3.md"),
        model_page_url("gemini-3.8-flash"): _md("models/gemini-3.8-flash.md"),
        model_page_url("gemini-3.1-flash-image"): _md("models/gemini-3.1-flash-image.md"),
        model_page_url("gemini-3.1-pro-preview"): _md("models/gemini-3.1-pro-preview.md"),
        model_page_url("gemini-3.5-transcribe"): _md("models/gemini-3.5-transcribe.md"),
        model_page_url("index-only"): _md("models/index-only.md"),
        model_page_url("sparse-test"): _md("models/sparse-test.md"),
        model_page_url("gemini-stable-dated"): _md("models/gemini-stable-dated.md"),
        model_page_url("gemini-alias-target"): _md("models/gemini-alias-target.md"),
        model_page_url("imagen"): _md("models/imagen.md"),
        model_page_url("gemini-2.0-flash"): _md("models/gemini-2.0-flash.md"),
    }
    pages.update(overrides)
    return pages


def test_parser_modules_do_not_import_http() -> None:
    for name in ("parse.py", "normalize.py", "observations.py", "sources.py"):
        path = GOOGLE_PROVIDER / name
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
        assert "samyak.model.providers.google.fetch" not in imported


def test_rejects_non_https() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="HTTPS"):
        transport.get("http://ai.google.dev/gemini-api/docs/models.md.txt")


def test_rejects_unapproved_host() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://evil.example/gemini-api/docs/models.md.txt")


def test_rejects_vertex_host() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://docs.cloud.google.com/vertex-ai/docs/models.md.txt")


def test_rejects_unapproved_path() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation path"):
        transport.get("https://ai.google.dev/gemini-api/pricing.md.txt")


def test_rejects_query_and_fragment() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="query"):
        transport.get("https://ai.google.dev/gemini-api/docs/models.md.txt?key=1")
    with pytest.raises(TransportSecurityError, match="fragment"):
        transport.get("https://ai.google.dev/gemini-api/docs/models.md.txt#top")


def test_rejects_credentials_in_url() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="credentials"):
        transport.get("https://user:pass@ai.google.dev/gemini-api/docs/models.md.txt")


def test_capture_required_sources() -> None:
    transport = _transport(_official_pages())
    index = capture_google_source(transport, models_index_descriptor())
    deprecations = capture_google_source(transport, deprecations_descriptor())
    gemini_3 = capture_google_source(transport, gemini_3_descriptor())
    assert index.source_id == "google-docs-models-index"
    assert deprecations.source_id == "google-docs-deprecations"
    assert gemini_3.source_id == "google-docs-gemini-3"
    assert index.retrieved_at == STAMP


def test_refresh_partial_when_model_page_missing() -> None:
    pages = _official_pages()
    del pages[model_page_url("index-only")]
    result = refresh_google_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is True
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:google" in codes
    assert "SOURCE_FETCH_FAILED:google-docs-model-page:index-only" in codes


def test_refresh_complete_when_all_pages_present() -> None:
    result = refresh_google_catalog(
        _transport(_official_pages()),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is False
    assert result.catalog is not None
    assert result.notices == ()
    assert any(item.provider_id == "google" for item in result.catalog.models)


def test_refresh_partial_when_model_page_malformed() -> None:
    result = refresh_google_catalog(
        _transport(_official_pages(**{model_page_url("index-only"): _md("models/missing-id.md")})),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is True
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:google" in codes
    assert "SOURCE_PARSE_FAILED:google-docs-model-page:index-only" in codes


def test_refresh_fails_closed_on_required_source() -> None:
    pages = _official_pages()
    pages[MODELS_INDEX_URL] = FixtureHop(status=500, body=b"error")
    result = refresh_google_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is False
    assert result.catalog is None
    assert "500" in (result.error or "")


def test_malformed_required_source_fails_refresh() -> None:
    pages = _official_pages()
    pages[MODELS_INDEX_URL] = _md("malformed.md")
    result = refresh_google_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is False
    assert result.catalog is None


def test_malformed_gemini_3_source_fails_refresh() -> None:
    pages = _official_pages()
    pages[GEMINI_3_URL] = _md("gemini-3-no-context-table.md")
    result = refresh_google_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is False
    assert result.catalog is None
    assert "context-window table" in (result.error or "")


def test_official_sources_are_markdown_txt_not_html_md() -> None:
    assert MODELS_INDEX_URL == "https://ai.google.dev/gemini-api/docs/models.md.txt"
    assert DEPRECATIONS_URL == "https://ai.google.dev/gemini-api/docs/deprecations.md.txt"
    assert GEMINI_3_URL == "https://ai.google.dev/gemini-api/docs/gemini-3.md.txt"
    assert model_page_url("gemini-3.8-flash").endswith(".md.txt")
    for descriptor in required_source_descriptors():
        path = expected_path_for_source(descriptor)
        assert path.endswith(".md.txt")
        assert not path.endswith(".md")
