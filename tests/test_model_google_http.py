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
    MODELS_INDEX_URL,
    deprecations_descriptor,
    expected_path_for_source,
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
        MODELS_INDEX_URL: _md("aistudio-models-subset.md"),
        DEPRECATIONS_URL: _md("aistudio-deprecations.md"),
        model_page_url("gemini-3.1-flash-lite"): _md("models/aistudio-gemini-3.1-flash-lite.md"),
        model_page_url("gemini-3.1-pro-preview"): _md("models/aistudio-gemini-3.1-pro-preview.md"),
        model_page_url("gemini-3-flash-preview"): _md("models/aistudio-gemini-3-flash-preview.md"),
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
        transport.get("http://aistudio.google.com/docs/models.md")


def test_rejects_unapproved_host() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://evil.example/docs/models.md")


def test_rejects_vertex_host() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://docs.cloud.google.com/vertex-ai/docs/models.md")


def test_rejects_unapproved_path() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation path"):
        transport.get("https://aistudio.google.com/docs/pricing.md")
    with pytest.raises(TransportSecurityError, match="approved documentation path"):
        transport.get("https://aistudio.google.com/docs/models/not-a-markdown-page.txt")


def test_rejects_query_and_fragment() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="query"):
        transport.get("https://aistudio.google.com/docs/models.md?key=1")
    with pytest.raises(TransportSecurityError, match="fragment"):
        transport.get("https://aistudio.google.com/docs/models.md#top")


def test_rejects_credentials_in_url() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="credentials"):
        transport.get("https://user:pass@aistudio.google.com/docs/models.md")


def test_capture_required_sources() -> None:
    transport = _transport(_official_pages())
    index = capture_google_source(transport, models_index_descriptor())
    deprecations = capture_google_source(transport, deprecations_descriptor())
    assert index.source_id == "google-docs-models-index"
    assert deprecations.source_id == "google-docs-deprecations"
    assert index.retrieved_at == STAMP


def test_refresh_partial_when_model_page_missing() -> None:
    pages = _official_pages()
    del pages[model_page_url("gemini-3.1-flash-lite")]
    result = refresh_google_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is True
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:google" in codes
    assert "SOURCE_FETCH_FAILED:google-docs-model-page:gemini-3.1-flash-lite" in codes


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
        _transport(
            _official_pages(
                **{model_page_url("gemini-3.1-flash-lite"): _md("models/missing-id.md")}
            )
        ),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is True
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:google" in codes
    assert "SOURCE_PARSE_FAILED:google-docs-model-page:gemini-3.1-flash-lite" in codes


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


def test_malformed_required_directive_table_fails_refresh() -> None:
    pages = _official_pages(
        **{
            MODELS_INDEX_URL: FixtureHop(
                body=(
                    b"# Models\n{% table %}\n* Model\n* Endpoint\n---\n"
                    b"* [Gemini](/docs/models/gemini)\n{% /table %}\n"
                )
            )
        }
    )
    result = refresh_google_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is False
    assert result.catalog is None
    assert "directive table" in (result.error or "")


def test_official_sources_are_canonical_ai_studio_markdown() -> None:
    assert MODELS_INDEX_URL == "https://aistudio.google.com/docs/models.md"
    assert DEPRECATIONS_URL == "https://aistudio.google.com/docs/deprecations.md"
    assert model_page_url("gemini-3.1-flash-lite") == (
        "https://aistudio.google.com/docs/models/gemini-3.1-flash-lite.md"
    )
    for descriptor in required_source_descriptors():
        path = expected_path_for_source(descriptor)
        assert path.endswith(".md")
