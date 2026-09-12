"""HTTP capture tests for Anthropic Model Intelligence. No live network."""

from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest

from samyak.model.http import FixtureHop, FixtureTransport, TransportSecurityError
from samyak.model.providers.anthropic.errors import AnthropicFetchError
from samyak.model.providers.anthropic.fetch import (
    anthropic_docs_policy,
    capture_anthropic_source,
    refresh_anthropic_catalog,
)
from samyak.model.providers.anthropic.sources import (
    DEPRECATIONS_URL,
    MODELS_INDEX_URL,
    deprecations_descriptor,
    model_page_url,
    models_index_descriptor,
)

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "anthropic"
ANTHROPIC_PROVIDER = Path(__file__).resolve().parents[1] / "src/samyak/model/providers/anthropic"
STAMP = "2026-09-11T12:00:00+00:00"


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
        policy=anthropic_docs_policy(),
        clock=clock if callable(clock) else (lambda: STAMP),
    )


def _official_pages(**overrides: FixtureHop) -> dict[str, FixtureHop]:
    pages = {
        MODELS_INDEX_URL: _md("overview.md"),
        DEPRECATIONS_URL: _md("deprecations.md"),
        model_page_url("opus-5"): _md("models/opus-5.md"),
        model_page_url("haiku-4-5"): _md("models/haiku-4-5.md"),
        model_page_url("sonnet-5"): _md("models/sonnet-5.md"),
        model_page_url("fable-5-1"): _md("models/fable-5-1.md"),
        model_page_url("index-only"): _md("models/index-only.md"),
        model_page_url("sparse-test"): _md("models/sparse-test.md"),
        model_page_url("no-api-access"): _md("models/no-api-access.md"),
    }
    pages.update(overrides)
    return pages


def test_parser_modules_do_not_import_http() -> None:
    for name in ("parse.py", "normalize.py", "observations.py", "sources.py"):
        path = ANTHROPIC_PROVIDER / name
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
        assert "samyak.model.providers.anthropic.fetch" not in imported


def test_rejects_non_https() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="HTTPS"):
        transport.get("http://platform.claude.com/docs/en/models/overview.md")


def test_rejects_unapproved_host() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://evil.example/docs/en/models/overview.md")


def test_rejects_unapproved_path() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation path"):
        transport.get("https://platform.claude.com/docs/en/pricing.md")


def test_rejects_query_and_fragment() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="query"):
        transport.get("https://platform.claude.com/docs/en/models/overview.md?token=1")
    with pytest.raises(TransportSecurityError, match="fragment"):
        transport.get("https://platform.claude.com/docs/en/models/overview.md#top")


def test_rejects_credentials_in_url() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="credentials"):
        transport.get("https://user:pass@platform.claude.com/docs/en/models/overview.md")


def test_capture_required_sources() -> None:
    transport = _transport(_official_pages())
    index = capture_anthropic_source(transport, models_index_descriptor())
    deprecations = capture_anthropic_source(transport, deprecations_descriptor())
    assert index.source_id == "anthropic-docs-models-index"
    assert deprecations.source_id == "anthropic-docs-deprecations"
    assert index.retrieved_at == STAMP


def test_refresh_partial_when_model_page_missing() -> None:
    pages = _official_pages()
    del pages[model_page_url("index-only")]
    result = refresh_anthropic_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is True
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:anthropic" in codes
    assert "SOURCE_FETCH_FAILED:anthropic-docs-model-page:index-only" in codes


def test_refresh_fails_closed_on_required_source() -> None:
    pages = _official_pages()
    pages[MODELS_INDEX_URL] = FixtureHop(status=500, body=b"error")
    result = refresh_anthropic_catalog(
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
    result = refresh_anthropic_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is False
    assert result.catalog is None


def test_capture_rejects_http_error_status() -> None:
    transport = _transport({MODELS_INDEX_URL: FixtureHop(status=404, body=b"missing")})
    with pytest.raises(AnthropicFetchError, match="404"):
        capture_anthropic_source(transport, models_index_descriptor())
