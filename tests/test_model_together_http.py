"""HTTP capture tests for Together AI Model Intelligence. No live network."""

from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest

from samyak.model.http import FixtureHop, FixtureTransport, TransportSecurityError
from samyak.model.providers.together.fetch import (
    capture_together_source,
    refresh_together_catalog,
    together_docs_policy,
)
from samyak.model.providers.together.sources import (
    CHANGELOG_URL,
    DEPRECATIONS_URL,
    SERVERLESS_URL,
    changelog_descriptor,
    expected_path_for_source,
    model_page_url,
    optional_source_descriptors,
    required_source_descriptors,
    serverless_descriptor,
)

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "together"
TOGETHER_PROVIDER = Path(__file__).resolve().parents[1] / "src/samyak/model/providers/together"
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
        policy=together_docs_policy(),
        clock=clock if callable(clock) else (lambda: STAMP),
    )


def _official_pages(**overrides: FixtureHop) -> dict[str, FixtureHop]:
    pages = {
        SERVERLESS_URL: _md("serverless.md"),
        DEPRECATIONS_URL: _md("deprecations.md"),
        CHANGELOG_URL: _md("changelog.md"),
    }
    for descriptor in optional_source_descriptors():
        if descriptor.url == CHANGELOG_URL:
            continue
        slug = descriptor.url.rsplit("/", 1)[-1].removesuffix(".md")
        pages[descriptor.url] = _md(f"{slug}.md")
    pages.update(overrides)
    return pages


def test_parser_modules_do_not_import_http() -> None:
    for name in ("parse.py", "normalize.py", "observations.py", "sources.py"):
        path = TOGETHER_PROVIDER / name
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
        assert "samyak.model.providers.together.fetch" not in imported


def test_rejects_non_https() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="HTTPS"):
        transport.get("http://docs.together.ai/docs/serverless/models.md")


def test_rejects_unapproved_host() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://evil.example/docs/serverless/models.md")
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://www.together.ai/models")
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://www.together.ai/models/glm-52")


def test_rejects_api_and_authenticated_hosts() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://api.together.ai/models")
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://api.together.xyz/v1/models")


def test_rejects_unapproved_path() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation path"):
        transport.get("https://docs.together.ai/api-reference/models")
    with pytest.raises(TransportSecurityError, match="approved documentation path"):
        transport.get("https://docs.together.ai/pricing.md")


def test_rejects_query_and_fragment() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="query"):
        transport.get("https://docs.together.ai/docs/serverless/models.md?key=1")
    with pytest.raises(TransportSecurityError, match="fragment"):
        transport.get("https://docs.together.ai/docs/serverless/models.md#top")


def test_rejects_credentials_in_url() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="credentials"):
        transport.get("https://user:pass@docs.together.ai/docs/serverless/models.md")


def test_capture_required_sources() -> None:
    transport = _transport(_official_pages())
    serverless = capture_together_source(transport, serverless_descriptor())
    changelog = capture_together_source(transport, changelog_descriptor())
    assert serverless.source_id == "together-docs-serverless-models"
    assert changelog.source_id == "together-docs-changelog"
    assert serverless.retrieved_at == STAMP


def test_refresh_serverless_catalogue_without_marking_partial() -> None:
    result = refresh_together_catalog(
        _transport(_official_pages()),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is False
    assert result.catalog is not None
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:together" not in codes
    assert any(item.provider_id == "together" for item in result.catalog.models)
    ids = {item.samyak_id for item in result.catalog.models}
    assert "together:mixedbread-ai/mxbai-rerank-large-v2" not in ids
    assert "together:togethercomputer/dedicated-changelog-only" not in ids
    assert any(item.samyak_id == "together:zai-org/GLM-5.2" for item in result.catalog.models)
    glm = next(
        item for item in result.catalog.models if item.samyak_id == "together:zai-org/GLM-5.2"
    )
    assert glm.max_output_tokens.value == 128_000


def test_refresh_partial_when_optional_model_page_missing() -> None:
    pages = _official_pages()
    del pages[model_page_url("glm-5.2-quickstart")]
    result = refresh_together_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is True
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:together" in codes
    assert "SOURCE_FETCH_FAILED:together-docs-model-page:glm-5.2-quickstart" in codes


def test_refresh_partial_when_optional_changelog_missing() -> None:
    pages = _official_pages()
    del pages[CHANGELOG_URL]
    result = refresh_together_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is True
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:together" in codes
    assert "SOURCE_FETCH_FAILED:together-docs-changelog" in codes


def test_refresh_partial_when_optional_changelog_malformed() -> None:
    result = refresh_together_catalog(
        _transport(_official_pages(**{CHANGELOG_URL: _md("malformed.md")})),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is True
    codes = {notice.code for notice in result.notices}
    assert "PARTIAL_MODEL_PAGES:together" in codes
    assert "SOURCE_PARSE_FAILED:together-docs-changelog" in codes


def test_empty_optional_changelog_is_not_partial() -> None:
    result = refresh_together_catalog(
        _transport(_official_pages(**{CHANGELOG_URL: _md("changelog-empty.md")})),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is True
    assert result.partial is False
    ids = {item.samyak_id for item in result.catalog.models}
    assert "together:togethercomputer/dedicated-changelog-only" not in ids


def test_refresh_fails_closed_on_required_source() -> None:
    pages = _official_pages()
    pages[SERVERLESS_URL] = FixtureHop(status=500, body=b"error")
    result = refresh_together_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is False
    assert result.catalog is None
    assert "500" in (result.error or "")


def test_malformed_required_source_fails_refresh() -> None:
    pages = _official_pages()
    pages[SERVERLESS_URL] = _md("malformed.md")
    result = refresh_together_catalog(
        _transport(pages),
        generated_at=STAMP,
        verified_at=STAMP,
    )
    assert result.ok is False
    assert result.catalog is None


def test_official_sources_are_markdown_on_docs_host() -> None:
    assert SERVERLESS_URL == "https://docs.together.ai/docs/serverless/models.md"
    assert DEPRECATIONS_URL == "https://docs.together.ai/docs/deprecations.md"
    assert CHANGELOG_URL == "https://docs.together.ai/docs/changelog.md"
    assert model_page_url("glm-5.2-quickstart") == (
        "https://docs.together.ai/docs/glm-5.2-quickstart.md"
    )
    for descriptor in required_source_descriptors():
        path = expected_path_for_source(descriptor)
        assert path.endswith(".md")
        assert not path.endswith(".md.txt")
        assert path.startswith("/docs")
    for descriptor in optional_source_descriptors():
        path = expected_path_for_source(descriptor)
        assert path.endswith(".md")
        assert path.startswith("/docs")
