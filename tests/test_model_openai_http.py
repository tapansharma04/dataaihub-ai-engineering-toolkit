"""HTTP capture tests for OpenAI Model Intelligence. No live network."""

from __future__ import annotations

import ast
import hashlib
import socket
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from samyak.model.catalog import CatalogOverlay
from samyak.model.facts import FactStatus
from samyak.model.http import (
    FixtureHop,
    FixtureTransport,
    TransportError,
    TransportPolicy,
    TransportSecurityError,
    TransportTimeoutError,
)
from samyak.model.providers.openai.errors import OpenAIFetchError
from samyak.model.providers.openai.fetch import (
    capture_openai_source,
    openai_docs_policy,
    openai_https_transport,
    refresh_openai_catalog,
)
from samyak.model.providers.openai.sources import (
    DEPRECATIONS_URL,
    MODELS_INDEX_URL,
    content_hash_for_bytes,
    deprecations_descriptor,
    model_page_descriptor,
    model_page_source_id,
    model_page_url,
    models_index_descriptor,
)

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "openai"
OPENAI_PROVIDER = Path(__file__).resolve().parents[1] / "src/samyak/model/providers/openai"
RETRIEVED_AT = "2026-09-10T12:00:00+00:00"
GENERATED_AT = "2026-09-10T12:01:00+00:00"
VERIFIED_AT = "2026-09-10T12:01:00+00:00"
INDEX_443 = "https://developers.openai.com:443/api/docs/models.md"
INDEX_ONLY_BODY = b"# Index Only\n\nModel ID: `index-only`\n"


@pytest.fixture(autouse=True)
def deny_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Model Intelligence HTTP tests must not use the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("urllib.request.urlopen", blocked)


def _policy(**overrides: object) -> TransportPolicy:
    policy = openai_docs_policy()
    values = {
        "allowed_hosts": policy.allowed_hosts,
        "allowed_path_prefixes": policy.allowed_path_prefixes,
        "allowed_content_types": policy.allowed_content_types,
        "allowed_schemes": policy.allowed_schemes,
        "max_bytes": policy.max_bytes,
        "max_redirects": policy.max_redirects,
        "timeout_seconds": policy.timeout_seconds,
        "allowed_ports": policy.allowed_ports,
    }
    values.update(overrides)
    return TransportPolicy(**values)  # type: ignore[arg-type]


def _md(relative: str, **kwargs: object) -> FixtureHop:
    return FixtureHop(body=(FIXTURES / relative).read_bytes(), **kwargs)  # type: ignore[arg-type]


def _transport(
    pages: dict[str, FixtureHop | tuple[FixtureHop, ...]],
    *,
    policy: TransportPolicy | None = None,
) -> FixtureTransport:
    return FixtureTransport(
        pages,
        policy=policy or openai_docs_policy(),
        clock=lambda: RETRIEVED_AT,
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


def test_parser_modules_do_not_import_http() -> None:
    for name in ("parse.py", "normalize.py", "observations.py", "sources.py"):
        path = OPENAI_PROVIDER / name
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
        assert "samyak.model.providers.openai.fetch" not in imported


def test_rejects_non_https() -> None:
    transport = _transport({MODELS_INDEX_URL: _md("models.md")})
    with pytest.raises(TransportSecurityError, match="HTTPS"):
        transport.get("http://developers.openai.com/api/docs/models.md")


def test_rejects_unapproved_host() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get("https://evil.example/api/docs/models.md")


def test_rejects_credentials_in_url() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="credentials"):
        transport.get("https://user:token@developers.openai.com/api/docs/models.md")


def test_rejects_path_escape() -> None:
    transport = _transport({})
    with pytest.raises(TransportSecurityError, match="path"):
        transport.get("https://developers.openai.com/api/docs/../secrets.md")


def test_redirect_to_unapproved_host_is_rejected() -> None:
    transport = _transport(
        {
            MODELS_INDEX_URL: FixtureHop(
                location="https://evil.example/api/docs/models.md",
            )
        }
    )
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get(MODELS_INDEX_URL)


def test_redirect_to_http_is_rejected() -> None:
    transport = _transport(
        {
            MODELS_INDEX_URL: FixtureHop(
                location="http://developers.openai.com/api/docs/models.md",
            )
        }
    )
    with pytest.raises(TransportSecurityError, match="HTTPS"):
        transport.get(MODELS_INDEX_URL)


def test_allowed_https_same_host_redirect() -> None:
    body = (FIXTURES / "models.md").read_bytes()
    transport = _transport(
        {
            MODELS_INDEX_URL: FixtureHop(location=INDEX_443),
            INDEX_443: FixtureHop(body=body),
        }
    )
    response = transport.get(MODELS_INDEX_URL)
    assert response.status == 200
    assert response.requested_url == MODELS_INDEX_URL
    assert response.final_url == INDEX_443
    assert response.body == body


def test_excessive_redirects_are_rejected() -> None:
    urls = [model_page_url(f"redir-{index}") for index in range(7)]
    pages: dict[str, FixtureHop] = {}
    for index in range(6):
        pages[urls[index]] = FixtureHop(location=urls[index + 1])
    pages[urls[6]] = FixtureHop(body=b"# Models\n")
    transport = _transport(pages)
    with pytest.raises(TransportSecurityError, match="too many redirects"):
        transport.get(urls[0])


def test_timeout_is_explicit() -> None:
    transport = _transport({MODELS_INDEX_URL: FixtureHop(timeout=True)})
    with pytest.raises(TransportTimeoutError, match="timed out"):
        transport.get(MODELS_INDEX_URL)


def test_response_size_limit() -> None:
    policy = _policy(max_bytes=16)
    transport = _transport(
        {MODELS_INDEX_URL: FixtureHop(body=b"x" * 17)},
        policy=policy,
    )
    with pytest.raises(TransportSecurityError, match="maximum permitted size"):
        transport.get(MODELS_INDEX_URL)


def test_unexpected_content_type_is_rejected() -> None:
    transport = _transport(
        {
            MODELS_INDEX_URL: FixtureHop(
                body=b"{}",
                headers=(("Content-Type", "application/json"),),
            )
        }
    )
    with pytest.raises(TransportSecurityError, match="content type"):
        transport.get(MODELS_INDEX_URL)


def test_gzip_content_encoding_is_rejected() -> None:
    transport = _transport(
        {
            MODELS_INDEX_URL: FixtureHop(
                body=b"# Models\n",
                headers=(
                    ("Content-Type", "text/markdown"),
                    ("Content-Encoding", "gzip"),
                ),
            )
        }
    )
    with pytest.raises(TransportSecurityError, match="content encoding"):
        transport.get(MODELS_INDEX_URL)


def test_non_2xx_is_returned_for_capture_to_reject() -> None:
    transport = _transport({MODELS_INDEX_URL: FixtureHop(status=503, body=b"unavailable")})
    response = transport.get(MODELS_INDEX_URL)
    assert response.status == 503
    with pytest.raises(OpenAIFetchError, match="HTTP 503"):
        capture_openai_source(transport, models_index_descriptor())


def test_missing_source_fails() -> None:
    transport = _transport({})
    with pytest.raises(TransportError, match="could not be retrieved"):
        transport.get(MODELS_INDEX_URL)


def test_final_url_host_mismatch_is_rejected() -> None:
    transport = _transport(
        {
            MODELS_INDEX_URL: FixtureHop(
                body=b"# Models\n",
                final_url="https://evil.example/api/docs/models.md",
            )
        }
    )
    with pytest.raises(TransportSecurityError, match="approved documentation host"):
        transport.get(MODELS_INDEX_URL)


def test_capture_source_id_url_hash_and_timestamp() -> None:
    body = (FIXTURES / "models.md").read_text(encoding="utf-8")
    transport = _transport({MODELS_INDEX_URL: _md("models.md")})
    captured = capture_openai_source(transport, models_index_descriptor())
    assert captured.source_id == "openai-docs-models-index"
    assert captured.source_url == MODELS_INDEX_URL
    assert captured.retrieved_at == RETRIEVED_AT
    assert captured.content_hash == "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert captured.content_hash == content_hash_for_bytes(captured.body_bytes)


def test_model_page_source_identity() -> None:
    url = model_page_url("gpt-5.6-sol")
    transport = _transport({url: _md("models/gpt-5.6-sol.md")})
    captured = capture_openai_source(transport, model_page_descriptor("gpt-5.6-sol"))
    assert captured.source_id == model_page_source_id("gpt-5.6-sol")
    assert captured.source_url == url
    assert captured.source_id != model_page_source_id("gpt-5.6")


def test_multiple_model_pages_keep_separate_identities() -> None:
    transport = _transport(
        {
            model_page_url("gpt-5.6-sol"): _md("models/gpt-5.6-sol.md"),
            model_page_url("gpt-5.6"): _md("models/gpt-5.6.md"),
        }
    )
    first = capture_openai_source(transport, model_page_descriptor("gpt-5.6-sol"))
    second = capture_openai_source(transport, model_page_descriptor("gpt-5.6"))
    assert first.source_id == "openai-docs-model-page:gpt-5.6-sol"
    assert second.source_id == "openai-docs-model-page:gpt-5.6"
    assert first.source_url != second.source_url
    assert first.content_hash != second.content_hash


def test_model_page_identity_mismatch_after_redirect() -> None:
    requested = model_page_url("gpt-5.6-sol")
    other = model_page_url("gpt-5.6")
    transport = _transport(
        {
            requested: FixtureHop(location=other),
            other: _md("models/gpt-5.6.md"),
        }
    )
    with pytest.raises(OpenAIFetchError, match="different official document"):
        capture_openai_source(transport, model_page_descriptor("gpt-5.6-sol"))


def test_fixture_transport_is_deterministic() -> None:
    transport = _transport({MODELS_INDEX_URL: _md("models.md")})
    first = transport.get(MODELS_INDEX_URL)
    second = transport.get(MODELS_INDEX_URL)
    assert first.body == second.body
    assert first.status == second.status
    assert first.retrieved_at == RETRIEVED_AT
    assert first.final_url == MODELS_INDEX_URL


def test_required_index_failure_produces_no_catalog() -> None:
    pages = _official_pages()
    pages[MODELS_INDEX_URL] = FixtureHop(status=500, body=b"error")
    result = refresh_openai_catalog(
        _transport(pages),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
    )
    assert result.ok is False
    assert result.catalog is None
    assert result.partial is False
    assert result.error is not None
    assert "500" in result.error


def test_required_deprecations_failure_produces_no_catalog() -> None:
    pages = _official_pages()
    pages[DEPRECATIONS_URL] = FixtureHop(status=404, body=b"missing")
    result = refresh_openai_catalog(
        _transport(pages),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
    )
    assert result.ok is False
    assert result.catalog is None
    assert result.partial is False
    assert result.error is not None
    assert "404" in result.error


def test_best_effort_detail_failure_keeps_catalog_and_notice() -> None:
    pages = _official_pages()
    del pages[model_page_url("index-only")]
    result = refresh_openai_catalog(
        _transport(pages),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
    )
    assert result.ok is True
    assert result.partial is True
    assert result.catalog is not None
    assert result.catalog.freshness.overlay is CatalogOverlay.USER_CACHE
    codes = {notice.code for notice in result.catalog.notices}
    assert "PARTIAL_MODEL_PAGES" in codes
    assert "SOURCE_FETCH_FAILED:openai-docs-model-page:index-only" in codes
    record = next(item for item in result.catalog.models if item.provider_model_id == "index-only")
    assert record.context_window.status is FactStatus.NOT_VERIFIED
    assert record.context_window.value is None
    assert any(item.provider_model_id == "gpt-5.6-sol" for item in result.catalog.models)


def test_successful_full_capture() -> None:
    result = refresh_openai_catalog(
        _transport(_official_pages()),
        generated_at=GENERATED_AT,
        verified_at=VERIFIED_AT,
    )
    assert result.ok is True
    assert result.error is None
    assert result.partial is False
    assert result.notices == ()
    assert result.catalog is not None
    assert result.catalog.notices == ()
    ids = {item.provider_model_id for item in result.catalog.models}
    assert {"gpt-5.6-sol", "gpt-5.6", "gpt-4.5-preview", "sparse-test", "index-only"} <= ids
    sol = next(item for item in result.catalog.models if item.provider_model_id == "gpt-5.6-sol")
    assert sol.display_name.status is FactStatus.KNOWN
    retrieved = {item.source_id for item in result.catalog.freshness.source_retrieved_at}
    assert "openai-docs-models-index" in retrieved
    assert "openai-docs-deprecations" in retrieved
    assert "openai-docs-model-page:gpt-5.6-sol" in retrieved


def test_https_transport_uses_tls_only_and_no_cookie_handler() -> None:
    transport = openai_https_transport(clock=lambda: RETRIEVED_AT)
    names = {type(handler).__name__ for handler in transport._opener.handlers}
    assert names == {"HTTPSHandler"}


def test_https_transport_rejects_http_before_connect() -> None:
    transport = openai_https_transport(clock=lambda: RETRIEVED_AT)
    with pytest.raises(TransportSecurityError, match="HTTPS"):
        transport.get("http://developers.openai.com/api/docs/models.md")


def test_https_transport_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = openai_https_transport(clock=lambda: RETRIEVED_AT)

    def boom(*args: object, **kwargs: object) -> None:
        raise TimeoutError("slow")

    monkeypatch.setattr(transport._opener, "open", boom)
    with pytest.raises(TransportTimeoutError, match="timed out"):
        transport.get(MODELS_INDEX_URL)


def test_https_transport_content_length_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = openai_https_transport(clock=lambda: RETRIEVED_AT)

    class FakeResponse:
        status = 200
        headers = {
            "Content-Type": "text/markdown",
            "Content-Length": str(10**9),
        }

        def geturl(self) -> str:
            return MODELS_INDEX_URL

        def getcode(self) -> int:
            return 200

        def read(self, size: int = -1) -> bytes:
            raise AssertionError("must not read a body larger than the declared limit")

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(transport._opener, "open", lambda *args, **kwargs: FakeResponse())
    with pytest.raises(TransportSecurityError, match="maximum permitted size"):
        transport.get(MODELS_INDEX_URL)


def test_https_transport_sends_no_credentials_or_cookies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = openai_https_transport(clock=lambda: RETRIEVED_AT)
    seen: dict[str, object] = {}

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "text/markdown"}

        def __init__(self) -> None:
            self._body = b"# Models\n\n- [X](/api/docs/models/x.md)\n"

        def geturl(self) -> str:
            return MODELS_INDEX_URL

        def getcode(self) -> int:
            return 200

        def read(self, size: int = -1) -> bytes:
            if size < 0:
                data = self._body
                self._body = b""
                return data
            data = self._body[:size]
            self._body = self._body[size:]
            return data

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def open_request(request: Request, timeout: float) -> FakeResponse:
        seen["timeout"] = timeout
        seen["headers"] = {key.lower(): value for key, value in request.header_items()}
        seen["has_cookie"] = request.has_header("Cookie")
        seen["has_auth"] = request.has_header("Authorization")
        return FakeResponse()

    monkeypatch.setattr(transport._opener, "open", open_request)
    response = transport.get(MODELS_INDEX_URL)
    assert response.status == 200
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert "cookie" not in headers
    assert "authorization" not in headers
    assert seen["has_cookie"] is False
    assert seen["has_auth"] is False
    assert headers["accept-encoding"] == "identity"
    assert str(headers["user-agent"]).startswith("Samyak/")
    assert seen["timeout"] == openai_docs_policy().timeout_seconds


def test_https_transport_maps_http_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = openai_https_transport(clock=lambda: RETRIEVED_AT)
    headers = Message()
    headers["Content-Type"] = "text/plain"

    def open_request(*args: object, **kwargs: object) -> None:
        raise HTTPError(
            MODELS_INDEX_URL,
            404,
            "Not Found",
            headers,
            BytesIO(b"missing"),
        )

    monkeypatch.setattr(transport._opener, "open", open_request)
    response = transport.get(MODELS_INDEX_URL)
    assert response.status == 404
    with pytest.raises(OpenAIFetchError, match="HTTP 404"):
        capture_openai_source(transport, models_index_descriptor())


def test_capture_non_utf8_is_rejected() -> None:
    transport = _transport({MODELS_INDEX_URL: FixtureHop(body=b"\xff\xfe not utf-8")})
    with pytest.raises(OpenAIFetchError, match="UTF-8"):
        capture_openai_source(transport, models_index_descriptor())


def test_deprecations_descriptor_is_required() -> None:
    assert models_index_descriptor().required is True
    assert deprecations_descriptor().required is True
    assert model_page_descriptor("gpt-5.6-sol").required is False
