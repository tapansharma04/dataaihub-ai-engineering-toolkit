"""Hardened HTTP GET for Model Intelligence source capture.

Not a general HTTP client. Callers inject this transport so tests never need
the network. Provider parsers do not import this module.
"""

from __future__ import annotations

import posixpath
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import HTTPSHandler, OpenerDirector, Request

from samyak.model.errors import ModelCatalogError

Clock = Callable[[], str]


class TransportError(ModelCatalogError):
    """A source document could not be retrieved."""


class TransportSecurityError(TransportError):
    """The request or response violated transport security policy."""


class TransportTimeoutError(TransportError):
    """The request exceeded the configured timeout."""


@dataclass(frozen=True, slots=True)
class TransportPolicy:
    """Limits applied to every hop, including redirects."""

    allowed_hosts: frozenset[str]
    allowed_path_prefixes: frozenset[str]
    allowed_content_types: frozenset[str] = frozenset({"text/markdown", "text/plain"})
    allowed_schemes: frozenset[str] = frozenset({"https"})
    max_bytes: int = 2 * 1024 * 1024
    max_redirects: int = 5
    timeout_seconds: float = 20.0
    allowed_ports: frozenset[int] = frozenset({443})


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Minimal GET result needed to capture a documentation source."""

    requested_url: str
    final_url: str
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    retrieved_at: str

    def header(self, name: str) -> str | None:
        wanted = name.lower()
        for key, value in self.headers:
            if key.lower() == wanted:
                return value
        return None

    @property
    def content_type(self) -> str:
        raw = self.header("content-type") or ""
        return raw.split(";", 1)[0].strip().lower()


class HttpTransport(Protocol):
    """GET-only transport. Implementations must not accept extra headers."""

    def get(self, url: str) -> HttpResponse:
        """Fetch ``url``. Must not send cookies or credentials."""
        ...


class RawRequester(Protocol):
    """Single-hop GET used by the shared redirect/security loop."""

    def request(self, url: str) -> HttpResponse:
        """Return one response without following redirects."""
        ...


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_USER_AGENT_PRODUCT = "Samyak"


def user_agent(version: str) -> str:
    return f"{_USER_AGENT_PRODUCT}/{version}"


def validate_transport_url(url: str, policy: TransportPolicy) -> str:
    """Reject non-HTTPS, credentials-in-URL, and off-allowlist hosts/paths."""
    if not isinstance(url, str) or not url.strip():
        raise TransportSecurityError("request URL is missing")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in policy.allowed_schemes:
        raise TransportSecurityError("only HTTPS URLs are permitted")
    host = (parsed.hostname or "").lower()
    if host not in policy.allowed_hosts:
        raise TransportSecurityError("URL host is not an approved documentation host")
    if parsed.username is not None or parsed.password is not None:
        raise TransportSecurityError("URLs must not include credentials")
    if parsed.port is not None and parsed.port not in policy.allowed_ports:
        raise TransportSecurityError("URL port is not permitted")
    raw_path = parsed.path or ""
    if "\x00" in url or ".." in raw_path.split("/"):
        raise TransportSecurityError("URL path is not an approved documentation path")
    path = posixpath.normpath(raw_path or "/")
    if not path.startswith("/"):
        path = f"/{path}"
    if not _path_allowed(path, policy.allowed_path_prefixes):
        raise TransportSecurityError("URL path is not an approved documentation path")
    if parsed.params:
        raise TransportSecurityError("URL parameters are not permitted")
    if parsed.query:
        raise TransportSecurityError("URL query is not permitted")
    if parsed.fragment:
        raise TransportSecurityError("URL fragment is not permitted")
    return urlunparse(
        (
            parsed.scheme.lower(),
            host if parsed.port is None else f"{host}:{parsed.port}",
            path,
            "",
            parsed.query,
            "",
        )
    )


def _path_allowed(path: str, prefixes: frozenset[str]) -> bool:
    for prefix in prefixes:
        normalized_prefix = posixpath.normpath(prefix) if prefix else "/"
        if not normalized_prefix.startswith("/"):
            normalized_prefix = f"/{normalized_prefix}"
        if path == normalized_prefix or path.startswith(f"{normalized_prefix}/"):
            return True
    return False


def complete_get(
    requester: RawRequester,
    url: str,
    *,
    policy: TransportPolicy,
    clock: Clock,
) -> HttpResponse:
    """Follow redirects under ``policy`` and return the final response."""
    requested = validate_transport_url(url, policy)
    current = requested
    retrieved_at = clock()
    follows = 0
    while True:
        current = validate_transport_url(current, policy)
        try:
            response = requester.request(current)
        except TimeoutError as exc:
            raise TransportTimeoutError("documentation request timed out") from exc
        except TransportTimeoutError:
            raise
        except TransportError:
            raise
        except URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                raise TransportTimeoutError("documentation request timed out") from exc
            raise TransportError("documentation source could not be retrieved") from exc
        except OSError as exc:
            raise TransportError("documentation source could not be retrieved") from exc
        if len(response.body) > policy.max_bytes:
            raise TransportSecurityError("response exceeds the maximum permitted size")
        if response.status in _REDIRECT_STATUSES:
            if follows >= policy.max_redirects:
                raise TransportSecurityError("too many redirects")
            location = response.header("location")
            if not location or not location.strip():
                raise TransportSecurityError("redirect is missing a Location header")
            current = urljoin(response.final_url or current, location.strip())
            follows += 1
            continue
        final_url = validate_transport_url(response.final_url or current, policy)
        _validate_content_type(response, policy)
        _validate_content_encoding(response)
        return HttpResponse(
            requested_url=requested,
            final_url=final_url,
            status=response.status,
            headers=response.headers,
            body=response.body,
            retrieved_at=retrieved_at,
        )


def _validate_content_type(response: HttpResponse, policy: TransportPolicy) -> None:
    if response.status < 200 or response.status >= 300:
        return
    if response.content_type not in policy.allowed_content_types:
        raise TransportSecurityError("response content type is not permitted")


def _validate_content_encoding(response: HttpResponse) -> None:
    if response.status < 200 or response.status >= 300:
        return
    encoding = (response.header("content-encoding") or "").strip().lower()
    if encoding and encoding != "identity":
        raise TransportSecurityError("response content encoding is not permitted")


def _headers_tuple(
    items: Mapping[str, str] | tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if isinstance(items, tuple):
        return tuple((str(key), str(value)) for key, value in items)
    return tuple((str(key), str(value)) for key, value in items.items())


def _mapping_headers(raw_headers: object) -> tuple[tuple[str, str], ...]:
    if raw_headers is None:
        return ()
    items = getattr(raw_headers, "items", None)
    if not callable(items):
        return ()
    return tuple((str(key), str(value)) for key, value in items())


class HttpsTransport:
    """Production GET over TLS. No cookies, credentials, or caller headers."""

    def __init__(
        self,
        *,
        policy: TransportPolicy,
        clock: Clock,
        user_agent_value: str,
    ) -> None:
        self._policy = policy
        self._clock = clock
        self._user_agent = user_agent_value
        context = ssl.create_default_context()
        opener = OpenerDirector()
        opener.add_handler(HTTPSHandler(context=context))
        self._opener = opener

    def get(self, url: str) -> HttpResponse:
        return complete_get(self, url, policy=self._policy, clock=self._clock)

    def request(self, url: str) -> HttpResponse:
        request = Request(
            url,
            method="GET",
            headers={
                "User-Agent": self._user_agent,
                "Accept": "text/markdown, text/plain;q=0.9",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with self._opener.open(request, timeout=self._policy.timeout_seconds) as response:
                return self._response_from_urllib(url, response)
        except TimeoutError as exc:
            raise TransportTimeoutError("documentation request timed out") from exc
        except HTTPError as exc:
            try:
                body = _read_limited(exc, self._policy.max_bytes)
                headers = _mapping_headers(exc.headers)
                return HttpResponse(
                    requested_url=url,
                    final_url=str(getattr(exc, "url", None) or url),
                    status=int(exc.code),
                    headers=headers,
                    body=body,
                    retrieved_at=self._clock(),
                )
            finally:
                exc.close()

    def _response_from_urllib(self, url: str, response: object) -> HttpResponse:
        status = int(getattr(response, "status", 0) or response.getcode())  # type: ignore[union-attr]
        final_url = str(getattr(response, "geturl", lambda: url)() or url)
        headers = _mapping_headers(getattr(response, "headers", None))
        content_length = None
        for key, value in headers:
            if key.lower() == "content-length" and value.isdigit():
                content_length = int(value)
                break
        if content_length is not None and content_length > self._policy.max_bytes:
            raise TransportSecurityError("response exceeds the maximum permitted size")
        body = _read_limited(response, self._policy.max_bytes)
        return HttpResponse(
            requested_url=url,
            final_url=final_url,
            status=status,
            headers=headers,
            body=body,
            retrieved_at=self._clock(),
        )


def _read_limited(stream: object, max_bytes: int) -> bytes:
    read = getattr(stream, "read", None)
    if not callable(read):
        raise TransportError("documentation source could not be retrieved")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = read(65536)
        if not chunk:
            break
        if not isinstance(chunk, bytes | bytearray):
            raise TransportError("documentation source could not be retrieved")
        total += len(chunk)
        if total > max_bytes:
            raise TransportSecurityError("response exceeds the maximum permitted size")
        chunks.append(bytes(chunk))
    return b"".join(chunks)


@dataclass(frozen=True, slots=True)
class FixtureHop:
    """One hop returned by FixtureTransport. No network."""

    status: int = 200
    headers: tuple[tuple[str, str], ...] = (("Content-Type", "text/markdown"),)
    body: bytes = b""
    location: str | None = None
    final_url: str | None = None
    timeout: bool = False
    missing: bool = False


class FixtureTransport:
    """Deterministic URL → response map for tests. Never opens sockets."""

    def __init__(
        self,
        pages: Mapping[str, FixtureHop | tuple[FixtureHop, ...]],
        *,
        policy: TransportPolicy,
        clock: Clock,
    ) -> None:
        mapped: dict[str, tuple[FixtureHop, ...]] = {}
        for url, hops in pages.items():
            mapped[url] = hops if isinstance(hops, tuple) else (hops,)
        self._pages = mapped
        self._cursors = {url: 0 for url in self._pages}
        self._policy = policy
        self._clock = clock

    def get(self, url: str) -> HttpResponse:
        return complete_get(self, url, policy=self._policy, clock=self._clock)

    def request(self, url: str) -> HttpResponse:
        hops = self._pages.get(url)
        if hops is None:
            raise TransportError("documentation source could not be retrieved")
        index = self._cursors.get(url, 0)
        if index >= len(hops):
            hop = hops[-1]
        else:
            hop = hops[index]
            self._cursors[url] = index + 1
        if hop.timeout:
            raise TransportTimeoutError("documentation request timed out")
        if hop.missing:
            raise TransportError("documentation source could not be retrieved")
        headers = list(_headers_tuple(hop.headers))
        if hop.location is not None:
            headers.append(("Location", hop.location))
            status = hop.status if hop.status in _REDIRECT_STATUSES else 302
        else:
            status = hop.status
        body = hop.body
        if len(body) > self._policy.max_bytes:
            raise TransportSecurityError("response exceeds the maximum permitted size")
        return HttpResponse(
            requested_url=url,
            final_url=hop.final_url if hop.final_url is not None else url,
            status=status,
            headers=tuple(headers),
            body=body,
            retrieved_at=self._clock(),
        )
