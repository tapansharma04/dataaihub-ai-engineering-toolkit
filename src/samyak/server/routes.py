"""Request routing for the local run-history viewer."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, unquote, urlparse

from samyak.comparison.compare import compare_runs, earlier_run
from samyak.comparison.errors import IncompatibleRunsError
from samyak.corpus.report import render_html_report
from samyak.server.compare_page import render_comparison_page
from samyak.server.dashboard import (
    HISTORY_PAGE_SIZE,
    render_bad_request_page,
    render_corrupt_run_page,
    render_dashboard,
    render_not_found_page,
    wrap_report_html,
)
from samyak.store.errors import RunCorruptError, RunNotFoundError, RunSchemaError
from samyak.store.filesystem import RunStore


class ViewerHandler(BaseHTTPRequestHandler):
    """HTTP handler bound to a :class:`RunStore` instance."""

    store: RunStore

    @classmethod
    def with_store(cls, store: RunStore) -> type[ViewerHandler]:
        return type("BoundViewerHandler", (cls,), {"store": store})

    def do_GET(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        # Same routing as GET, no body. Useful for probes; not a public API.
        self._handle(body=False)

    def _handle(self, *, body: bool = True) -> None:
        path = _normalize_request_path(self.path)
        if path is None:
            self._send_html(404, render_not_found_page("Page not found."), body=body)
            return

        parsed = urlparse(self.path)
        if path == "/":
            query = parse_qs(parsed.query)
            offset = _parse_offset(query.get("offset", ["0"])[0])
            page = self.store.list_runs(limit=HISTORY_PAGE_SIZE, offset=offset)
            html = render_dashboard(page)
            self._send_html(200, html, body=body)
            return

        if path.startswith("/runs/"):
            run_id = path.removeprefix("/runs/")
            if not run_id:
                self._send_html(404, render_not_found_page("That run was not found."), body=body)
                return
            self._send_run(run_id, body=body)
            return

        if path == "/compare":
            self._send_compare(parsed, body=body)
            return

        self._send_html(404, render_not_found_page("Page not found."), body=body)

    def _send_run(self, run_id: str, *, body: bool) -> None:
        try:
            stored = self.store.get_run(run_id)
        except RunNotFoundError:
            self._send_html(404, render_not_found_page("That run was not found."), body=body)
            return
        except RunSchemaError:
            self._send_html(
                409,
                render_corrupt_run_page(run_id, "This run uses an unsupported report schema."),
                body=body,
            )
            return
        except RunCorruptError:
            self._send_html(
                409,
                render_corrupt_run_page(run_id, "This saved run could not be read."),
                body=body,
            )
            return

        report_html = render_html_report(stored.report)
        html = wrap_report_html(report_html, stored.metadata)
        self._send_html(200, html, body=body)

    def _send_compare(self, parsed, *, body: bool) -> None:
        parsed_ids = _compare_run_ids(parse_qs(parsed.query))
        if parsed_ids is None:
            self._send_html(
                400,
                render_bad_request_page("Select exactly two runs to compare."),
                body=body,
            )
            return
        left_id, right_id, explicit_baseline = parsed_ids
        if left_id == right_id:
            self._send_html(
                400,
                render_bad_request_page("Select two different runs to compare."),
                body=body,
            )
            return

        loaded: list = []
        for run_id in (left_id, right_id):
            try:
                loaded.append(self.store.get_run(run_id))
            except RunNotFoundError:
                self._send_html(404, render_not_found_page("That run was not found."), body=body)
                return
            except RunSchemaError:
                self._send_html(
                    409,
                    render_corrupt_run_page(run_id, "This run uses an unsupported report schema."),
                    body=body,
                )
                return
            except RunCorruptError:
                self._send_html(
                    409,
                    render_corrupt_run_page(run_id, "This saved run could not be read."),
                    body=body,
                )
                return

        baseline, current = loaded[0], loaded[1]
        if not explicit_baseline:
            baseline, current = earlier_run(loaded[0], loaded[1])
        try:
            result = compare_runs(baseline, current)
        except IncompatibleRunsError as exc:
            self._send_html(409, render_bad_request_page(exc.user_message), body=body)
            return

        self._send_html(200, render_comparison_page(result), body=body)

    def _send_html(self, status: int, html: str, *, body: bool) -> None:
        payload = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if body:
            self.wfile.write(payload)


def _normalize_request_path(raw: str) -> str | None:
    """Return a safe path or None if the request must not be served."""
    parsed = urlparse(raw)
    path = unquote(parsed.path)
    if "\x00" in path or "\\" in path:
        return None
    if not path.startswith("/"):
        return None
    parts = [part for part in path.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        return None
    if any("/" in part or "\\" in part for part in parts):
        return None
    normalized = "/" + "/".join(parts) if parts else "/"
    return normalized


def _compare_run_ids(query: dict[str, list[str]]) -> tuple[str, str, bool] | None:
    left = (query.get("left") or [""])[0].strip()
    right = (query.get("right") or [""])[0].strip()
    if left and right:
        return left, right, True
    runs = [item.strip() for item in query.get("run", []) if item.strip()]
    if len(runs) == 2:
        return runs[0], runs[1], False
    return None


def _parse_offset(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(value, 0)
