"""Server-side HTML for the local run-history dashboard."""

from __future__ import annotations

from html import escape

from samyak.store.models import RunMetadata, RunPage

HISTORY_PAGE_SIZE = 50


def render_dashboard(page: RunPage) -> str:
    """Render the recent-runs index from already-validated listing metadata."""
    rows = _rows(page.runs)
    empty = ""
    if not page.runs:
        empty = (
            '<p class="empty">No saved runs yet. Analyze a corpus with '
            "<code>samyak corpus PATH --save</code>, then refresh this page. "
            "Historical reports can be opened here without rerunning analysis.</p>"
        )
    skipped = ""
    if page.skipped:
        skipped = (
            f'<p class="muted">{_e(page.skipped)} saved run(s) on this machine '
            "could not be read and were skipped.</p>"
        )
    showing = _showing_label(page)
    table = ""
    if rows:
        hint = '<p class="scroll-hint muted">On a narrow screen, scroll the table sideways.</p>'
        wrap = f'<div class="table-wrap">{rows}</div>'
        if len(page.runs) >= 2:
            table = (
                f"{hint}"
                '<form class="compare-form" method="get" action="/compare">'
                f"{wrap}"
                '<p class="compare-bar">'
                '<button type="submit">Compare selected runs</button>'
                '<span class="muted">Select exactly two runs. The earlier saved run '
                "is the baseline; deltas are baseline → current.</span>"
                "</p>"
                "</form>"
            )
        else:
            table = f"{hint}{wrap}"
    return _page(
        title="Samyak Run History",
        body="".join(
            [
                "<header>",
                '<p class="eyebrow">Samyak</p>',
                "<h1>Run History</h1>",
                '<p class="lede">Local Samyak workspace. Data stored on this machine.</p>',
                "</header>",
                "<section>",
                "<h2>Recent Runs</h2>",
                f'<p class="muted">{_e(showing)}</p>',
                skipped,
                empty,
                table,
                _pagination(page),
                "</section>",
                "<footer>",
                "<p>This viewer reads saved analysis snapshots from the local "
                "Samyak cache. Nothing is sent over the network. "
                "Select two saved runs to compare findings.</p>",
                "</footer>",
            ]
        ),
    )


def wrap_report_html(report_html: str, metadata: RunMetadata) -> str:
    """Prefix the existing HTML report with local-viewer navigation and run metadata."""
    corpus = metadata.corpus_label or "corpus"
    path_row = ""
    if metadata.corpus_path:
        path_row = (
            f"<div><dt>Corpus path</dt><dd><code>{_e(metadata.corpus_path)}</code></dd></div>"
        )
    chrome = (
        f"<style>{_NAV_CSS}</style>"
        '<nav class="sv-nav">'
        '<a href="/">← Run History</a>'
        '<span class="sv-local">Local workspace</span>'
        "</nav>"
        '<aside class="sv-run-meta">'
        f"<div><dt>Corpus</dt><dd>{_e(corpus)}</dd></div>"
        f"{path_row}"
        f"<div><dt>Run ID</dt><dd><code>{_e(metadata.run_id)}</code></dd></div>"
        f"<div><dt>Timestamp</dt><dd>{_e(_format_timestamp(metadata.created_at))}</dd></div>"
        f"<div><dt>Duration</dt><dd>{_e(_format_duration(metadata.duration_seconds))}</dd></div>"
        f"<div><dt>Status</dt><dd>{_e(metadata.status)}</dd></div>"
        "</aside>"
    )
    if "<body>" in report_html:
        return report_html.replace("<body>", f"<body>\n{chrome}\n", 1)
    return chrome + report_html


def render_not_found_page(message: str) -> str:
    return _page(
        title="Run not found — Samyak",
        body="".join(
            [
                "<header>",
                '<p class="eyebrow">Samyak</p>',
                "<h1>Not found</h1>",
                f"<p>{_e(message)}</p>",
                '<p><a href="/">← Run History</a></p>',
                "</header>",
            ]
        ),
    )


def render_bad_request_page(message: str) -> str:
    return _page(
        title="Cannot compare — Samyak",
        body="".join(
            [
                "<header>",
                '<p class="eyebrow">Samyak</p>',
                "<h1>Cannot compare</h1>",
                f"<p>{_e(message)}</p>",
                '<p><a href="/">← Run History</a></p>',
                "</header>",
            ]
        ),
    )


def render_corrupt_run_page(run_id: str, message: str) -> str:
    return _page(
        title="Saved run unavailable — Samyak",
        body="".join(
            [
                "<header>",
                '<p class="eyebrow">Samyak</p>',
                "<h1>Saved run unavailable</h1>",
                f"<p>{_e(message)}</p>",
                f'<p class="muted">Run ID: <code>{_e(run_id)}</code></p>',
                '<p><a href="/">← Run History</a></p>',
                "</header>",
            ]
        ),
    )


def _rows(runs: tuple[RunMetadata, ...]) -> str:
    if not runs:
        return ""
    selectable = len(runs) >= 2
    select_header = "<th>Compare</th>" if selectable else ""
    header = (
        "<thead><tr>"
        f"{select_header}"
        "<th>Timestamp</th>"
        "<th>Corpus</th>"
        "<th>Run ID</th>"
        "<th>Duration</th>"
        "<th>Files discovered</th>"
        "<th>Files analyzed</th>"
        "<th>Findings</th>"
        "<th>Load errors</th>"
        "<th>Discovery errors</th>"
        "<th>Status</th>"
        "</tr></thead>"
    )
    body_rows = []
    for run in runs:
        href = f"/runs/{_e(run.run_id)}"
        label = run.corpus_label or "corpus"
        select_cell = ""
        if selectable:
            select_cell = (
                "<td>"
                f'<label class="pick">'
                f'<input type="checkbox" name="run" value="{_e(run.run_id)}">'
                "<span>Select</span>"
                "</label>"
                "</td>"
            )
        body_rows.append(
            "<tr>"
            f"{select_cell}"
            f"<td>{_e(_format_timestamp(run.created_at))}</td>"
            f"<td>{_e(label)}</td>"
            f'<td><a href="{href}"><code>{_e(run.run_id)}</code></a></td>'
            f"<td>{_e(_format_duration(run.duration_seconds))}</td>"
            f"<td>{_e(run.files_discovered)}</td>"
            f"<td>{_e(run.files_analyzed)}</td>"
            f"<td>{_e(run.findings_count)}</td>"
            f"<td>{_e(run.load_errors)}</td>"
            f"<td>{_e(run.discovery_errors)}</td>"
            f"<td>{_e(run.status)}</td>"
            "</tr>"
        )
    return f'<table class="runs">{header}<tbody>{"".join(body_rows)}</tbody></table>'


def _pagination(page: RunPage) -> str:
    if page.total <= page.limit and page.offset == 0:
        return ""
    links: list[str] = []
    if page.offset > 0:
        prev_offset = max(page.offset - page.limit, 0)
        links.append(f'<a href="/?offset={prev_offset}">Newer runs</a>')
    if page.offset + page.limit < page.total:
        next_offset = page.offset + page.limit
        links.append(f'<a href="/?offset={next_offset}">Older runs</a>')
    if not links:
        return ""
    return f'<nav class="pager">{"".join(links)}</nav>'


def _showing_label(page: RunPage) -> str:
    if page.total == 0:
        return "0 saved runs"
    start = page.offset + 1
    end = min(page.offset + len(page.runs), page.total)
    return f"Showing {start}–{end} of {page.total} saved runs"


def _page(*, title: str, body: str) -> str:
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_e(title)}</title>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            body,
            "</body>",
            "</html>",
            "",
        ]
    )


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def _format_timestamp(value: str) -> str:
    if value.endswith("+00:00"):
        return value.removesuffix("+00:00") + " UTC"
    if value.endswith("Z"):
        return value.removesuffix("Z") + " UTC"
    return value


def _e(value: object) -> str:
    return escape(str(value), quote=True)


_CSS = (
    ":root{color-scheme:light}"
    "*{box-sizing:border-box}"
    "html{max-width:100%;overflow-x:hidden}"
    "body{margin:0 auto;max-width:72rem;width:100%;padding:2rem 1.25rem 3rem;"
    "font:16px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif;"
    "color:#1a1a1a;background:#f6f4f0}"
    "header,section,footer{margin-bottom:2rem}"
    ".eyebrow{margin:0;letter-spacing:.08em;text-transform:uppercase;"
    "font-size:.75rem;color:#5c574f}"
    "h1{margin:.35rem 0 .75rem;font-size:1.85rem}"
    "h2{margin:0 0 .75rem;font-size:1.2rem;border-bottom:1px solid #d9d3c8;"
    "padding-bottom:.35rem}"
    ".lede,footer p,.muted,.empty{color:#5c574f}"
    ".muted{font-size:.92rem}"
    ".empty{background:#fff;border:1px solid #e4ddd2;padding:1rem 1.1rem}"
    "code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
    "font-size:.85em;word-break:break-all}"
    "a{color:#1d3f6e}"
    ".table-wrap{width:100%;max-width:100%;overflow-x:auto;"
    "-webkit-overflow-scrolling:touch;border:1px solid #e4ddd2;background:#fff}"
    "table.runs{min-width:56rem;width:100%;border-collapse:collapse;"
    "font-size:.92rem}"
    "table.runs th,table.runs td{text-align:left;padding:.55rem .7rem;"
    "border-bottom:1px solid #ece7de;vertical-align:top;white-space:nowrap}"
    "table.runs th{font-weight:600}"
    "label.pick{display:inline-flex;align-items:center;gap:.35rem;cursor:pointer}"
    "label.pick span{font-size:.85rem;color:#5c574f}"
    ".compare-form{margin:0}"
    ".compare-bar{display:flex;flex-wrap:wrap;gap:.75rem 1rem;align-items:center;"
    "margin:.85rem 0 0}"
    ".compare-bar button{font:inherit;padding:.4rem .8rem;background:#1d3f6e;"
    "color:#fff;border:0;cursor:pointer}"
    "nav.pager{display:flex;gap:1rem;margin-top:1rem;flex-wrap:wrap}"
    ".scroll-hint{display:none;margin:.35rem 0 .6rem}"
    "@media (max-width:40rem){body{padding:1.25rem .85rem 2rem}"
    ".scroll-hint{display:block}}"
    "footer{font-size:.9rem}"
)

_NAV_CSS = (
    ".sv-nav{display:flex;flex-wrap:wrap;gap:.75rem 1.25rem;align-items:center;"
    "margin:0 0 1rem;padding:.75rem 0;border-bottom:1px solid #d9d3c8;"
    "font:14px/1.4 ui-sans-serif,system-ui,-apple-system,sans-serif}"
    ".sv-nav a{color:#1d3f6e;text-decoration:none}"
    ".sv-local{color:#5c574f}"
    ".sv-run-meta{display:grid;gap:.35rem 1rem;margin:0 0 1.5rem;"
    "font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}"
    ".sv-run-meta>div{display:grid;grid-template-columns:minmax(7rem,28%) 1fr;"
    "gap:.35rem 1rem}"
    ".sv-run-meta dt{font-weight:600;color:#5c574f}"
    ".sv-run-meta dd{margin:0;overflow-wrap:anywhere;word-break:break-word}"
    ".sv-run-meta code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
    "font-size:.85em}"
)
