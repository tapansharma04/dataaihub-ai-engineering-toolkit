"""HTML for the local run-comparison page."""

from __future__ import annotations

from html import escape

from samyak.comparison.models import (
    CORPUS_CROSS,
    CORPUS_DIFFERENT_NAME_UNCONFIRMED,
    CORPUS_SAME_LOCATION,
    CORPUS_SAME_NAME_UNCONFIRMED,
    ConfigChange,
    FieldChange,
    FindingMatch,
    MetricChange,
    RunComparison,
)
from samyak.corpus.models import Finding
from samyak.store.models import StoredRun


def render_comparison_page(result: RunComparison) -> str:
    baseline = result.baseline
    current = result.current
    body = "".join(
        [
            "<header>",
            '<p class="eyebrow">Samyak</p>',
            "<h1>Run comparison</h1>",
            '<p class="lede">Local comparison of two saved analysis runs. '
            "Deltas are baseline → current. Nothing is sent over the network.</p>",
            "</header>",
            _runs_header(baseline, current, result),
            _corpus_banner(result),
            _summary_section(result),
            _finding_section("New findings", result.new, empty="No new findings."),
            _finding_section(
                "No longer detected",
                result.no_longer_detected,
                empty="No findings disappeared between baseline and current.",
            ),
            _finding_section("Changed findings", result.changed, empty="No changed findings."),
            _unchanged_section(result.unchanged),
            "<footer>",
            '<p><a href="/">← Run History</a></p>',
            "</footer>",
        ]
    )
    return _page(title="Compare runs — Samyak", body=body)


def _runs_header(baseline: StoredRun, current: StoredRun, result: RunComparison) -> str:
    return (
        "<section>"
        "<h2>Runs</h2>"
        '<div class="run-pair">'
        f"{_run_card(baseline, 'Baseline')}"
        f"{_run_card(current, 'Current')}"
        "</div>"
        f"{_version_note_html(result.version_note)}"
        "</section>"
    )


def _run_card(stored: StoredRun, heading: str) -> str:
    meta = stored.metadata
    label = meta.corpus_label or stored.report.summary.corpus_root
    path = ""
    if meta.corpus_path:
        path = f"<div><dt>Corpus path</dt><dd><code>{_e(meta.corpus_path)}</code></dd></div>"
    href = f"/runs/{_e(meta.run_id)}"
    return (
        '<article class="run-card">'
        f"<h3>{_e(heading)}</h3>"
        "<dl>"
        f"<div><dt>Timestamp</dt><dd>{_e(_format_timestamp(meta.created_at))}</dd></div>"
        f'<div><dt>Run ID</dt><dd><a href="{href}"><code>{_e(meta.run_id)}</code></a></dd></div>'
        f"<div><dt>Corpus</dt><dd>{_e(label)}</dd></div>"
        f"{path}"
        f"<div><dt>Samyak</dt><dd>{_e(stored.report.version)}</dd></div>"
        "</dl>"
        "</article>"
    )


def _version_note_html(note: str | None) -> str:
    if not note:
        return ""
    return f'<p class="banner">{_e(note)}</p>'


def _corpus_banner(result: RunComparison) -> str:
    kind = result.corpus.kind
    if kind == CORPUS_CROSS:
        text = (
            "Different saved corpus locations. Matching finding codes does not mean "
            "the documents are the same, and a finding that is no longer detected "
            "is not proof that a problem was fixed."
        )
        return f'<p class="banner warn">{_e(text)}</p>'
    if kind == CORPUS_DIFFERENT_NAME_UNCONFIRMED:
        text = (
            "These runs have different corpus names, and at least one is missing a "
            "saved corpus path. They may not be the same document collection."
        )
        return f'<p class="banner warn">{_e(text)}</p>'
    if kind == CORPUS_SAME_NAME_UNCONFIRMED:
        text = (
            "These runs share a corpus folder name, but a full corpus path was not "
            "recorded for one or both. Matching folder names does not mean they are "
            "the same corpus."
        )
        return f'<p class="banner">{_e(text)}</p>'
    if kind == CORPUS_SAME_LOCATION:
        text = "These runs recorded the same saved corpus path."
        return f'<p class="muted">{_e(text)}</p>'
    return ""


def _summary_section(result: RunComparison) -> str:
    n_new, n_gone, n_changed, n_unchanged = result.finding_counts
    metrics = "".join(_metric_row(item) for item in result.metrics)
    config = _config_section(result.config_changes)
    return (
        "<section>"
        "<h2>Summary</h2>"
        '<p class="muted">Values are baseline → current.</p>'
        f'<table class="metrics"><tbody>{metrics}</tbody></table>'
        f"{config}"
        "<h3>Finding changes</h3>"
        '<ul class="counts">'
        f"<li>New: {_e(n_new)}</li>"
        f"<li>No longer detected: {_e(n_gone)}</li>"
        f"<li>Changed: {_e(n_changed)}</li>"
        f"<li>Unchanged: {_e(n_unchanged)}</li>"
        "</ul>"
        "</section>"
    )


def _config_section(changes: tuple[ConfigChange, ...]) -> str:
    if not changes:
        return ""
    rows = "".join(
        "<tr>"
        f"<th><code>{_e(item.key)}</code></th>"
        f"<td>{_e(item.before)} → {_e(item.after)}</td>"
        "</tr>"
        for item in changes
    )
    return (
        '<p class="banner warn">Analysis configuration changed. Inventory and '
        "finding differences may come from threshold changes rather than from "
        "the corpus itself.</p>"
        "<h3>Configuration</h3>"
        f'<table class="metrics"><tbody>{rows}</tbody></table>'
    )


def _metric_row(item: MetricChange) -> str:
    klass = ' class="changed"' if item.changed else ""
    return f"<tr{klass}><th>{_e(item.label)}</th><td>{_e(item.before)} → {_e(item.after)}</td></tr>"


def _finding_section(title: str, matches: tuple[FindingMatch, ...], *, empty: str) -> str:
    if not matches:
        return f"<section><h2>{_e(title)}</h2><p class='muted'>{_e(empty)}</p></section>"
    cards = "".join(_match_card(item) for item in matches)
    return f"<section><h2>{_e(title)}</h2>{cards}</section>"


def _unchanged_section(matches: tuple[FindingMatch, ...]) -> str:
    if not matches:
        return (
            "<section><h2>Unchanged findings</h2>"
            '<p class="muted">No unchanged findings.</p></section>'
        )
    items = "".join(f"<li><code>{_e(item.code)}</code> {_e(item.title)}</li>" for item in matches)
    return f'<section><h2>Unchanged findings</h2><ul class="unchanged">{items}</ul></section>'


def _match_card(match: FindingMatch) -> str:
    finding = match.after or match.before
    if finding is None:
        return ""
    if match.status == "changed":
        return _changed_card(match, finding)
    return _simple_card(match, finding)


def _simple_card(match: FindingMatch, finding: Finding) -> str:
    sev = finding.severity.value
    count = _affected_label(finding)
    return (
        f'<article class="finding {_e(match.status)}">'
        "<header>"
        f'<span class="sev sev-{_e(sev.lower())}">{_e(sev)}</span> '
        f'<code class="code">{_e(finding.code)}</code>'
        f"<h3>{_e(finding.title)}</h3>"
        "</header>"
        f"<p>{_e(finding.message)}</p>"
        f"{count}"
        "</article>"
    )


def _changed_card(match: FindingMatch, finding: Finding) -> str:
    sev = finding.severity.value
    rows = "".join(_change_block(change) for change in match.changes)
    return (
        f'<article class="finding changed">'
        "<header>"
        f'<span class="sev sev-{_e(sev.lower())}">{_e(sev)}</span> '
        f'<code class="code">{_e(finding.code)}</code>'
        f"<h3>{_e(finding.title)}</h3>"
        "</header>"
        f"{rows}"
        "</article>"
    )


def _change_block(change: FieldChange) -> str:
    extra = ""
    if change.added_paths or change.removed_paths:
        added = "".join(f"<li><code>{_e(path)}</code></li>" for path in change.added_paths)
        removed = "".join(f"<li><code>{_e(path)}</code></li>" for path in change.removed_paths)
        if added:
            extra += f'<p class="muted">Added</p><ul class="paths">{added}</ul>'
        if removed:
            extra += f'<p class="muted">Removed</p><ul class="paths">{removed}</ul>'
    label = change.field.replace("_", " ").capitalize()
    if change.field == "affected documents":
        value = f"{_e(change.before)} → {_e(change.after)}"
        return f"<div class='delta'><h4>{_e(label)}</h4><p>{value}</p>{extra}</div>"
    if change.field == "severity":
        return (
            f"<div class='delta'><h4>{_e(label)}</h4>"
            f"<p>{_e(change.before)} → {_e(change.after)}</p></div>"
        )
    if change.field.startswith("evidence."):
        return (
            f"<div class='delta'><h4>{_e(label)}</h4>"
            f"<p>{_e(change.before)} → {_e(change.after)}</p></div>"
        )
    return (
        f"<div class='delta'><h4>{_e(label)}</h4>"
        f'<p class="muted">Before</p><p>{_e(change.before)}</p>'
        f'<p class="muted">After</p><p>{_e(change.after)}</p></div>'
    )


def _affected_label(finding: Finding) -> str:
    raw = finding.evidence.get("affected_document_count")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raw = len(finding.affected_documents)
    if not raw:
        return ""
    return f'<p class="muted">Affected documents: {_e(raw)}</p>'


def _format_timestamp(value: str) -> str:
    if value.endswith("+00:00"):
        return value.removesuffix("+00:00") + " UTC"
    if value.endswith("Z"):
        return value.removesuffix("Z") + " UTC"
    return value


def _e(value: object) -> str:
    return escape(str(value), quote=True)


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


_CSS = (
    ":root{color-scheme:light}"
    "*{box-sizing:border-box}"
    "html{max-width:100%;overflow-x:hidden}"
    "body{margin:0 auto;max-width:52rem;width:100%;padding:2rem 1.25rem 3rem;"
    "font:16px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif;"
    "color:#1a1a1a;background:#f6f4f0}"
    "header,section,footer{margin-bottom:2rem}"
    ".eyebrow{margin:0;letter-spacing:.08em;text-transform:uppercase;"
    "font-size:.75rem;color:#5c574f}"
    "h1{margin:.35rem 0 .75rem;font-size:1.85rem}"
    "h2{margin:0 0 .75rem;font-size:1.2rem;border-bottom:1px solid #d9d3c8;"
    "padding-bottom:.35rem}"
    "h3{margin:.2rem 0 .5rem;font-size:1.05rem}"
    "h4{margin:.85rem 0 .35rem;font-size:.85rem;text-transform:uppercase;"
    "letter-spacing:.04em;color:#5c574f}"
    ".lede,footer p,.muted{color:#5c574f}"
    ".muted{font-size:.92rem}"
    "code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
    "font-size:.85em;word-break:break-word}"
    "a{color:#1d3f6e}"
    ".run-pair{display:grid;gap:1rem}"
    "@media (min-width:40rem){.run-pair{grid-template-columns:1fr 1fr}}"
    ".run-card{background:#fff;border:1px solid #e4ddd2;padding:1rem 1.1rem}"
    ".run-card dl{display:grid;gap:.35rem 1rem;margin:.5rem 0 0}"
    ".run-card dl>div{display:grid;grid-template-columns:minmax(6.5rem,32%) 1fr;"
    "gap:.35rem 1rem}"
    ".run-card dt{font-weight:600;color:#5c574f}"
    ".run-card dd{margin:0;overflow-wrap:anywhere;word-break:break-word}"
    ".banner{background:#fff;border:1px solid #e4ddd2;padding:.85rem 1rem}"
    ".banner.warn{border-color:#d9b08c;background:#fbf4ea}"
    "table.metrics{width:100%;border-collapse:collapse;background:#fff;"
    "border:1px solid #e4ddd2}"
    "table.metrics th,table.metrics td{text-align:left;padding:.45rem .7rem;"
    "border-bottom:1px solid #ece7de;vertical-align:top}"
    "table.metrics th{width:45%;font-weight:600}"
    "table.metrics tr.changed td{font-weight:600}"
    "ul.counts,ul.unchanged,ul.paths{margin:.35rem 0 0;padding-left:1.2rem}"
    ".finding{background:#fff;border:1px solid #e4ddd2;padding:1rem 1.1rem;margin:.75rem 0}"
    ".finding p{overflow-wrap:anywhere}"
    ".delta{overflow-wrap:anywhere}"
    ".sev{display:inline-block;padding:.1rem .45rem;font-size:.75rem;"
    "font-weight:700;letter-spacing:.04em}"
    ".sev-high{background:#f8d0c8;color:#7a1f12}"
    ".sev-medium{background:#f7e1b5;color:#6d4a00}"
    ".sev-low{background:#d5e4f5;color:#1d3f6e}"
    ".sev-info{background:#e3e3e3;color:#333}"
    "footer{font-size:.9rem}"
    "@media (max-width:40rem){body{padding:1.25rem .85rem 2rem}}"
)
