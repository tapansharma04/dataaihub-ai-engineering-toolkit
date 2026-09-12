"""HTML for Model Intelligence catalog listing. Offline, no JavaScript."""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import quote

from samyak.model.catalog import (
    CatalogFreshness,
    CatalogNotice,
    CatalogOverlay,
    FreshnessStatus,
    ModelCatalog,
)
from samyak.model.facts import ContextWindow, FactStatus, LifecycleState
from samyak.model.records import ModelRecord
from samyak.server.layout import escape_html as _e
from samyak.server.layout import format_as_of_date, render_page

PARTIAL_NOTICE_PREFIX = "PARTIAL_MODEL_PAGES"
LIFECYCLE_FILTERS = frozenset({"all", "active", "legacy", "deprecated", "retired", "uncertain"})
UPDATE_COMMAND = "samyak model update openai"

_FRESHNESS_STATUS_LABEL = {
    FreshnessStatus.AS_OF: "As of",
    FreshnessStatus.STALE: "Stale",
    FreshnessStatus.UNKNOWN: "Unknown",
}
_OVERLAY_LABEL = {
    CatalogOverlay.USER_CACHE: "user cache",
    CatalogOverlay.BUNDLED: "bundled",
}
_LIFECYCLE_FILTER_LABEL = {
    "all": "All",
    "active": "Active",
    "legacy": "Legacy",
    "deprecated": "Deprecated",
    "retired": "Retired",
    "uncertain": "Uncertain",
}


def model_href(samyak_id: str) -> str:
    return "/models/" + quote(samyak_id, safe=":@._-")


def format_token_count(tokens: int) -> str:
    if tokens >= 1000 and tokens % 1000 == 0:
        thousands = tokens // 1000
        if thousands >= 1000 and thousands % 1000 == 0:
            return f"{thousands // 1000}M"
        return f"{thousands}K"
    return str(tokens)


def lifecycle_matches(record: ModelRecord, lifecycle: str) -> bool:
    if lifecycle == "all":
        return True
    fact = record.lifecycle
    if lifecycle == "uncertain":
        return fact.status is not FactStatus.KNOWN
    if fact.status is not FactStatus.KNOWN:
        return False
    value = fact.value
    return isinstance(value, LifecycleState) and value.value == lifecycle


def search_matches(record: ModelRecord, query: str) -> bool:
    needle = query.strip().lower()
    if not needle:
        return True
    haystacks = [record.samyak_id, record.provider_model_id, record.provider_id]
    if record.display_name.status is FactStatus.KNOWN and isinstance(
        record.display_name.value, str
    ):
        haystacks.append(record.display_name.value)
    if record.aliases.status is FactStatus.KNOWN and record.aliases.value:
        haystacks.extend(record.aliases.value)
    return any(needle in item.lower() for item in haystacks)


def filter_models(
    models: Sequence[ModelRecord], *, q: str = "", lifecycle: str = "all"
) -> tuple[ModelRecord, ...]:
    chosen = lifecycle if lifecycle in LIFECYCLE_FILTERS else "all"
    return tuple(
        record
        for record in models
        if lifecycle_matches(record, chosen) and search_matches(record, q)
    )


def catalog_is_partial(notices: Sequence[CatalogNotice]) -> bool:
    return any(notice.code.startswith(PARTIAL_NOTICE_PREFIX) for notice in notices)


def lifecycle_counts(models: Sequence[ModelRecord]) -> dict[str, int]:
    counts = {state.value: 0 for state in LifecycleState}
    counts["uncertain"] = 0
    for record in models:
        fact = record.lifecycle
        if fact.status is FactStatus.KNOWN and isinstance(fact.value, LifecycleState):
            counts[fact.value.value] += 1
        else:
            counts["uncertain"] += 1
    return counts


def render_freshness_block(freshness: CatalogFreshness, *, compact: bool = False) -> str:
    status = _FRESHNESS_STATUS_LABEL.get(freshness.status, freshness.status.value)
    as_of = format_as_of_date(freshness.generated_at)
    overlay = _OVERLAY_LABEL.get(freshness.overlay, freshness.overlay.value)
    if compact:
        return f'<p class="muted">{_e(status)} {_e(as_of)}</p>'
    rows = [
        f"<div><dt>Status</dt><dd>{_e(status)}</dd></div>",
        f"<div><dt>As of</dt><dd>{_e(as_of)}</dd></div>",
        f"<div><dt>Overlay</dt><dd>{_e(overlay)}</dd></div>",
    ]
    if freshness.oldest_verified_at:
        rows.append(
            "<div><dt>Oldest evidence</dt>"
            f"<dd>{_e(format_as_of_date(freshness.oldest_verified_at))}</dd></div>"
        )
    if freshness.stale_after:
        rows.append(
            "<div><dt>Labeled stale after</dt>"
            f"<dd>{_e(format_as_of_date(freshness.stale_after))}</dd></div>"
        )
    return f'<dl class="meta">{"".join(rows)}</dl>'


def render_notices(notices: Sequence[CatalogNotice]) -> str:
    if not notices:
        return ""
    items = "".join(
        f"<li><code>{_e(notice.code)}</code> {_e(notice.message)}</li>" for notice in notices
    )
    banner = ""
    if catalog_is_partial(notices):
        banner = (
            '<p class="banner warn">This catalog is incomplete. Some model pages '
            "could not be retrieved, and facts from those pages are omitted.</p>"
        )
    return f"{banner}<ul class='notices'>{items}</ul>"


def render_catalog_missing_page() -> str:
    return render_page(
        title="Model Catalog — Samyak",
        current="/models",
        extra_css=_CSS,
        body=_missing_body(),
    )


def render_catalog_unavailable_page(*, unsupported_schema: bool) -> str:
    if unsupported_schema:
        title = "Unsupported catalog schema — Samyak"
        heading = "Unsupported catalog schema"
        message = (
            "This catalog cannot be read by this Samyak version. "
            f"Update Samyak, or replace the catalog with <code>{_e(UPDATE_COMMAND)}</code>."
        )
    else:
        title = "Model catalog unavailable — Samyak"
        heading = "Model catalog unavailable"
        message = (
            "The local model catalog could not be read. It may be corrupt. "
            f"Replace it with <code>{_e(UPDATE_COMMAND)}</code>."
        )
    return render_page(
        title=title,
        current="/models",
        extra_css=_CSS,
        body="".join(
            [
                "<header>",
                '<p class="eyebrow">Samyak</p>',
                f"<h1>{_e(heading)}</h1>",
                f"<p>{message}</p>",
                '<p><a href="/">Workspace</a></p>',
                "</header>",
            ]
        ),
    )


def render_model_catalog(catalog: ModelCatalog, *, q: str = "", lifecycle: str = "all") -> str:
    chosen = lifecycle if lifecycle in LIFECYCLE_FILTERS else "all"
    filtered = filter_models(catalog.models, q=q, lifecycle=chosen)
    body = "".join(
        [
            "<header>",
            '<p class="eyebrow">Samyak</p>',
            "<h1>Model Catalog</h1>",
            '<p class="lede">Provider model information verified from captured documentation.</p>',
            "</header>",
            "<section>",
            "<h2>Catalog</h2>",
            render_freshness_block(catalog.freshness),
            _provider_line(catalog),
            render_notices(catalog.notices),
            f'<p class="muted">Refresh with <code>{_e(UPDATE_COMMAND)}</code> or '
            "<code>samyak model update anthropic</code>.</p>",
            "</section>",
            "<section>",
            "<h2>Models</h2>",
            _filter_form(q=q, lifecycle=chosen),
            _catalog_table(catalog, filtered),
            "</section>",
        ]
    )
    return render_page(
        title="Model Catalog — Samyak",
        current="/models",
        extra_css=_CSS,
        body=body,
    )


def _missing_body() -> str:
    return "".join(
        [
            "<header>",
            '<p class="eyebrow">Samyak</p>',
            "<h1>Model Catalog</h1>",
            '<p class="lede">Provider model information verified from captured documentation.</p>',
            "</header>",
            '<p class="empty">No local model catalog yet. Model Intelligence becomes '
            "available after "
            f"<code>{_e(UPDATE_COMMAND)}</code>. Then refresh this page. "
            "Browsing stays offline.</p>",
        ]
    )


def _provider_line(catalog: ModelCatalog) -> str:
    if not catalog.providers:
        return ""
    parts = [f"{_e(item.id)}: {_e(item.model_count)}" for item in catalog.providers]
    return f'<p class="muted">Providers: {", ".join(parts)}</p>'


def _filter_form(*, q: str, lifecycle: str) -> str:
    options = []
    for value in ("all", "active", "legacy", "deprecated", "retired", "uncertain"):
        selected = " selected" if value == lifecycle else ""
        options.append(
            f'<option value="{_e(value)}"{selected}>{_e(_LIFECYCLE_FILTER_LABEL[value])}</option>'
        )
    return (
        '<form class="filters" method="get" action="/models">'
        '<label>Search <input type="search" name="q" '
        f'value="{_e(q)}" aria-label="Search models"></label>'
        "<label>Lifecycle "
        f'<select name="lifecycle" aria-label="Lifecycle">{"".join(options)}</select>'
        "</label>"
        '<button type="submit">Apply</button>'
        "</form>"
    )


def _catalog_table(catalog: ModelCatalog, models: Sequence[ModelRecord]) -> str:
    if not catalog.models:
        return '<p class="empty">This catalog contains no models.</p>'
    if not models:
        return '<p class="empty">No models match this search.</p>'
    rows = "".join(_model_row(record) for record in models)
    hint = '<p class="scroll-hint muted">On a narrow screen, scroll the table sideways.</p>'
    return (
        f"{hint}"
        '<div class="table-wrap">'
        '<table class="models">'
        "<thead><tr><th>Model</th><th>Lifecycle</th><th>Context</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</div>"
    )


def _model_row(record: ModelRecord) -> str:
    name = _model_label(record)
    href = model_href(record.samyak_id)
    return (
        "<tr>"
        "<td>"
        f'<a href="{_e(href)}">{_e(name)}</a>'
        f'<div class="muted id">{_e(record.samyak_id)}</div>'
        "</td>"
        f"<td>{lifecycle_html(record)}</td>"
        f"<td>{context_html(record)}</td>"
        "</tr>"
    )


def _model_label(record: ModelRecord) -> str:
    if record.display_name.status is FactStatus.KNOWN and isinstance(
        record.display_name.value, str
    ):
        return record.display_name.value
    return record.provider_model_id


def lifecycle_html(record: ModelRecord) -> str:
    fact = record.lifecycle
    if fact.status is FactStatus.KNOWN and isinstance(fact.value, LifecycleState):
        label = fact.value.value.capitalize()
        return f'<span class="life life-{_e(fact.value.value)}">{_e(label)}</span>'
    return f'<span class="muted">{_e(_status_label(fact.status))}</span>'


def context_html(record: ModelRecord) -> str:
    fact = record.context_window
    if fact.status is FactStatus.KNOWN and isinstance(fact.value, ContextWindow):
        return _e(format_token_count(fact.value.tokens))
    return f'<span class="muted">{_e(_status_label(fact.status))}</span>'


def _status_label(status: FactStatus) -> str:
    labels = {
        FactStatus.UNKNOWN: "Unknown",
        FactStatus.NOT_VERIFIED: "Not verified",
        FactStatus.NOT_APPLICABLE: "Not applicable",
        FactStatus.CONFLICT: "Conflict",
        FactStatus.KNOWN: "Known",
    }
    return labels.get(status, status.value)


_CSS = (
    "dl.meta{display:grid;gap:.35rem 1rem;margin:.5rem 0 1rem}"
    "dl.meta>div{display:grid;grid-template-columns:minmax(8rem,28%) 1fr;gap:.35rem 1rem}"
    "dl.meta dt{font-weight:600;color:#5c574f}"
    "dl.meta dd{margin:0}"
    "ul.notices{margin:.75rem 0 0;padding-left:1.2rem}"
    "form.filters{display:flex;flex-wrap:wrap;gap:.75rem 1rem;align-items:end;margin:0 0 1rem}"
    "form.filters label{display:grid;gap:.25rem;font-size:.92rem;color:#5c574f}"
    "form.filters input,form.filters select,form.filters button{font:inherit}"
    "form.filters input,form.filters select{padding:.3rem .45rem;min-width:10rem}"
    "form.filters button{padding:.4rem .8rem;background:#1d3f6e;color:#fff;border:0;cursor:pointer}"
    "table.models{min-width:28rem;width:100%;border-collapse:collapse;font-size:.92rem}"
    "table.models th,table.models td{text-align:left;padding:.55rem .7rem;"
    "border-bottom:1px solid #ece7de;vertical-align:top}"
    "table.models th{font-weight:600}"
    "table.models .id{margin:.15rem 0 0;word-break:break-all}"
    ".life{display:inline-block;padding:.1rem .45rem;font-size:.75rem;"
    "font-weight:700;letter-spacing:.04em}"
    ".life-active{background:#d5e4f5;color:#1d3f6e}"
    ".life-legacy{background:#e8e1f4;color:#3d2b66}"
    ".life-deprecated{background:#f7e1b5;color:#6d4a00}"
    ".life-retired{background:#e3e3e3;color:#333}"
    ".scroll-hint{display:none;margin:.35rem 0 .6rem}"
    "@media (max-width:40rem){.scroll-hint{display:block}}"
)
