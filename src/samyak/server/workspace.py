"""Local Samyak workspace home. Offline, no JavaScript."""

from __future__ import annotations

from samyak.model.catalog import ModelCatalog
from samyak.server.layout import escape_html as _e
from samyak.server.layout import format_as_of_date, render_page
from samyak.server.model_catalog import (
    UPDATE_COMMAND,
    catalog_is_partial,
    lifecycle_counts,
)
from samyak.store.models import RunPage

CATALOG_MISSING = "missing"
CATALOG_CORRUPT = "corrupt"
CATALOG_SCHEMA = "schema"


def render_workspace(
    runs: RunPage,
    catalog: ModelCatalog | None,
    *,
    catalog_state: str | None = None,
) -> str:
    body = "".join(
        [
            "<header>",
            '<p class="eyebrow">Samyak</p>',
            "<h1>Local workspace</h1>",
            '<p class="lede">Local Samyak workspace. Data stored on this machine.</p>',
            "</header>",
            _corpus_section(runs),
            _model_section(catalog, catalog_state),
            "<footer>",
            "<p>This viewer reads local Samyak cache data. Nothing is sent over the network. "
            f"Refresh Model Intelligence with <code>{_e(UPDATE_COMMAND)}</code> or "
            "<code>samyak model update anthropic</code>.</p>",
            "</footer>",
        ]
    )
    return render_page(
        title="Samyak workspace",
        current="/",
        body=body,
    )


def _corpus_section(runs: RunPage) -> str:
    total = runs.total
    count = "1 run" if total == 1 else f"{total} runs"
    empty = ""
    if total == 0:
        empty = (
            '<p class="empty">No saved runs yet. Analyze a corpus with '
            "<code>samyak corpus PATH --save</code>, then refresh this page.</p>"
        )
    skipped = ""
    if runs.skipped:
        skipped = (
            f'<p class="muted">{_e(runs.skipped)} saved run(s) on this machine '
            "could not be read and were skipped.</p>"
        )
    return (
        "<section>"
        "<h2>Corpus Intelligence</h2>"
        "<h3>Run History</h3>"
        "<p>Your saved corpus analyses.</p>"
        f'<p class="muted">{_e(count)}</p>'
        f"{empty}{skipped}"
        '<p><a href="/runs">Open Run History</a></p>'
        "</section>"
    )


def _model_section(catalog: ModelCatalog | None, catalog_state: str | None) -> str:
    if catalog is None and catalog_state in {None, CATALOG_MISSING}:
        return (
            "<section>"
            "<h2>Model Intelligence</h2>"
            "<h3>Model Catalog</h3>"
            "<p>Provider model information verified from captured documentation.</p>"
            '<p class="empty">No local model catalog yet. Model Intelligence becomes '
            "available after "
            f"<code>{_e(UPDATE_COMMAND)}</code>. Then refresh this page.</p>"
            "</section>"
        )
    if catalog_state == CATALOG_SCHEMA:
        return (
            "<section>"
            "<h2>Model Intelligence</h2>"
            "<h3>Model Catalog</h3>"
            '<p class="empty">This catalog cannot be read by this Samyak version. '
            f"Update Samyak, or replace the catalog with <code>{_e(UPDATE_COMMAND)}</code>.</p>"
            "</section>"
        )
    if catalog_state == CATALOG_CORRUPT or catalog is None:
        return (
            "<section>"
            "<h2>Model Intelligence</h2>"
            "<h3>Model Catalog</h3>"
            '<p class="empty">The local model catalog could not be read. It may be corrupt. '
            f"Replace it with <code>{_e(UPDATE_COMMAND)}</code>.</p>"
            "</section>"
        )

    counts = lifecycle_counts(catalog.models)
    total = len(catalog.models)
    model_label = "1 model" if total == 1 else f"{total} models"
    parts = [
        model_label,
        f"{counts['active']} active",
    ]
    if counts.get("legacy", 0):
        parts.append(f"{counts['legacy']} legacy")
    parts.extend(
        [
            f"{counts['deprecated']} deprecated",
            f"{counts['retired']} retired",
        ]
    )
    summary = " · ".join(parts)
    as_of = format_as_of_date(catalog.freshness.generated_at)
    partial = ""
    if catalog_is_partial(catalog.notices):
        partial = '<p class="muted">Incomplete: some model pages could not be retrieved.</p>'
    return (
        "<section>"
        "<h2>Model Intelligence</h2>"
        "<h3>Model Catalog</h3>"
        "<p>Provider model information verified from captured documentation.</p>"
        f'<p class="muted">{_e(summary)}</p>'
        f'<p class="muted">As of {_e(as_of)}</p>'
        f"{partial}"
        '<p><a href="/models">Browse Model Catalog</a></p>'
        "</section>"
    )
