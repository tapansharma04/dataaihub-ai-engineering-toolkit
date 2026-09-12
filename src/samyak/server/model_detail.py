"""HTML for a single Model Intelligence record. Offline, no JavaScript."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from samyak.model.catalog import ModelCatalog
from samyak.model.facts import (
    AvailabilityScope,
    ContextWindow,
    ContextWindowKind,
    Fact,
    FactStatus,
    Modality,
    Provenance,
)
from samyak.model.identity import ModelIdentity
from samyak.model.records import RECORD_FACT_NAMES, ModelRecord
from samyak.server.layout import escape_html as _e
from samyak.server.layout import format_timestamp, render_page
from samyak.server.model_catalog import (
    format_token_count,
    lifecycle_html,
    model_href,
    render_freshness_block,
)

_PROVIDER_LABEL = {"openai": "OpenAI", "anthropic": "Anthropic"}
_FACT_STATUS_LABEL = {
    FactStatus.UNKNOWN: "Unknown",
    FactStatus.NOT_VERIFIED: "Not verified",
    FactStatus.NOT_APPLICABLE: "Not applicable",
    FactStatus.CONFLICT: "Conflict",
    FactStatus.KNOWN: "Known",
}
_FACT_LABEL = {
    "display_name": "Display name",
    "aliases": "Aliases",
    "resolves_to": "Resolves to",
    "family": "Family",
    "lifecycle": "Lifecycle",
    "deprecated_at": "Deprecated at",
    "retirement_at": "Retirement at",
    "replacement": "Replacement",
    "context_window": "Context window",
    "max_input_tokens": "Max input tokens",
    "max_output_tokens": "Max output tokens",
    "input_modalities": "Input modalities",
    "output_modalities": "Output modalities",
    "tool_calling": "Tool calling",
    "structured_output": "Structured output",
    "api_access": "API access",
    "availability_scope": "Availability",
    "regions": "Regions",
}


def render_model_not_found_page() -> str:
    return render_page(
        title="Model not found — Samyak",
        current="/models",
        extra_css=_CSS,
        body="".join(
            [
                "<header>",
                '<p class="eyebrow">Samyak</p>',
                "<h1>Not found</h1>",
                "<p>That model was not found.</p>",
                '<p><a href="/models">← Model Catalog</a></p>',
                "</header>",
            ]
        ),
    )


def render_model_detail(catalog: ModelCatalog, record: ModelRecord) -> str:
    present = {item.samyak_id: item for item in catalog.models}
    title = _heading(record)
    body = "".join(
        [
            '<p class="sv-back"><a href="/models">← Model Catalog</a></p>',
            "<header>",
            '<p class="eyebrow">Samyak</p>',
            f"<h1>{_e(title)}</h1>",
            f'<p class="lede"><code>{_e(record.samyak_id)}</code></p>',
            render_freshness_block(catalog.freshness, compact=True),
            "</header>",
            "<section>",
            "<h2>Lifecycle</h2>",
            f"<p>{lifecycle_html(record)}</p>",
            _dl(
                [
                    ("Deprecated at", _render_fact(record.deprecated_at)),
                    ("Retirement at", _render_fact(record.retirement_at)),
                ]
            ),
            "</section>",
            "<section>",
            "<h2>Identity</h2>",
            _dl(
                [
                    ("Provider", _e(_provider_label(record.provider_id))),
                    ("Provider model ID", f"<code>{_e(record.provider_model_id)}</code>"),
                    ("Kind", _e(record.identity_kind.value)),
                    ("Display name", _render_fact(record.display_name)),
                    ("Family", _render_fact(record.family)),
                    ("Aliases", _render_fact(record.aliases, format_known=_format_strings)),
                    (
                        "Resolves to",
                        _render_fact(
                            record.resolves_to,
                            format_known=lambda value: _identity_html(value, present),
                        ),
                    ),
                ]
            ),
            "</section>",
            "<section>",
            "<h2>Limits</h2>",
            _dl(
                [
                    (
                        "Context window",
                        _render_fact(record.context_window, format_known=_format_context),
                    ),
                    (
                        "Max input tokens",
                        _render_fact(record.max_input_tokens, format_known=_format_int),
                    ),
                    (
                        "Max output tokens",
                        _render_fact(record.max_output_tokens, format_known=_format_int),
                    ),
                ]
            ),
            "</section>",
            "<section>",
            "<h2>Capabilities</h2>",
            _dl(
                [
                    ("API access", _render_fact(record.api_access, format_known=_format_bool)),
                    ("Tool calling", _render_fact(record.tool_calling, format_known=_format_bool)),
                    (
                        "Structured output",
                        _render_fact(record.structured_output, format_known=_format_bool),
                    ),
                    (
                        "Input modalities",
                        _render_fact(record.input_modalities, format_known=_format_modalities),
                    ),
                    (
                        "Output modalities",
                        _render_fact(record.output_modalities, format_known=_format_modalities),
                    ),
                    (
                        "Availability",
                        _render_fact(record.availability_scope, format_known=_format_scope),
                    ),
                    ("Regions", _render_fact(record.regions, format_known=_format_strings)),
                ]
            ),
            "</section>",
            "<section>",
            "<h2>Replacement</h2>",
            _render_fact(
                record.replacement,
                format_known=lambda value: _replacements_html(value, present),
            ),
            "</section>",
            _evidence_section(catalog, record),
        ]
    )
    return render_page(
        title=f"{title} — Samyak",
        current="/models",
        extra_css=_CSS,
        body=body,
    )


def _heading(record: ModelRecord) -> str:
    if record.display_name.status is FactStatus.KNOWN and isinstance(
        record.display_name.value, str
    ):
        return record.display_name.value
    return record.provider_model_id


def _provider_label(provider_id: str) -> str:
    return _PROVIDER_LABEL.get(provider_id, provider_id)


def _render_fact(fact: Fact, *, format_known=None) -> str:
    if fact.status is FactStatus.KNOWN:
        formatter = format_known or (lambda value: _e(value))
        return formatter(fact.value)
    if fact.status is FactStatus.CONFLICT:
        return _conflict_html(fact, format_known=format_known)
    label = _FACT_STATUS_LABEL.get(fact.status, fact.status.value)
    return f'<span class="muted">{_e(label)}</span>'


def _conflict_html(fact: Fact, *, format_known=None) -> str:
    formatter = format_known or (lambda value: _e(value))
    claims = "".join(
        "<li>"
        f"{formatter(claim.value)}"
        f' <span class="muted">({_e(claim.provenance.source_id)})</span>'
        "</li>"
        for claim in fact.claims
    )
    return (
        '<p class="banner warn">Conflict: Samyak did not resolve competing evidence.</p>'
        f"<ul class='claims'>{claims}</ul>"
    )


def _format_bool(value: object) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return _e(value)


def _format_int(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return _e(format_token_count(value))
    return _e(value)


def _format_strings(value: object) -> str:
    if isinstance(value, tuple):
        return _e(", ".join(str(item) for item in value))
    return _e(value)


def _format_modalities(value: object) -> str:
    if isinstance(value, tuple):
        labels = [item.value if isinstance(item, Modality) else str(item) for item in value]
        return _e(", ".join(labels))
    return _e(value)


def _format_scope(value: object) -> str:
    if isinstance(value, AvailabilityScope):
        return _e(value.value)
    return _e(value)


def _format_context(value: object) -> str:
    if not isinstance(value, ContextWindow):
        return _e(value)
    tokens = format_token_count(value.tokens)
    if value.kind is ContextWindowKind.UNKNOWN:
        return _e(tokens)
    return _e(f"{tokens} ({value.kind.value})")


def _identity_html(value: object, present: Mapping[str, ModelRecord]) -> str:
    if not isinstance(value, ModelIdentity):
        return _e(value)
    return _linked_identity(value, present)


def _replacements_html(value: object, present: Mapping[str, ModelRecord]) -> str:
    if not isinstance(value, tuple):
        return _e(value)
    items = "".join(f"<li>{_linked_identity(item, present)}</li>" for item in value)
    return f"<ul class='replacements'>{items}</ul>"


def _linked_identity(identity: ModelIdentity, present: Mapping[str, ModelRecord]) -> str:
    label = f"<code>{_e(identity.samyak_id)}</code>"
    if identity.samyak_id in present:
        return f'<a href="{_e(model_href(identity.samyak_id))}">{label}</a>'
    return f'{label} <span class="muted">not in this catalog</span>'


def _dl(rows: Sequence[tuple[str, str]]) -> str:
    items = "".join(f"<div><dt>{_e(label)}</dt><dd>{value}</dd></div>" for label, value in rows)
    return f'<dl class="meta">{items}</dl>'


def _evidence_section(catalog: ModelCatalog, record: ModelRecord) -> str:
    fact_blocks = []
    for name in RECORD_FACT_NAMES:
        fact = getattr(record, name)
        if not isinstance(fact, Fact):
            continue
        if fact.status is FactStatus.NOT_VERIFIED:
            continue
        block = _evidence_fact(name, fact)
        if block:
            fact_blocks.append(block)
    return (
        "<section>"
        "<details>"
        "<summary>Evidence &amp; freshness</summary>"
        f"{render_freshness_block(catalog.freshness)}"
        f"{''.join(fact_blocks)}"
        "</details>"
        "</section>"
    )


def _evidence_fact(name: str, fact: Fact) -> str:
    label = _FACT_LABEL.get(name, name)
    if fact.status is FactStatus.CONFLICT:
        claims = "".join(
            _provenance_block(claim.provenance, heading=_claim_heading(claim.value))
            for claim in fact.claims
        )
        return f"<h3>{_e(label)}</h3>{claims}"
    if fact.provenance is None:
        return ""
    return f"<h3>{_e(label)}</h3>{_provenance_block(fact.provenance)}"


def _claim_heading(value: object) -> str:
    if isinstance(value, ModelIdentity):
        return _e(value.samyak_id)
    if isinstance(value, tuple):
        return _e(", ".join(str(item) for item in value))
    if isinstance(value, ContextWindow):
        return _e(format_token_count(value.tokens))
    return _e(value)


def _provenance_block(provenance: Provenance, heading: str | None = None) -> str:
    title = f"<p>{heading}</p>" if heading is not None else ""
    url = ""
    if provenance.source_url:
        url = f"<div><dt>Source URL</dt><dd><code>{_e(provenance.source_url)}</code></dd></div>"
    digest = ""
    if provenance.content_hash:
        digest = (
            f"<div><dt>Content hash</dt><dd><code>{_e(provenance.content_hash)}</code></dd></div>"
        )
    retrieved = ""
    if provenance.retrieved_at:
        retrieved = (
            f"<div><dt>Retrieved</dt><dd>{_e(format_timestamp(provenance.retrieved_at))}</dd></div>"
        )
    verified = ""
    if provenance.verified_at:
        verified = (
            f"<div><dt>Verified</dt><dd>{_e(format_timestamp(provenance.verified_at))}</dd></div>"
        )
    observed = ""
    if provenance.observed_at:
        observed = (
            f"<div><dt>Observed</dt><dd>{_e(format_timestamp(provenance.observed_at))}</dd></div>"
        )
    confidence = ""
    if provenance.confidence is not None:
        confidence = f"<div><dt>Confidence</dt><dd>{_e(provenance.confidence.value)}</dd></div>"
    return (
        f"{title}"
        '<dl class="meta evidence">'
        f"<div><dt>Source type</dt><dd>{_e(provenance.source_kind.value)}</dd></div>"
        f"<div><dt>Source ID</dt><dd><code>{_e(provenance.source_id)}</code></dd></div>"
        f"{url}{digest}{retrieved}{verified}{observed}{confidence}"
        "</dl>"
    )


_CSS = (
    "body{max-width:52rem}"
    "p.sv-back{margin:0 0 1rem;font-size:.92rem}"
    "dl.meta{display:grid;gap:.35rem 1rem;margin:.5rem 0 0}"
    "dl.meta>div{display:grid;grid-template-columns:minmax(8.5rem,32%) 1fr;gap:.35rem 1rem}"
    "dl.meta dt{font-weight:600;color:#5c574f}"
    "dl.meta dd{margin:0;overflow-wrap:anywhere;word-break:break-word}"
    "ul.claims,ul.replacements{margin:.35rem 0 0;padding-left:1.2rem}"
    "details{background:#fff;border:1px solid #e4ddd2;padding:.85rem 1rem}"
    "details summary{cursor:pointer;font-weight:650}"
    "details h3{margin:1rem 0 .35rem;font-size:.95rem}"
    ".life{display:inline-block;padding:.1rem .45rem;font-size:.75rem;"
    "font-weight:700;letter-spacing:.04em}"
    ".life-active{background:#d5e4f5;color:#1d3f6e}"
    ".life-legacy{background:#e8e1f4;color:#3d2b66}"
    ".life-deprecated{background:#f7e1b5;color:#6d4a00}"
    ".life-retired{background:#e3e3e3;color:#333}"
    ".banner{background:#fff;border:1px solid #e4ddd2;padding:.85rem 1rem}"
    ".banner.warn{border-color:#d9b08c;background:#fbf4ea}"
)
