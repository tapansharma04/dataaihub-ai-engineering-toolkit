"""Shared local-viewer page chrome. No JavaScript and no external assets."""

from __future__ import annotations

from datetime import datetime
from html import escape

NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("/", "Workspace"),
    ("/runs", "Run History"),
    ("/models", "Model Catalog"),
)


def escape_html(value: object) -> str:
    return escape(str(value), quote=True)


def render_nav(*, current: str = "") -> str:
    links: list[str] = ['<a class="sv-brand" href="/">Samyak</a>']
    for href, label in NAV_ITEMS:
        klass = ' class="sv-current"' if current == href else ""
        aria = ' aria-current="page"' if current == href else ""
        links.append(f'<a href="{escape_html(href)}"{klass}{aria}>{escape_html(label)}</a>')
    links.append('<span class="sv-local">Local workspace</span>')
    return f'<nav class="sv-nav">{"".join(links)}</nav>'


def render_page(
    *,
    title: str,
    body: str,
    extra_css: str = "",
    current: str = "",
    include_nav: bool = True,
) -> str:
    nav = render_nav(current=current) if include_nav else ""
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape_html(title)}</title>",
            f"<style>{_BASE_CSS}{_NAV_CSS}{extra_css}</style>",
            "</head>",
            "<body>",
            nav,
            body,
            "</body>",
            "</html>",
            "",
        ]
    )


def format_timestamp(value: str) -> str:
    if value.endswith("+00:00"):
        return value.removesuffix("+00:00") + " UTC"
    if value.endswith("Z"):
        return value.removesuffix("Z") + " UTC"
    return value


def format_as_of_date(value: str) -> str:
    """Calendar date for freshness labels. Does not convert timezones."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return format_timestamp(value)
    return f"{parsed.strftime('%b')} {parsed.day}, {parsed.year}"


_BASE_CSS = (
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
    "h3{margin:.2rem 0 .5rem;font-size:1.05rem}"
    ".lede,footer p,.muted,.empty{color:#5c574f}"
    ".muted{font-size:.92rem}"
    ".empty{background:#fff;border:1px solid #e4ddd2;padding:1rem 1.1rem}"
    "code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
    "font-size:.85em;word-break:break-all}"
    "a{color:#1d3f6e}"
    ".banner{background:#fff;border:1px solid #e4ddd2;padding:.85rem 1rem}"
    ".banner.warn{border-color:#d9b08c;background:#fbf4ea}"
    ".table-wrap{width:100%;max-width:100%;overflow-x:auto;"
    "-webkit-overflow-scrolling:touch;border:1px solid #e4ddd2;background:#fff}"
    "@media (max-width:40rem){body{padding:1.25rem .85rem 2rem}}"
    "footer{font-size:.9rem}"
)

_NAV_CSS = (
    ".sv-nav{display:flex;flex-wrap:wrap;gap:.75rem 1.25rem;align-items:center;"
    "margin:0 0 1.25rem;padding:.75rem 0;border-bottom:1px solid #d9d3c8;"
    "font:14px/1.4 ui-sans-serif,system-ui,-apple-system,sans-serif}"
    ".sv-nav a{color:#1d3f6e;text-decoration:none}"
    ".sv-nav a.sv-brand{font-weight:650;color:#1a1a1a}"
    ".sv-nav a.sv-current{font-weight:650}"
    ".sv-local{color:#5c574f;margin-left:auto}"
    ".sv-run-meta{display:grid;gap:.35rem 1rem;margin:0 0 1.5rem;"
    "font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}"
    ".sv-run-meta>div{display:grid;grid-template-columns:minmax(7rem,28%) 1fr;"
    "gap:.35rem 1rem}"
    ".sv-run-meta dt{font-weight:600;color:#5c574f}"
    ".sv-run-meta dd{margin:0;overflow-wrap:anywhere;word-break:break-word}"
    ".sv-run-meta code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
    "font-size:.85em}"
)
