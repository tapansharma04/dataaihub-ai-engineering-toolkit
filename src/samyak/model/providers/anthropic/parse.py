"""Parse captured Anthropic Markdown into provider observations.

Claude API only. Partner-platform IDs and lifecycle tables are ignored.

Boolean capabilities:
- omitted / unlabeled → not extracted (later unknown or not_verified)
- explicit "tool use" on the current-lineup overview → tool_calling True
- structured_output is never inferred from JSON, JSON mode, or tool use
- api_access is not inferred from appearing on the overview
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from urllib.parse import urlparse

from samyak.model.identity import IdentityKind
from samyak.model.markdown import (
    code_spans,
    has_heading,
    labeled_value,
    markdown_links,
    normalize_heading,
    pipe_tables,
    section_named,
    split_sections,
)
from samyak.model.providers.anthropic.errors import AnthropicParseError
from samyak.model.providers.anthropic.observations import (
    AnthropicLifecycle,
    AnthropicModelObservation,
    AnthropicSourceRef,
)
from samyak.model.providers.anthropic.sources import AnthropicSourceType, CapturedSource

_MODEL_ID = re.compile(r"^claude-[a-z0-9][a-z0-9._-]*$")
_MODEL_PAGE_PATH = re.compile(
    r"^/docs/en/models/([a-z0-9][a-z0-9._-]*)/overview(?:\.md)?$",
    re.IGNORECASE,
)
_SKIP_PAGE_SLUGS = frozenset({"overview"})
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_TOKEN = re.compile(
    r"^(?P<num>[\d,.]+)\s*(?P<unit>[kKmM])?(?:\s*tokens?)?\s*$",
)
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_ENGLISH_DATE = re.compile(r"^(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})$")
_COMMITMENT = re.compile(r"not\s+sooner\s+than", re.IGNORECASE)
_PROSE_DEPRECATED = re.compile(
    r"`(claude-[a-z0-9][a-z0-9._-]*)`\)?\s+is deprecated\b",
    re.IGNORECASE,
)
_FAMILY_PREFIXES = (
    ("claude fable", "Claude Fable"),
    ("claude mythos", "Claude Mythos"),
    ("claude opus", "Claude Opus"),
    ("claude sonnet", "Claude Sonnet"),
    ("claude haiku", "Claude Haiku"),
    ("claude instant", "Claude Instant"),
)
_CURRENT_CAPABILITIES = re.compile(
    r"all current models support\s+(.+?)\.\s+each model's page",
    re.IGNORECASE | re.DOTALL,
)
_NA = frozenset({"", "n/a", "na", "none", "—", "–", "-", "---"})
_LIFECYCLE = {
    "active": AnthropicLifecycle.ACTIVE,
    "legacy": AnthropicLifecycle.LEGACY,
    "deprecated": AnthropicLifecycle.DEPRECATED,
    "retired": AnthropicLifecycle.RETIRED,
}


def parse_anthropic_sources(
    sources: Sequence[CapturedSource],
) -> tuple[AnthropicModelObservation, ...]:
    """Parse captured official Anthropic Markdown. Does not fetch the network."""
    if not sources:
        raise AnthropicParseError("Anthropic documentation sources are missing")
    observations: list[AnthropicModelObservation] = []
    for source in sources:
        try:
            observations.extend(_parse_source(source))
        except AnthropicParseError:
            raise
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise AnthropicParseError(
                f"Anthropic documentation {source.source_id} could not be parsed"
            ) from exc
    return tuple(observations)


def discover_model_page_identities(body: str) -> tuple[str, ...]:
    """Return docs slugs for individual model overview pages linked from Markdown."""
    slugs: list[str] = []
    seen: set[str] = set()
    for _label, target in markdown_links(body):
        slug = _page_slug_from_href(target)
        if slug is None or slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    return tuple(slugs)


def _parse_source(source: CapturedSource) -> tuple[AnthropicModelObservation, ...]:
    if source.media_type != "text/markdown":
        raise AnthropicParseError(
            f"Anthropic documentation {source.source_id} must be Markdown, not {source.media_type}"
        )
    if source.source_type is AnthropicSourceType.MODELS_INDEX:
        return _parse_overview(source)
    if source.source_type is AnthropicSourceType.DEPRECATIONS:
        return _parse_deprecations(source)
    return _parse_model_page(source)


def _source_ref(source: CapturedSource) -> AnthropicSourceRef:
    return AnthropicSourceRef(
        source_id=source.source_id,
        source_kind=source.source_kind,
        source_url=source.source_url,
        content_hash=source.content_hash,
        retrieved_at=source.retrieved_at,
        source_type=source.source_type,
    )


def _parse_overview(source: CapturedSource) -> tuple[AnthropicModelObservation, ...]:
    body = _strip_front_matter(source.body)
    sections = split_sections(body)
    compare = section_named(sections, "Compare models")
    if compare is None:
        raise AnthropicParseError("Anthropic models overview is missing a Compare models heading")
    table = _compare_table(compare.body)
    if table is None:
        raise AnthropicParseError("Anthropic models overview did not list any Claude API models")
    current_tool_use, current_input, current_output = _current_lineup_capabilities(compare.body)
    ref = _source_ref(source)
    observations: list[AnthropicModelObservation] = []
    for column in table:
        observations.extend(
            _overview_column_observations(
                column,
                source=ref,
                tool_calling=current_tool_use,
                input_modalities=current_input,
                output_modalities=current_output,
            )
        )
    if not any(item.listed_on_index for item in observations):
        raise AnthropicParseError("Anthropic models overview did not list any Claude API models")
    return tuple(observations)


def _compare_table(text: str) -> tuple[dict[str, str], ...] | None:
    for table in pipe_tables(text):
        if not table:
            continue
        feature_header = next(
            (key for key in table[0] if normalize_heading(_visible_text(key)) == "feature"),
            None,
        )
        if feature_header is None:
            continue
        model_headers = [key for key in table[0] if key != feature_header]
        if not model_headers:
            continue
        rows_by_feature: dict[str, dict[str, str]] = {}
        for row in table:
            feature = normalize_heading(_visible_text(row.get(feature_header, "")))
            if not feature:
                continue
            rows_by_feature[feature] = row
        if "claude api id" not in rows_by_feature:
            continue
        columns: list[dict[str, str]] = []
        for header in model_headers:
            column = {
                "display_name": _visible_text(header),
                "header": header,
            }
            for feature, row in rows_by_feature.items():
                column[feature] = row.get(header, "")
            columns.append(column)
        if columns:
            return tuple(columns)
    return None


def _overview_column_observations(
    column: dict[str, str],
    *,
    source: AnthropicSourceRef,
    tool_calling: bool | None,
    input_modalities: tuple[str, ...] | None,
    output_modalities: tuple[str, ...] | None,
) -> tuple[AnthropicModelObservation, ...]:
    model_id = _claude_api_id(column.get("claude api id", ""))
    if model_id is None:
        return ()
    alias_id = _claude_api_id(column.get("claude api alias", ""))
    display_name = column.get("display_name") or None
    family = _family_from_name(display_name)
    context_window = _parse_tokens(column.get("context window", ""))
    max_output = _parse_tokens(column.get("max output", ""))
    aliases: tuple[str, ...] | None = None
    extras: list[AnthropicModelObservation] = []
    if alias_id is not None and alias_id != model_id:
        aliases = (alias_id,)
        extras.append(
            AnthropicModelObservation(
                provider_model_id=alias_id,
                source=source,
                identity_kind=IdentityKind.ALIAS,
                display_name=display_name,
                resolves_to=model_id,
                family=family,
            )
        )
    extras.insert(
        0,
        AnthropicModelObservation(
            provider_model_id=model_id,
            source=source,
            display_name=display_name,
            listed_on_index=True,
            aliases=aliases,
            family=family,
            context_window_tokens=context_window,
            max_output_tokens=max_output,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            tool_calling=tool_calling,
        ),
    )
    return tuple(extras)


def _current_lineup_capabilities(
    text: str,
) -> tuple[bool | None, tuple[str, ...] | None, tuple[str, ...] | None]:
    match = _CURRENT_CAPABILITIES.search(" ".join(text.split()))
    if match is None:
        return None, None, None
    claim = match.group(1).lower()
    tool_calling = True if "tool use" in claim else None
    input_modalities: tuple[str, ...] | None = None
    output_modalities: tuple[str, ...] | None = None
    if "text and image input" in claim or "text and images" in claim:
        input_modalities = ("text", "image")
    elif "text input" in claim:
        input_modalities = ("text",)
    if "text output" in claim:
        output_modalities = ("text",)
    return tool_calling, input_modalities, output_modalities


def _parse_deprecations(source: CapturedSource) -> tuple[AnthropicModelObservation, ...]:
    body = _strip_front_matter(source.body)
    sections = split_sections(body)
    if not has_heading(sections, "Model status"):
        raise AnthropicParseError("Anthropic deprecations page is missing a Model status heading")
    ref = _source_ref(source)
    current: list[AnthropicModelObservation] = []
    past: list[AnthropicModelObservation] = []
    status_section = section_named(sections, "Model status")
    status_text = status_section.body if status_section is not None else body
    for table in pipe_tables(status_text):
        if not _is_status_table(table):
            continue
        current.extend(_status_table_observations(table, source=ref))
    current_ids = {item.provider_model_id for item in current}
    overview = section_named(sections, "Overview")
    if overview is not None:
        for match in _PROSE_DEPRECATED.finditer(overview.body):
            model_id = _claude_api_id(match.group(1))
            if model_id is None or model_id in current_ids:
                continue
            current.append(
                AnthropicModelObservation(
                    provider_model_id=model_id,
                    source=ref,
                    lifecycle=AnthropicLifecycle.DEPRECATED,
                    deprecation_listing="current",
                )
            )
            current_ids.add(model_id)
    history_text = _section_with_children(sections, "Deprecation history")
    for table in pipe_tables(history_text):
        if not _is_history_table(table):
            continue
        past.extend(_history_table_observations(table, source=ref))
    if not current and not past:
        raise AnthropicParseError("Anthropic deprecations page did not list any models")
    replacements: dict[str, list[str]] = {}
    for item in past:
        if not item.replacements:
            continue
        bucket = replacements.setdefault(item.provider_model_id, [])
        for replacement in item.replacements:
            if replacement not in bucket:
                bucket.append(replacement)
    merged: list[AnthropicModelObservation] = []
    for observation in current:
        found = tuple(replacements.get(observation.provider_model_id, ()))
        if found:
            observation = replace(observation, replacements=found)
        merged.append(observation)
    past_only = [item for item in past if item.provider_model_id not in current_ids]
    return tuple(merged + past_only)


def _section_with_children(sections: tuple, title: str) -> str:
    wanted = normalize_heading(title)
    collecting = False
    level: int | None = None
    parts: list[str] = []
    for section in sections:
        if not collecting:
            if normalize_heading(section.title) == wanted:
                collecting = True
                level = section.level
                parts.append(section.body)
            continue
        if level is not None and section.level <= level:
            break
        parts.append(section.body)
    return "\n".join(parts)


def _is_status_table(table: tuple[dict[str, str], ...]) -> bool:
    if not table:
        return False
    keys = {normalize_heading(_visible_text(key)) for key in table[0]}
    return "api model name" in keys and "current state" in keys


def _is_history_table(table: tuple[dict[str, str], ...]) -> bool:
    if not table:
        return False
    keys = {normalize_heading(_visible_text(key)) for key in table[0]}
    return "deprecated model" in keys and "recommended replacement" in keys


def _status_table_observations(
    table: tuple[dict[str, str], ...],
    *,
    source: AnthropicSourceRef,
) -> tuple[AnthropicModelObservation, ...]:
    observations: list[AnthropicModelObservation] = []
    for row in table:
        cells = {normalize_heading(_visible_text(key)): value for key, value in row.items()}
        model_id = _claude_api_id(cells.get("api model name", ""))
        if model_id is None:
            continue
        lifecycle = _lifecycle_state(cells.get("current state", ""))
        deprecated_at = _parse_date(cells.get("deprecated", ""))
        retirement_at = _retirement_date(cells.get("tentative retirement date", ""))
        api_access = None
        if lifecycle is AnthropicLifecycle.RETIRED:
            api_access = False
        elif lifecycle is AnthropicLifecycle.DEPRECATED:
            api_access = True
        observations.append(
            AnthropicModelObservation(
                provider_model_id=model_id,
                source=source,
                lifecycle=lifecycle,
                deprecation_listing="current",
                deprecated_at=deprecated_at,
                retirement_at=retirement_at,
                api_access=api_access,
            )
        )
    return tuple(observations)


def _history_table_observations(
    table: tuple[dict[str, str], ...],
    *,
    source: AnthropicSourceRef,
) -> tuple[AnthropicModelObservation, ...]:
    observations: list[AnthropicModelObservation] = []
    for row in table:
        cells = {normalize_heading(_visible_text(key)): value for key, value in row.items()}
        model_id = _claude_api_id(cells.get("deprecated model", ""))
        if model_id is None:
            continue
        replacement = _claude_api_id(cells.get("recommended replacement", ""))
        retirement_at = _retirement_date(cells.get("retirement date", ""))
        observations.append(
            AnthropicModelObservation(
                provider_model_id=model_id,
                source=source,
                lifecycle=AnthropicLifecycle.RETIRED,
                deprecation_listing="past",
                retirement_at=retirement_at,
                replacements=(replacement,) if replacement is not None else None,
                api_access=False,
            )
        )
    return tuple(observations)


def _parse_model_page(source: CapturedSource) -> tuple[AnthropicModelObservation, ...]:
    raw_body = source.body
    body = _strip_front_matter(raw_body)
    sections = split_sections(body)
    display_name = _front_matter_title(raw_body)
    if not display_name:
        for section in sections:
            if section.title:
                display_name = _visible_text(section.title)
                break
    model_id = _model_id_from_page(body)
    if model_id is None:
        raise AnthropicParseError("Anthropic model page is missing a Model ID")
    if not display_name:
        display_name = model_id
    capabilities = _feature_value_table(sections, "Capabilities")
    availability = _feature_value_table(sections, "Availability")
    model_ids = _platform_id_table(sections)
    alias_id = _claude_api_id(model_ids.get("claude api alias", ""))
    context_window = _parse_tokens(
        capabilities.get("context window") or labeled_value(body, "Context window") or ""
    )
    max_output = _parse_tokens(
        capabilities.get("max output") or labeled_value(body, "Max output") or ""
    )
    io = capabilities.get("input → output") or capabilities.get("input -> output") or ""
    input_modalities, output_modalities = _modalities_from_io(io)
    extras: list[AnthropicModelObservation] = []
    aliases: tuple[str, ...] | None = None
    family = _family_from_name(display_name)
    if alias_id is not None and alias_id != model_id:
        aliases = (alias_id,)
        extras.append(
            AnthropicModelObservation(
                provider_model_id=alias_id,
                source=_source_ref(source),
                identity_kind=IdentityKind.ALIAS,
                display_name=display_name,
                resolves_to=model_id,
                family=family,
            )
        )
    extras.insert(
        0,
        AnthropicModelObservation(
            provider_model_id=model_id,
            source=_source_ref(source),
            display_name=display_name,
            aliases=aliases,
            family=family,
            lifecycle=_lifecycle_state(availability.get("status", "")),
            context_window_tokens=context_window,
            max_output_tokens=max_output,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            tool_calling=_tool_calling_from_capabilities(capabilities),
            structured_output=_structured_output_from_capabilities(capabilities),
            api_access=_api_access_from_availability(availability),
        ),
    )
    return tuple(extras)


def _model_id_from_page(text: str) -> str | None:
    labeled = labeled_value(text, "Model ID")
    if labeled is not None:
        found = _claude_api_id(labeled)
        if found is not None:
            return found
    return _claude_api_id(_platform_id_table(split_sections(text)).get("claude api", ""))


def _feature_value_table(sections: tuple, title: str) -> dict[str, str]:
    section = section_named(sections, title)
    if section is None:
        return {}
    values: dict[str, str] = {}
    for table in pipe_tables(section.body):
        if not table:
            continue
        keys = {normalize_heading(_visible_text(key)) for key in table[0]}
        if "feature" not in keys or "value" not in keys:
            continue
        feature_key = next(
            key for key in table[0] if normalize_heading(_visible_text(key)) == "feature"
        )
        value_key = next(
            key for key in table[0] if normalize_heading(_visible_text(key)) == "value"
        )
        for row in table:
            feature = normalize_heading(_visible_text(row.get(feature_key, "")))
            if feature:
                values[feature] = row.get(value_key, "")
    return values


def _platform_id_table(sections: tuple) -> dict[str, str]:
    section = section_named(sections, "Model IDs")
    if section is None:
        return {}
    values: dict[str, str] = {}
    for table in pipe_tables(section.body):
        if not table:
            continue
        keys = {normalize_heading(_visible_text(key)) for key in table[0]}
        if "platform" not in keys or "model id" not in keys:
            continue
        platform_key = next(
            key for key in table[0] if normalize_heading(_visible_text(key)) == "platform"
        )
        id_key = next(
            key for key in table[0] if normalize_heading(_visible_text(key)) == "model id"
        )
        for row in table:
            platform = normalize_heading(_visible_text(row.get(platform_key, "")))
            if platform:
                values[platform] = row.get(id_key, "")
    return values


def _modalities_from_io(raw: str) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    text = _visible_text(raw)
    if not text:
        return None, None
    if "→" in text:
        left, right = text.split("→", 1)
    elif "->" in text:
        left, right = text.split("->", 1)
    else:
        return None, None
    return _modality_tokens(left), _modality_tokens(right)


def _modality_tokens(raw: str) -> tuple[str, ...] | None:
    text = raw.lower()
    found: list[str] = []
    if "text" in text:
        found.append("text")
    if "image" in text:
        found.append("image")
    if "audio" in text:
        found.append("audio")
    if "video" in text:
        found.append("video")
    return tuple(found) or None


def _tool_calling_from_capabilities(capabilities: dict[str, str]) -> bool | None:
    for key, value in capabilities.items():
        if "tool" not in key:
            continue
        lowered = _visible_text(value).lower()
        if lowered in {"yes", "supported", "true"}:
            return True
        if lowered in {"no", "not supported", "false"}:
            return False
        if "tool use" in lowered or "tool calling" in lowered:
            return True
    return None


def _structured_output_from_capabilities(capabilities: dict[str, str]) -> bool | None:
    for key, value in capabilities.items():
        if "structured output" not in key:
            continue
        lowered = _visible_text(value).lower()
        if lowered in {"yes", "supported", "true"}:
            return True
        if lowered in {"no", "not supported", "false"}:
            return False
    return None


def _api_access_from_availability(availability: dict[str, str]) -> bool | None:
    platforms = availability.get("platforms")
    if platforms is None or not _visible_text(platforms):
        return None
    text = _visible_text(platforms).lower()
    return "claude api" in text


def _lifecycle_state(raw: str) -> AnthropicLifecycle | None:
    text = _visible_text(raw).lower()
    if not text or _is_na(text):
        return None
    token = text.split("(")[0].strip()
    return _LIFECYCLE.get(token)


def _family_from_name(name: str | None) -> str | None:
    if not name:
        return None
    lowered = name.strip().lower()
    for prefix, family in _FAMILY_PREFIXES:
        if lowered.startswith(prefix) or prefix in lowered:
            return family
    return None


def _claude_api_id(raw: str) -> str | None:
    text = raw.strip()
    if not text:
        return None
    candidates = list(code_spans(text))
    if not candidates:
        candidates = [_visible_text(text)]
    for candidate in candidates:
        token = candidate.strip().strip("*")
        if not token or "@" in token or token.startswith("anthropic."):
            continue
        if _MODEL_ID.fullmatch(token):
            return token
    return None


def _parse_tokens(raw: str) -> int | None:
    text = _visible_text(raw).split("(")[0].strip()
    if not text or _is_na(text):
        return None
    compact = text.replace(" ", "")
    match = _TOKEN.fullmatch(compact) or _TOKEN.fullmatch(text)
    if match is None:
        return None
    number = match.group("num").replace(",", "")
    try:
        value = float(number)
    except ValueError:
        return None
    unit = (match.group("unit") or "").lower()
    if unit == "k":
        value *= 1_000
    elif unit == "m":
        value *= 1_000_000
    tokens = int(value)
    return tokens if tokens > 0 else None


def _retirement_date(raw: str) -> str | None:
    text = _visible_text(raw)
    if not text or _is_na(text) or _COMMITMENT.search(text):
        return None
    return _parse_date(text)


def _parse_date(raw: str) -> str | None:
    text = _visible_text(raw)
    if not text or _is_na(text) or _COMMITMENT.search(text):
        return None
    text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return None
    match = _ENGLISH_DATE.fullmatch(text)
    if match is None:
        return None
    month = _MONTHS.get(match.group("month").lower())
    if month is None:
        return None
    try:
        return date(int(match.group("year")), month, int(match.group("day"))).isoformat()
    except ValueError:
        return None


def _is_na(text: str) -> bool:
    return text.strip().lower() in _NA


def _visible_text(raw: str) -> str:
    text = _LINK.sub(r"\1", raw or "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return " ".join(text.split())


def _strip_front_matter(body: str) -> str:
    if not body.startswith("---"):
        return body
    rest = body[3:]
    end = rest.find("\n---")
    if end == -1:
        return body
    return rest[end + 4 :].lstrip("\n")


def _front_matter_title(body: str) -> str | None:
    if not body.startswith("---"):
        return None
    rest = body[3:]
    end = rest.find("\n---")
    if end == -1:
        return None
    for line in rest[:end].splitlines():
        if not line.lower().startswith("title:"):
            continue
        title = line.split(":", 1)[1].strip().strip('"').strip("'")
        return title or None
    return None


def _page_slug_from_href(href: str) -> str | None:
    parsed = urlparse(href.strip())
    path = parsed.path or href.strip()
    match = _MODEL_PAGE_PATH.fullmatch(path)
    if match is None:
        return None
    slug = match.group(1).lower()
    if slug in _SKIP_PAGE_SLUGS:
        return None
    return slug
