"""Parse captured Google Gemini API Markdown into provider observations.

Gemini API / AI Studio only. Vertex AI and Google Cloud resource IDs are ignored.

Lifecycle:
- Stable / Preview / Latest / Experimental are version channels, not states
- index "(Shut down)" and page "has been shut down" → retired
- index "(Deprecated)" and "deprecated" warnings without shutdown → deprecated
- "Previous models" grouping is not itself a lifecycle state
- deprecations table shutdown dates are earliest-possible and are not retirement_at

Boolean capabilities:
- omitted / unlabeled → not extracted
- Function calling / Structured outputs Supported → True
- Not supported → False
- family-level or latest-model pages are not copied onto other IDs
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from urllib.parse import urlparse

from samyak.model.identity import IdentityKind
from samyak.model.markdown import (
    code_spans,
    markdown_links,
    normalize_heading,
    pipe_tables,
    split_sections,
    strip_trailing_markdown_footnote,
)
from samyak.model.providers.google.errors import GoogleParseError
from samyak.model.providers.google.observations import (
    GoogleLifecycle,
    GoogleModelObservation,
    GoogleSourceRef,
)
from samyak.model.providers.google.sources import CapturedSource, GoogleSourceType

_MODEL_ID = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)+$")
_MODEL_PAGE_PATH = re.compile(
    r"^/gemini-api/docs/models/([^/]+?)(?:\.md(?:\.txt)?)?$",
    re.IGNORECASE,
)
_SKIP_PAGE_SLUGS = frozenset({"models"})
_SKIP_INDEX_HEADINGS = frozenset(
    {
        "model version name patterns",
        "model deprecations",
        "stable",
        "preview",
        "latest",
        "experimental",
        "documentation",
    }
)
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_TOKEN = re.compile(
    r"(?P<num>[\d,]+)\s*(?P<unit>[kKmM])?(?:\s*tokens?)?\b",
)
_EXPLICIT_CONTEXT_WINDOW = re.compile(
    r"(?<!\binput )(?<!\boutput )"
    r"context\s+(?:window|length|size)\s*:?\s*"
    r"(?P<num>[\d,]+)\s*(?P<unit>[kKmM])?(?:\s*tokens?)?\b",
    re.IGNORECASE,
)
_CONTEXT_WINDOW_LABELS = frozenset({"context window", "context length", "context size"})
_MODEL_ID_TABLE_HEADERS = frozenset({"model id", "model identifier"})
_CONTEXT_WINDOW_TABLE_HEADERS = frozenset(
    {
        "context window",
        "context window (in / out)",
        "context window (in/out)",
        "context window (input / output)",
        "context window (input/output)",
        "context length",
        "context size",
    }
)
_LIMITS_PROPERTIES = ("token limits", "limits")
_MAX_INPUT_LABELS = ("input token limit", "input tokens", "input context window")
_MAX_OUTPUT_LABELS = ("output token limit", "output tokens")
_ENGLISH_DATE = re.compile(
    r"(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_HAS_BEEN_SHUT_DOWN = re.compile(
    r"has been shut down\s+" + _ENGLISH_DATE.pattern,
    re.IGNORECASE,
)
_ALIAS_FOR = re.compile(
    r"`([^`]+)`\s+is an alias for\s+`([^`]+)`",
    re.IGNORECASE,
)
_SHUT_DOWN_VERSION = re.compile(
    r"shut\s*down[^`]*`([^`]+)`",
    re.IGNORECASE,
)
_BOLD_FIELD = re.compile(
    r"\*\*(?:\[(?P<link>[^\]]+)\]\([^)]+\)|(?P<label>[^*]+))\*\*\s*"
    r"(?P<value>.*?)(?=\s*\*\*|\s*$)",
    re.DOTALL,
)
_TRUE_STATUS = frozenset(
    {
        "supported",
        "yes",
        "true",
        "experimental",
    }
)
_FALSE_STATUS = frozenset({"not supported", "no", "false", "unsupported"})
_MODALITY_WORDS = {
    "text": "text",
    "image": "image",
    "images": "image",
    "audio": "audio",
    "video": "video",
}


def parse_google_sources(
    sources: Sequence[CapturedSource],
) -> tuple[GoogleModelObservation, ...]:
    """Parse captured official Gemini API Markdown. Does not fetch the network."""
    if not sources:
        raise GoogleParseError("Google documentation sources are missing")
    observations: list[GoogleModelObservation] = []
    for source in sources:
        try:
            observations.extend(_parse_source(source))
        except GoogleParseError:
            raise
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise GoogleParseError(
                f"Google documentation {source.source_id} could not be parsed"
            ) from exc
    return tuple(observations)


def discover_model_page_identities(body: str) -> tuple[str, ...]:
    """Return docs slugs for individual Gemini API model pages linked from Markdown."""
    slugs: list[str] = []
    seen: set[str] = set()
    for _label, target in markdown_links(body):
        slug = _page_slug_from_href(target)
        if slug is None or slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    return tuple(slugs)


def _parse_source(source: CapturedSource) -> tuple[GoogleModelObservation, ...]:
    if source.media_type != "text/markdown":
        raise GoogleParseError(
            f"Google documentation {source.source_id} must be Markdown, not {source.media_type}"
        )
    if source.source_type is GoogleSourceType.MODELS_INDEX:
        return _parse_index(source)
    if source.source_type is GoogleSourceType.DEPRECATIONS:
        return _parse_deprecations(source)
    if source.source_type is GoogleSourceType.GEMINI_3:
        return _parse_gemini_3(source)
    return _parse_model_page(source)


def _source_ref(source: CapturedSource) -> GoogleSourceRef:
    return GoogleSourceRef(
        source_id=source.source_id,
        source_kind=source.source_kind,
        source_url=source.source_url,
        content_hash=source.content_hash,
        retrieved_at=source.retrieved_at,
        source_type=source.source_type,
    )


def _parse_index(source: CapturedSource) -> tuple[GoogleModelObservation, ...]:
    sections = split_sections(source.body)
    if (
        not any(
            "gemini" in normalize_heading(section.title)
            or "model" in normalize_heading(section.title)
            for section in sections
            if section.title
        )
        and "Endpoint" not in source.body
    ):
        raise GoogleParseError("Google models index is missing a models catalogue")
    ref = _source_ref(source)
    seen: dict[str, GoogleModelObservation] = {}
    for section in sections:
        heading = normalize_heading(section.title)
        if heading in _SKIP_INDEX_HEADINGS:
            continue
        previous = heading == "previous models"
        for table in pipe_tables(section.body):
            for observation in _index_table_observations(
                table, source=ref, previous_models=previous
            ):
                if observation.provider_model_id not in seen:
                    seen[observation.provider_model_id] = observation
    if not seen:
        raise GoogleParseError("Google models index did not list any Gemini API models")
    return tuple(seen.values())


def _index_table_observations(
    table: tuple[dict[str, str], ...],
    *,
    source: GoogleSourceRef,
    previous_models: bool,
) -> tuple[GoogleModelObservation, ...]:
    if not table:
        return ()
    columns = {_header_key(key): key for key in table[0]}
    endpoint_key = columns.get("endpoint")
    model_key = columns.get("model")
    if endpoint_key is None and model_key is None:
        return ()
    observations: list[GoogleModelObservation] = []
    for row in table:
        model_cell = row.get(model_key, "") if model_key is not None else ""
        endpoint_cell = row.get(endpoint_key, "") if endpoint_key is not None else ""
        model_ids = _model_ids_from_cell(endpoint_cell) or _model_ids_from_cell(model_cell)
        if not model_ids:
            continue
        display_name = _index_display_name(model_cell) or None
        lifecycle = _index_lifecycle(model_cell)
        for model_id in model_ids:
            observations.append(
                GoogleModelObservation(
                    provider_model_id=model_id,
                    source=source,
                    display_name=display_name,
                    listed_on_index=not previous_models,
                    lifecycle=lifecycle,
                    api_access=_index_api_access(lifecycle, previous_models=previous_models),
                )
            )
    return tuple(observations)


def _index_display_name(raw: str) -> str:
    text = _visible_text(raw)
    text = re.sub(r"\((?:shut down|deprecated)\)", "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip(" -")


def _index_lifecycle(model_cell: str) -> GoogleLifecycle | None:
    text = _visible_text(model_cell).lower()
    if "shut down" in text:
        return GoogleLifecycle.RETIRED
    if "deprecated" in text:
        return GoogleLifecycle.DEPRECATED
    return None


def _index_api_access(lifecycle: GoogleLifecycle | None, *, previous_models: bool) -> bool | None:
    if lifecycle is GoogleLifecycle.RETIRED:
        return False
    if previous_models:
        return None
    return True


def _parse_gemini_3(source: CapturedSource) -> tuple[GoogleModelObservation, ...]:
    """Parse Context Window (In / Out) tables from the Gemini 3 developer guide.

    Google's token documentation defines the context window as the combined
    input and output limits. ``1M / 64k`` is therefore the documented input
    limit and output limit, not a combined context size of ``1M``. Combined
    context is derived later, only for applicable generative models.

    Thinking-level tables and family-level prose are ignored.
    """
    ref = _source_ref(source)
    observations: list[GoogleModelObservation] = []
    found_table = False
    for table in pipe_tables(source.body):
        if not table:
            continue
        model_header = _table_header(table[0], _MODEL_ID_TABLE_HEADERS)
        context_header = _table_header(table[0], _CONTEXT_WINDOW_TABLE_HEADERS)
        if model_header is None or context_header is None:
            continue
        found_table = True
        for row in table:
            model_ids = _model_ids_from_cell(row.get(model_header, ""))
            limits = _in_out_token_limits(row.get(context_header, ""))
            if limits is None:
                continue
            max_input, max_output = limits
            for model_id in model_ids:
                observations.append(
                    GoogleModelObservation(
                        provider_model_id=model_id,
                        source=ref,
                        max_input_tokens=max_input,
                        max_output_tokens=max_output,
                    )
                )
    if not found_table:
        raise GoogleParseError("Google Gemini 3 documentation is missing a context-window table")
    if not observations:
        raise GoogleParseError("Google Gemini 3 documentation did not list any context windows")
    return tuple(observations)


def _table_header(row: dict[str, str], names: frozenset[str]) -> str | None:
    for header in row:
        if _table_header_key(header) in names:
            return header
    return None


def _table_header_key(header: str) -> str:
    return _header_key(strip_trailing_markdown_footnote(header)).rstrip("*").strip()


def _in_out_token_limits(raw: str) -> tuple[int, int] | None:
    """Return the In and Out sides of a Context Window (In / Out) cell.

    Both sides must be explicit. The values are not summed here.
    """
    text = _visible_text(raw)
    if not text or "/" not in text:
        return None
    left, right = text.split("/", 1)
    max_input = _parse_tokens(left.strip())
    max_output = _parse_tokens(right.strip())
    if max_input is None or max_output is None:
        return None
    return max_input, max_output


def _parse_deprecations(source: CapturedSource) -> tuple[GoogleModelObservation, ...]:
    sections = split_sections(source.body)
    if "earliest possible" not in source.body.lower() and "deprecation" not in source.body.lower():
        raise GoogleParseError("Google deprecations page is missing a deprecation schedule")
    ref = _source_ref(source)
    observations: list[GoogleModelObservation] = []
    listing = "schedule"
    for section in sections:
        heading = normalize_heading(section.title)
        if heading == "deprecated models" or heading.startswith("current"):
            listing = "current"
        elif heading.startswith("upcoming"):
            listing = "upcoming"
        elif heading.startswith("past"):
            listing = "past"
        elif heading.endswith("models") or heading == "":
            listing = "schedule"
        for table in pipe_tables(section.body):
            observations.extend(_deprecation_table_observations(table, source=ref, listing=listing))
    if not observations:
        raise GoogleParseError("Google deprecations page did not list any models")
    return tuple(observations)


def _deprecation_table_observations(
    table: tuple[dict[str, str], ...],
    *,
    source: GoogleSourceRef,
    listing: str,
) -> tuple[GoogleModelObservation, ...]:
    observations: list[GoogleModelObservation] = []
    for row in table:
        cells = {_header_key(key): value for key, value in row.items()}
        model_ids = _model_ids_from_cell(cells.get("model", ""))
        if not model_ids:
            continue
        replacements = _gemini_api_ids_only(
            _model_ids_from_cell(cells.get("recommended replacement", ""))
        )
        for model_id in model_ids:
            observations.append(
                GoogleModelObservation(
                    provider_model_id=model_id,
                    source=source,
                    deprecation_listing=listing,
                    replacements=replacements or None,
                )
            )
    return tuple(observations)


def _parse_model_page(source: CapturedSource) -> tuple[GoogleModelObservation, ...]:
    body = source.body.replace("<br />", "\n").replace("<br/>", "\n")
    sections = split_sections(body)
    properties = _property_fields(body)
    model_ids = _model_ids_from_cell(properties.get("model code", ""))
    if not model_ids:
        heading_id = _heading_model_id(sections)
        if heading_id is not None:
            model_ids = (heading_id,)
    if not model_ids:
        raise GoogleParseError("Google model page is missing a Model code")
    display_name = _page_display_name(sections)
    inputs, outputs = _modalities_from_data_types(properties.get("supported data types", ""))
    tokens = properties.get("token limits", "")
    max_input = _token_limit(tokens, _MAX_INPUT_LABELS)
    max_output = _token_limit(tokens, _MAX_OUTPUT_LABELS)
    context_window = _context_window_tokens(properties)
    capabilities = properties.get("capabilities", "")
    tool_calling = _capability_flag(capabilities, ("function calling",))
    structured_output = _capability_flag(capabilities, ("structured outputs", "structured output"))
    warning_lifecycle, warning_retirement = _warning_lifecycle(body)
    extras: list[GoogleModelObservation] = []
    ref = _source_ref(source)
    primary = model_ids[0]
    alias_target, alias_ids = _alias_targets(body, primary)
    identity_kind = IdentityKind.ALIAS if alias_target is not None else IdentityKind.CANONICAL
    page_api_access = warning_lifecycle is not GoogleLifecycle.RETIRED
    for model_id in model_ids:
        extras.append(
            GoogleModelObservation(
                provider_model_id=model_id,
                source=ref,
                identity_kind=identity_kind if model_id == primary else IdentityKind.CANONICAL,
                display_name=display_name,
                aliases=alias_ids or None if model_id == primary else None,
                resolves_to=alias_target if model_id == primary else None,
                lifecycle=warning_lifecycle,
                retirement_at=warning_retirement,
                context_window_tokens=context_window,
                max_input_tokens=max_input,
                max_output_tokens=max_output,
                input_modalities=inputs,
                output_modalities=outputs,
                tool_calling=tool_calling,
                structured_output=structured_output,
                api_access=page_api_access,
            )
        )
    for alias_id in alias_ids:
        extras.append(
            GoogleModelObservation(
                provider_model_id=alias_id,
                source=ref,
                identity_kind=IdentityKind.ALIAS,
                resolves_to=primary,
            )
        )
    if alias_target is not None:
        extras.append(
            GoogleModelObservation(
                provider_model_id=alias_target,
                source=ref,
                aliases=(primary,),
            )
        )
    for shut_down_id in _shut_down_version_ids(properties.get("versions", "") + "\n" + body):
        if shut_down_id in {item.provider_model_id for item in extras}:
            continue
        extras.append(
            GoogleModelObservation(
                provider_model_id=shut_down_id,
                source=ref,
                lifecycle=GoogleLifecycle.RETIRED,
                api_access=False,
            )
        )
    return tuple(extras)


def _property_fields(body: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for table in pipe_tables(body):
        if not table:
            continue
        columns = {_header_key(key): key for key in table[0]}
        property_key = columns.get("property")
        description_key = columns.get("description")
        if property_key is None or description_key is None:
            continue
        for row in table:
            name = normalize_heading(
                strip_trailing_markdown_footnote(_visible_text(row.get(property_key, "")))
            )
            if name:
                values[name] = row.get(description_key, "")
    return values


def _page_display_name(sections: tuple) -> str | None:
    for section in sections:
        title = _visible_text(section.title)
        if not title:
            continue
        lowered = normalize_heading(title)
        if lowered in {"documentation", "feature support and limitations", "supported languages"}:
            continue
        if _MODEL_ID.fullmatch(title.strip("`")):
            continue
        return title
    return None


def _heading_model_id(sections: tuple) -> str | None:
    for section in sections:
        candidate = _visible_text(section.title).strip("`")
        if _MODEL_ID.fullmatch(candidate):
            return candidate
    return None


def _modalities_from_data_types(
    raw: str,
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    fields = _bold_fields(raw)
    if not fields:
        return _modality_tokens(raw), None
    inputs = None
    outputs = None
    for label, value in fields:
        if label in {"inputs", "input"}:
            inputs = _modality_tokens(value)
        elif label in {"output", "outputs"}:
            outputs = _modality_tokens(value)
    return inputs, outputs


def _token_limit(raw: str, labels: tuple[str, ...]) -> int | None:
    fields = _bold_fields(raw)
    for label, value in fields:
        if label in labels:
            return _parse_tokens(value)
    return None


def _context_window_tokens(properties: dict[str, str]) -> int | None:
    """Return an explicit context-window size. Input/output limits are ignored."""
    for key in _CONTEXT_WINDOW_LABELS:
        raw = properties.get(key)
        if not raw:
            continue
        found = _labeled_or_explicit_context(raw)
        if found is None:
            found = _parse_tokens(raw)
        if found is not None:
            return found
    for key in _LIMITS_PROPERTIES:
        raw = properties.get(key, "")
        if not raw:
            continue
        found = _labeled_or_explicit_context(raw)
        if found is not None:
            return found
    return None


def _labeled_or_explicit_context(raw: str) -> int | None:
    found = _token_limit(raw, tuple(_CONTEXT_WINDOW_LABELS))
    if found is not None:
        return found
    match = _EXPLICIT_CONTEXT_WINDOW.search(_visible_text(raw))
    if match is None:
        return None
    return _tokens_from_parts(match.group("num"), match.group("unit"))


def _capability_flag(raw: str, names: tuple[str, ...]) -> bool | None:
    for label, value in _bold_fields(raw):
        if label not in names:
            continue
        return _bool_status(value)
    return None


def _bool_status(raw: str) -> bool | None:
    text = _visible_text(raw).lower()
    if not text:
        return None
    token = text.split(".", 1)[0]
    token = re.sub(r"\s*\(.*$", "", token).strip()
    if token in _TRUE_STATUS or token.startswith("supported"):
        return True
    if token in _FALSE_STATUS or token.startswith("not supported"):
        return False
    return None


def _warning_lifecycle(body: str) -> tuple[GoogleLifecycle | None, str | None]:
    match = _HAS_BEEN_SHUT_DOWN.search(body)
    if match is not None:
        retirement = _english_date(match)
        return GoogleLifecycle.RETIRED, retirement
    lowered = body.lower()
    if "are deprecated" in lowered or "is deprecated" in lowered:
        return GoogleLifecycle.DEPRECATED, None
    return None, None


def _english_date(match: re.Match[str]) -> str | None:
    month = _MONTHS.get(match.group("month").lower())
    if month is None:
        return None
    try:
        return date(int(match.group("year")), month, int(match.group("day"))).isoformat()
    except ValueError:
        return None


def _alias_targets(text: str, page_model_id: str) -> tuple[str | None, tuple[str, ...]]:
    aliases: list[str] = []
    resolves_to: str | None = None
    for match in _ALIAS_FOR.finditer(text):
        alias_id = _require_model_id(match.group(1), required=False)
        target_id = _require_model_id(match.group(2), required=False)
        if alias_id is None or target_id is None:
            continue
        if alias_id == page_model_id:
            resolves_to = target_id
        elif target_id == page_model_id:
            aliases.append(alias_id)
    return resolves_to, tuple(dict.fromkeys(aliases))


def _shut_down_version_ids(text: str) -> tuple[str, ...]:
    ids: list[str] = []
    for match in _SHUT_DOWN_VERSION.finditer(text):
        model_id = _require_model_id(match.group(1), required=False)
        if model_id is not None:
            ids.append(model_id)
    return tuple(dict.fromkeys(ids))


def _bold_fields(raw: str) -> tuple[tuple[str, str], ...]:
    fields: list[tuple[str, str]] = []
    for match in _BOLD_FIELD.finditer(raw.replace("\n", " ")):
        label = match.group("link") or match.group("label") or ""
        value = match.group("value") or ""
        fields.append((normalize_heading(label), value.strip()))
    return tuple(fields)


def _modality_tokens(raw: str) -> tuple[str, ...] | None:
    text = _visible_text(raw).lower()
    if not text:
        return None
    found: list[str] = []
    for token in re.split(r"[,/]| and ", text):
        mapped = _MODALITY_WORDS.get(token.strip().strip("."))
        if mapped is not None and mapped not in found:
            found.append(mapped)
    return tuple(found) or None


def _parse_tokens(raw: str) -> int | None:
    text = _visible_text(raw)
    if not text:
        return None
    match = _TOKEN.search(text)
    if match is None:
        return None
    return _tokens_from_parts(match.group("num"), match.group("unit"))


def _tokens_from_parts(num: str, unit: str | None) -> int | None:
    try:
        value = float(num.replace(",", ""))
    except ValueError:
        return None
    suffix = (unit or "").lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    tokens = int(value)
    return tokens if tokens > 0 else None


def _model_ids_from_cell(raw: str) -> tuple[str, ...]:
    ids: list[str] = []
    for token in code_spans(raw):
        model_id = _require_model_id(token, required=False)
        if model_id is not None:
            ids.append(model_id)
    if not ids:
        for token in re.split(r"\s+", _visible_text(raw)):
            model_id = _require_model_id(token.strip("`"), required=False)
            if model_id is not None:
                ids.append(model_id)
    return tuple(dict.fromkeys(ids))


def _gemini_api_ids_only(model_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item for item in model_ids if not item.startswith("anthropic."))


def _require_model_id(value: str, *, required: bool) -> str | None:
    token = value.strip().strip("*").strip()
    if not token:
        if required:
            raise GoogleParseError("Google model page is missing a Model code")
        return None
    if "/" in token or "@" in token or token.startswith("models/"):
        if required:
            raise GoogleParseError(
                f"Google model identifier {token!r} is not a Gemini API model id"
            )
        return None
    if not _MODEL_ID.fullmatch(token):
        if required:
            raise GoogleParseError(
                f"Google model identifier {token!r} is not a documented Gemini API model id"
            )
        return None
    return token


def _page_slug_from_href(href: str) -> str | None:
    raw = href.strip()
    if not raw or raw.startswith("#"):
        return None
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if host and host != "ai.google.dev":
        return None
    path = parsed.path or raw.split("#", 1)[0].split("?", 1)[0]
    match = _MODEL_PAGE_PATH.fullmatch(path)
    if match is None:
        return None
    slug = match.group(1)
    if slug in _SKIP_PAGE_SLUGS:
        return None
    if not re.fullmatch(r"^[a-z0-9][a-z0-9._-]*$", slug):
        return None
    return slug


def _visible_text(raw: str) -> str:
    text = _LINK.sub(r"\1", raw or "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return " ".join(text.split())


def _header_key(raw: str) -> str:
    return normalize_heading(_visible_text(raw))
