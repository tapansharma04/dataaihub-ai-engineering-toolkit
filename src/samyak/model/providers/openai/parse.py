"""Parse captured OpenAI Markdown into provider observations.

Boolean capabilities:
- omitted / unlabeled → not extracted (later unknown or not_verified)
- explicit positive (``function_calling`` in Supported features, Support=Supported)
  → True
- explicit negative exists only on the Endpoints Support column (``Not supported``).
  Supported features has no negative form; omission there is not False.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from urllib.parse import urlparse

from samyak.model.identity import IdentityKind
from samyak.model.markdown import (
    bullet_items,
    code_spans,
    has_heading,
    labeled_value,
    markdown_links,
    normalize_heading,
    pipe_tables,
    section_named,
    split_sections,
)
from samyak.model.providers.openai.errors import OpenAIParseError
from samyak.model.providers.openai.observations import (
    OpenAILifecycle,
    OpenAIModelObservation,
    OpenAISourceRef,
)
from samyak.model.providers.openai.sources import (
    CapturedSource,
    OpenAISourceType,
)

_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MODEL_PAGE_PATH = re.compile(r"^/api/docs/models/([^/]+)\.md$")
_HEADING_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})\b")
_CONTEXT_WINDOW = re.compile(r"^([\d,]+)\s+context window\b", re.IGNORECASE)
_MAX_OUTPUT = re.compile(r"^([\d,]+)\s+max output tokens\b", re.IGNORECASE)
_TOKEN_COUNT = re.compile(r"^([\d,]+)\s+tokens?\b", re.IGNORECASE)
_ALIAS_ROUTES = re.compile(
    r"`([^`]+)`\s+alias routes requests to\s+`([^`]+)`",
    re.IGNORECASE,
)
_ALIAS_FOR = re.compile(
    r"`([^`]+)`\s+is an alias for\s+`([^`]+)`",
    re.IGNORECASE,
)
_FAMILY = re.compile(r"\bin the ([A-Za-z0-9][A-Za-z0-9.+ -]*?) family\b", re.IGNORECASE)
_FEATURE_TOOLS = frozenset({"function_calling", "function calling", "tool_calling", "tool calling"})
_FEATURE_STRUCTURED = frozenset({"structured_outputs", "structured outputs", "structured_output"})
_MODALITY_WORDS = {
    "text": "text",
    "image": "image",
    "audio": "audio",
    "video": "video",
}
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
_NON_MODEL_TOKENS = frozenset(
    {
        "---",
        "n/a",
        "none",
        "videos api",
        "assistants api",
        "responses api",
        "realtime api",
        "completions api",
    }
)


def parse_openai_sources(sources: Sequence[CapturedSource]) -> tuple[OpenAIModelObservation, ...]:
    """Parse captured official OpenAI Markdown. Does not fetch the network."""
    if not sources:
        raise OpenAIParseError("OpenAI documentation sources are missing")
    observations: list[OpenAIModelObservation] = []
    for source in sources:
        try:
            observations.extend(_parse_source(source))
        except OpenAIParseError:
            raise
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise OpenAIParseError(
                f"OpenAI documentation {source.source_id} could not be parsed"
            ) from exc
    return tuple(observations)


def _parse_source(source: CapturedSource) -> tuple[OpenAIModelObservation, ...]:
    if source.media_type != "text/markdown":
        raise OpenAIParseError(
            f"OpenAI documentation {source.source_id} must be Markdown, not {source.media_type}"
        )
    if source.source_type is OpenAISourceType.MODELS_INDEX:
        return _parse_index(source)
    if source.source_type is OpenAISourceType.DEPRECATIONS:
        return _parse_deprecations(source)
    return _parse_model_page(source)


def _source_ref(source: CapturedSource) -> OpenAISourceRef:
    return OpenAISourceRef(
        source_id=source.source_id,
        source_kind=source.source_kind,
        source_url=source.source_url,
        content_hash=source.content_hash,
        retrieved_at=source.retrieved_at,
        source_type=source.source_type,
    )


def _parse_index(source: CapturedSource) -> tuple[OpenAIModelObservation, ...]:
    sections = split_sections(source.body)
    if not has_heading(sections, "Models"):
        raise OpenAIParseError("OpenAI models index is missing a Models heading")
    seen: dict[str, OpenAIModelObservation] = {}
    ref = _source_ref(source)
    for label, target in markdown_links(source.body):
        model_id = _model_id_from_docs_href(target)
        if model_id is None:
            continue
        if model_id in seen:
            continue
        seen[model_id] = OpenAIModelObservation(
            provider_model_id=model_id,
            source=ref,
            display_name=label or None,
            listed_on_index=True,
        )
    if not seen:
        raise OpenAIParseError("OpenAI models index did not list any models")
    return tuple(seen.values())


def _parse_deprecations(source: CapturedSource) -> tuple[OpenAIModelObservation, ...]:
    sections = split_sections(source.body)
    if not has_heading(sections, "Deprecations"):
        raise OpenAIParseError("OpenAI deprecations page is missing a Deprecations heading")
    observations: list[OpenAIModelObservation] = []
    listing: str | None = None
    announced_on: str | None = None
    ref = _source_ref(source)
    for section in sections:
        title = normalize_heading(section.title)
        if title == "upcoming deprecations":
            listing = "upcoming"
            announced_on = None
            continue
        if title == "past deprecations":
            listing = "past"
            announced_on = None
            continue
        if listing is None:
            continue
        heading_date = _HEADING_DATE.match(section.title.strip())
        if heading_date is not None:
            announced_on = heading_date.group(1)
        for table in pipe_tables(section.body):
            observations.extend(
                _observations_from_deprecation_table(
                    table,
                    source=ref,
                    listing=listing,
                    announced_on=announced_on,
                )
            )
    if not observations:
        raise OpenAIParseError("OpenAI deprecations page did not list any models")
    return tuple(observations)


def _observations_from_deprecation_table(
    table: tuple[dict[str, str], ...],
    *,
    source: OpenAISourceRef,
    listing: str,
    announced_on: str | None,
) -> tuple[OpenAIModelObservation, ...]:
    observations: list[OpenAIModelObservation] = []
    for row in table:
        cells = {_normalize_column(key): value for key, value in row.items()}
        model_ids = _model_ids_from_cell(
            _first_cell(cells, ("model / system", "model snapshot", "model"))
        )
        if not model_ids:
            continue
        replacements = _model_ids_from_cell(
            _first_cell(
                cells,
                (
                    "recommended replacement",
                    "substitute model",
                    "recommended replacement base model",
                ),
            )
        )
        retirement_at = _parse_iso_date(
            _first_cell(cells, ("shutdown date", "retired date", "retirement date"))
        )
        deprecated_at = _parse_iso_date(_first_cell(cells, ("deprecation date", "deprecated date")))
        if deprecated_at is None:
            deprecated_at = announced_on
        if (
            deprecated_at is not None
            and retirement_at is not None
            and retirement_at < deprecated_at
        ):
            raise OpenAIParseError(
                f"OpenAI deprecations for {model_ids[0]} have retirement_at "
                "earlier than deprecated_at"
            )
        lifecycle = OpenAILifecycle.DEPRECATED
        for model_id in model_ids:
            observations.append(
                OpenAIModelObservation(
                    provider_model_id=model_id,
                    source=source,
                    lifecycle=lifecycle,
                    deprecation_listing=listing,
                    deprecated_at=deprecated_at,
                    retirement_at=retirement_at,
                    replacements=replacements or None,
                )
            )
    return tuple(observations)


def _parse_model_page(source: CapturedSource) -> tuple[OpenAIModelObservation, ...]:
    sections = split_sections(source.body)
    if not sections or sections[0].level != 1 or not sections[0].title:
        raise OpenAIParseError("OpenAI model page is missing a title heading")
    display_name = sections[0].title
    model_id = _model_id_from_labeled_field(source.body)
    if model_id is None:
        raise OpenAIParseError("OpenAI model page is missing a Model ID")
    details = section_named(sections, "Model details")
    details_text = details.body if details is not None else ""
    features = section_named(sections, "Supported features")
    endpoints = section_named(sections, "Endpoints")
    full_text = source.body

    resolves_to, extra_alias_ids = _alias_targets(_alias_search_text(sections), model_id)
    identity_kind = IdentityKind.ALIAS if resolves_to is not None else IdentityKind.CANONICAL

    lifecycle = None
    if _page_states_deprecated(full_text):
        lifecycle = OpenAILifecycle.DEPRECATED

    observation = OpenAIModelObservation(
        provider_model_id=model_id,
        source=_source_ref(source),
        identity_kind=identity_kind,
        display_name=display_name,
        aliases=extra_alias_ids or None,
        resolves_to=resolves_to,
        family=_family_from_prose(full_text),
        lifecycle=lifecycle,
        context_window_tokens=_context_window_tokens(details_text),
        max_input_tokens=_labeled_token_count(
            details_text, ("Maximum input tokens", "Max input tokens")
        ),
        max_output_tokens=_max_output_tokens(details_text),
        input_modalities=_modalities(labeled_value(details_text, "Input modalities")),
        output_modalities=_modalities(labeled_value(details_text, "Output modalities")),
        tool_calling=_feature_flag(features.body if features is not None else None, _FEATURE_TOOLS),
        structured_output=_feature_flag(
            features.body if features is not None else None, _FEATURE_STRUCTURED
        ),
        api_access=_api_access(endpoints.body if endpoints is not None else None),
        availability_scope=_scope_if_stated(full_text),
        regions=_regions_if_stated(full_text),
    )
    extras: list[OpenAIModelObservation] = [observation]
    for alias_id in extra_alias_ids:
        extras.append(
            OpenAIModelObservation(
                provider_model_id=alias_id,
                source=_source_ref(source),
                identity_kind=IdentityKind.ALIAS,
                display_name=None,
                resolves_to=model_id,
            )
        )
    if resolves_to is not None:
        extras.append(
            OpenAIModelObservation(
                provider_model_id=resolves_to,
                source=_source_ref(source),
                aliases=(model_id,),
            )
        )
    return tuple(extras)


def _model_id_from_docs_href(href: str) -> str | None:
    path = href.strip()
    parsed = urlparse(path)
    if parsed.scheme and parsed.netloc and parsed.netloc != "developers.openai.com":
        return None
    candidate = parsed.path or path
    match = _MODEL_PAGE_PATH.match(candidate)
    if match is None:
        return None
    return _require_model_id(match.group(1), required=False)


def _model_id_from_labeled_field(text: str) -> str | None:
    raw = labeled_value(text, "Model ID")
    if raw is None:
        return None
    spans = code_spans(raw)
    token = spans[0] if spans else raw.strip()
    return _require_model_id(token, required=True)


def _require_model_id(value: str, *, required: bool) -> str | None:
    token = value.strip().strip("*").strip()
    if not token:
        if required:
            raise OpenAIParseError("OpenAI model page is missing a Model ID")
        return None
    if not _MODEL_ID.fullmatch(token):
        if required:
            raise OpenAIParseError(
                f"OpenAI model identifier {token!r} is not a documented model id"
            )
        return None
    return token


def _alias_search_text(sections: tuple) -> str:
    """Read alias statements from intro and Snapshots/Aliases sections only.

    Pricing, endpoints, and comparison prose are ignored so similar wording
    there cannot create alias records.
    """
    parts: list[str] = []
    if sections and sections[0].level == 1:
        parts.append(sections[0].body)
    for title in ("Snapshots", "Aliases"):
        section = section_named(sections, title)
        if section is not None:
            parts.append(section.body)
    return "\n".join(parts)


def _alias_targets(text: str, page_model_id: str) -> tuple[str | None, tuple[str, ...]]:
    aliases_of_this: list[str] = []
    resolves_to: str | None = None
    for match in _ALIAS_ROUTES.finditer(text):
        alias_id = _require_model_id(match.group(1), required=False)
        target_id = _require_model_id(match.group(2), required=False)
        if alias_id is None or target_id is None:
            continue
        if alias_id == page_model_id:
            resolves_to = target_id
        elif target_id == page_model_id:
            aliases_of_this.append(alias_id)
    for match in _ALIAS_FOR.finditer(text):
        alias_id = _require_model_id(match.group(1), required=False)
        target_id = _require_model_id(match.group(2), required=False)
        if alias_id is None or target_id is None:
            continue
        if alias_id == page_model_id:
            resolves_to = target_id
        elif target_id == page_model_id:
            aliases_of_this.append(alias_id)
    unique = tuple(dict.fromkeys(aliases_of_this))
    return resolves_to, unique


def _family_from_prose(text: str) -> str | None:
    match = _FAMILY.search(text)
    if match is None:
        return None
    family = " ".join(match.group(1).split())
    lowered = family.lower()
    if "api" in lowered or lowered in {"openai", "the"}:
        return None
    return family


def _page_states_deprecated(text: str) -> bool:
    for raw in text.split("\n"):
        line = raw.strip()
        if line.startswith(">"):
            line = line[1:].strip()
        lowered = line.lower()
        if lowered.startswith("deprecated") and "see " not in lowered[:20]:
            return True
    return False


def _context_window_tokens(details: str) -> int | None:
    for item in bullet_items(details):
        match = _CONTEXT_WINDOW.match(item)
        if match is not None:
            return _parse_token_count(match.group(1))
    return None


def _max_output_tokens(details: str) -> int | None:
    labeled = _labeled_token_count(details, ("Maximum output tokens", "Max output tokens"))
    if labeled is not None:
        return labeled
    for item in bullet_items(details):
        match = _MAX_OUTPUT.match(item)
        if match is not None:
            return _parse_token_count(match.group(1))
    return None


def _labeled_token_count(details: str, labels: tuple[str, ...]) -> int | None:
    for label in labels:
        raw = labeled_value(details, label)
        if raw is None:
            continue
        stripped = raw.replace(" tokens", "").strip()
        match = _TOKEN_COUNT.match(raw) or re.match(r"^([\d,]+)$", stripped)
        if match is not None:
            return _parse_token_count(match.group(1))
    return None


def _parse_token_count(raw: str) -> int | None:
    digits = raw.replace(",", "").strip()
    if not digits.isdigit():
        return None
    value = int(digits)
    if value <= 0:
        return None
    return value


def _modalities(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not raw.strip():
        return None
    found: list[str] = []
    for part in re.split(r"[,/]| and ", raw.lower()):
        token = part.strip()
        mapped = _MODALITY_WORDS.get(token)
        if mapped is not None and mapped not in found:
            found.append(mapped)
    return tuple(found) or None


def _feature_flag(features_body: str | None, names: frozenset[str]) -> bool | None:
    """Return True only for an explicit positive listing.

    OpenAI Supported features has no explicit negative. Missing names stay
    unset (unknown), never False.
    """
    if features_body is None:
        return None
    items = {item.strip().lower().replace("-", "_") for item in bullet_items(features_body)}
    normalized = {item.replace(" ", "_") for item in items} | items
    wanted = {name.replace(" ", "_") for name in names}
    if normalized & wanted:
        return True
    return None


def _api_access(endpoints_body: str | None) -> bool | None:
    """Map the Endpoints Support column.

    ``Supported`` on any row is True. A table whose stated rows are all
    ``Not supported`` is False. A missing table is unset, not False.
    """
    if endpoints_body is None:
        return None
    tables = pipe_tables(endpoints_body)
    if not tables:
        return None
    saw_row = False
    supported = False
    for table in tables:
        for row in table:
            cells = {_normalize_column(key): value for key, value in row.items()}
            support = cells.get("support", "").strip().lower()
            if not support:
                continue
            saw_row = True
            if support == "supported":
                supported = True
    if not saw_row:
        return None
    return supported


def _scope_if_stated(text: str) -> str | None:
    raw = labeled_value(text, "Availability") or labeled_value(text, "Availability scope")
    if raw is None:
        return None
    token = raw.strip().split()[0].lower() if raw.strip() else ""
    if token in {"global", "regional", "account"}:
        return token
    return None


def _regions_if_stated(text: str) -> tuple[str, ...] | None:
    raw = labeled_value(text, "Regions")
    if raw is None or not raw.strip():
        return None
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parts or None


def _model_ids_from_cell(cell: str) -> tuple[str, ...]:
    if not cell.strip():
        return ()
    lowered = cell.strip().lower()
    if lowered in _NON_MODEL_TOKENS:
        return ()
    ids: list[str] = []
    for span in code_spans(cell):
        model_id = _require_model_id(span, required=False)
        if model_id is not None:
            ids.append(model_id)
    if not ids:
        for part in re.split(r"[,]| or | \| ", cell):
            model_id = _require_model_id(part, required=False)
            if model_id is not None:
                ids.append(model_id)
    return tuple(dict.fromkeys(ids))


def _first_cell(cells: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        if name in cells:
            return cells[name]
    return ""


def _normalize_column(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").split())


def _parse_iso_date(raw: str) -> str | None:
    text = raw.strip()
    if not text or text in {"---", "-"}:
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
