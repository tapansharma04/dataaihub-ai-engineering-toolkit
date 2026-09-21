"""Parse captured Together Markdown into provider observations.

Together AI serverless serving source only. Organization/creator identities
are not created. Display names are never used as API model IDs. Dedicated
endpoints, DCI, customer uploads, and www.together.ai/models are ignored.

Allowlisted docs.together.ai serving-model quickstarts contribute explicit
token facts. Context, max input, and max output are captured only when the
page states them; input is never inferred as context minus output.

Lifecycle:
- serverless catalogue listing is not itself a lifecycle state
- scheduled deprecation with a removal date → deprecated
- after the documented removal date → retired (normalized from verified_at)
- deprecation history → retired
- redirect/upgrade → replacement, not retirement, not alias, not current serverless membership
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import date

from samyak.model.markdown import code_spans, normalize_heading, pipe_tables, split_sections
from samyak.model.providers.together.errors import TogetherParseError
from samyak.model.providers.together.observations import (
    TogetherLifecycle,
    TogetherModelObservation,
    TogetherSourceRef,
)
from samyak.model.providers.together.sources import CapturedSource, TogetherSourceType

_MODEL_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._+-]+$")
_PLACEHOLDER = re.compile(r"[<>]|YOUR_|ACCOUNT_ID|DEPLOYMENT")
_ISO_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
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
_YES_NO = {
    "yes": True,
    "true": True,
    "supported": True,
    "no": False,
    "false": False,
    "not supported": False,
}
_CATEGORY = {
    "chat": "chat",
    "chat models": "chat",
    "image": "image",
    "image models": "image",
    "vision": "vision",
    "vision models": "vision",
    "video": "video",
    "video models": "video",
    "audio": "audio",
    "audio models": "audio",
    "embedding": "embedding",
    "embedding models": "embedding",
    "embeddings": "embedding",
    "moderation": "moderation",
    "moderation models": "moderation",
}
_ID_HEADERS = frozenset(
    {
        "api model string",
        "model string for api",
        "model",
    }
)
_SKIP_ID_HEADERS = frozenset({"model name", "organization"})
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_REPLACEMENT = re.compile(
    r"(?i)recommended replacement:\s*`([^`]+)`",
)
_AMOUNT = r"(?P<num>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<unit>[kKmM])?"
_PRIMARY_MODEL_ID = re.compile(
    r"(?i)(?:the\s+model\s+id\s+is|model\s+(?:id|string)(?:\s+for\s+api)?\s*(?:is|:))\s*`([^`]+)`"
)
_CONTEXT_WINDOW = re.compile(
    rf"{_AMOUNT}(?:-|\s+)?(?:token\s+)?context\s+window",
    re.IGNORECASE,
)
_CONTEXT_LENGTH = re.compile(
    rf"context\s+length:\s*{_AMOUNT}(?:\s*tokens?)?",
    re.IGNORECASE,
)
_OUTPUT_UP_TO = re.compile(
    rf"(?:supports\s+)?up\s+to\s+{_AMOUNT}\s+output\s+tokens",
    re.IGNORECASE,
)
_OUTPUT_CAP = re.compile(
    rf"{_AMOUNT}(?:-|\s+)token(?:s)?\s+output\s+cap",
    re.IGNORECASE,
)
_INPUT_CAP = re.compile(
    rf"accepts\s+up\s+to\s+{_AMOUNT}\s+input\s+tokens",
    re.IGNORECASE,
)


def parse_together_sources(
    sources: Sequence[CapturedSource],
) -> tuple[TogetherModelObservation, ...]:
    """Parse captured official Together Markdown. Does not fetch the network."""
    if not sources:
        raise TogetherParseError("Together documentation sources are missing")
    observations: list[TogetherModelObservation] = []
    for source in sources:
        try:
            observations.extend(_parse_source(source))
        except TogetherParseError:
            raise
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise TogetherParseError(
                f"Together documentation {source.source_id} could not be parsed"
            ) from exc
    if not observations:
        raise TogetherParseError("Together documentation did not list any serving model ids")
    return tuple(observations)


def _parse_source(source: CapturedSource) -> tuple[TogetherModelObservation, ...]:
    if source.media_type != "text/markdown":
        raise TogetherParseError(
            f"Together documentation {source.source_id} must be Markdown, not {source.media_type}"
        )
    if source.source_type is TogetherSourceType.SERVERLESS:
        return _parse_serverless(source)
    if source.source_type is TogetherSourceType.DEPRECATIONS:
        return _parse_deprecations(source)
    if source.source_type is TogetherSourceType.MODEL_PAGE:
        return _parse_model_page(source)
    return _parse_changelog(source)


def _source_ref(source: CapturedSource) -> TogetherSourceRef:
    return TogetherSourceRef(
        source_id=source.source_id,
        source_kind=source.source_kind,
        source_url=source.source_url,
        content_hash=source.content_hash,
        retrieved_at=source.retrieved_at,
        source_type=source.source_type,
    )


def _parse_serverless(source: CapturedSource) -> tuple[TogetherModelObservation, ...]:
    if "serverless" not in source.body.lower():
        raise TogetherParseError("Together serverless page is missing serverless documentation")
    ref = _source_ref(source)
    grouped: dict[str, list[TogetherModelObservation]] = defaultdict(list)
    for section in split_sections(source.body):
        category = _category_for_heading(section.title)
        if category is None:
            continue
        if category == "rerank":
            continue
        for table in pipe_tables(section.body):
            for observation in _serverless_table_observations(table, source=ref, category=category):
                grouped[observation.provider_model_id].append(observation)
    merged: list[TogetherModelObservation] = []
    for items in grouped.values():
        merged.extend(_serverless_observations_for_id(items))
    if not merged:
        raise TogetherParseError("Together serverless page did not list any serving model ids")
    return tuple(merged)


def _category_for_heading(title: str) -> str | None:
    heading = normalize_heading(title)
    if heading.startswith("rerank"):
        return "rerank"
    if "example" in heading:
        return None
    return _CATEGORY.get(heading)


def _serverless_table_observations(
    table: tuple[dict[str, str], ...],
    *,
    source: TogetherSourceRef,
    category: str,
) -> tuple[TogetherModelObservation, ...]:
    if not table:
        return ()
    columns = {_header_key(key): key for key in table[0]}
    id_key = _id_column(columns)
    if id_key is None:
        return ()
    name_key = columns.get("model name")
    context_key = columns.get("context length") or columns.get("context")
    tool_key = columns.get("function calling")
    structured_key = columns.get("structured outputs")
    modality_key = columns.get("modality")
    observations: list[TogetherModelObservation] = []
    for row in table:
        model_id = _first_model_id(row.get(id_key, ""))
        if model_id is None:
            continue
        display = _visible_text(row.get(name_key, "")) if name_key else ""
        inputs, outputs = _modalities_for_category(
            category, _visible_text(row.get(modality_key, "")) if modality_key else ""
        )
        observations.append(
            TogetherModelObservation(
                provider_model_id=model_id,
                source=source,
                display_name=display or None,
                documented_serverless=True,
                context_window_tokens=(
                    _parse_tokens(row.get(context_key, "")) if context_key else None
                ),
                input_modalities=inputs,
                output_modalities=outputs,
                tool_calling=_parse_yes_no(row.get(tool_key, "")) if tool_key else None,
                structured_output=(
                    _parse_yes_no(row.get(structured_key, "")) if structured_key else None
                ),
                api_access=True,
            )
        )
    return tuple(observations)


def _serverless_observations_for_id(
    items: Sequence[TogetherModelObservation],
) -> tuple[TogetherModelObservation, ...]:
    if len(items) == 1:
        return tuple(items)
    inputs, outputs, display = _union_serverless_identity(items)
    patched = tuple(
        TogetherModelObservation(
            provider_model_id=item.provider_model_id,
            source=item.source,
            display_name=display,
            documented_serverless=True,
            context_window_tokens=item.context_window_tokens,
            input_modalities=inputs,
            output_modalities=outputs,
            tool_calling=item.tool_calling,
            structured_output=item.structured_output,
            api_access=True,
        )
        for item in items
    )
    contexts = {
        item.context_window_tokens for item in patched if item.context_window_tokens is not None
    }
    tools = {item.tool_calling for item in patched if item.tool_calling is not None}
    structured = {item.structured_output for item in patched if item.structured_output is not None}
    if len(contexts) > 1 or len(tools) > 1 or len(structured) > 1:
        return patched
    first = patched[0]
    return (
        TogetherModelObservation(
            provider_model_id=first.provider_model_id,
            source=first.source,
            display_name=display,
            documented_serverless=True,
            context_window_tokens=next(iter(contexts), None),
            input_modalities=inputs,
            output_modalities=outputs,
            tool_calling=next(iter(tools), None),
            structured_output=next(iter(structured), None),
            api_access=True,
        ),
    )


def _union_serverless_identity(
    items: Sequence[TogetherModelObservation],
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None, str | None]:
    inputs: list[str] = []
    outputs: list[str] = []
    display = None
    for item in items:
        if display is None and item.display_name:
            display = item.display_name
        for value in item.input_modalities or ():
            if value not in inputs:
                inputs.append(value)
        for value in item.output_modalities or ():
            if value not in outputs:
                outputs.append(value)
    return (tuple(inputs) or None, tuple(outputs) or None, display)


def _modalities_for_category(
    category: str, audio_modality: str
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    if category == "chat":
        return ("text",), ("text",)
    if category == "vision":
        return ("text", "image"), ("text",)
    if category == "image":
        return None, ("image",)
    if category == "video":
        return None, ("video",)
    if category == "embedding":
        return ("text",), None
    if category == "moderation":
        return ("text",), None
    if category == "audio":
        lowered = audio_modality.lower()
        if "speech-to-text" in lowered or "transcription" in lowered:
            return ("audio",), ("text",)
        if "text-to-speech" in lowered:
            return ("text",), ("audio",)
        return None, None
    return None, None


def _parse_deprecations(source: CapturedSource) -> tuple[TogetherModelObservation, ...]:
    lowered = source.body.lower()
    if "deprecat" not in lowered and "redirect" not in lowered:
        raise TogetherParseError("Together deprecations page is missing deprecation documentation")
    ref = _source_ref(source)
    observations: list[TogetherModelObservation] = []
    listing = "schedule"
    for section in split_sections(source.body):
        heading = normalize_heading(section.title)
        if heading.startswith("scheduled"):
            listing = "current"
        elif "history" in heading or heading.startswith("inference"):
            listing = "past"
        observations.extend(_redirect_observations(section.body, source=ref))
        observations.extend(
            _deprecation_table_observations(section.body, source=ref, listing=listing)
        )
    return tuple(observations)


def _redirect_observations(
    body: str, *, source: TogetherSourceRef
) -> tuple[TogetherModelObservation, ...]:
    observations: list[TogetherModelObservation] = []
    for table in pipe_tables(body):
        if not table:
            continue
        columns = {_header_key(key): key for key in table[0]}
        original_key = columns.get("original model")
        target_key = columns.get("redirects to")
        if original_key is None or target_key is None:
            continue
        for row in table:
            original = _first_model_id(row.get(original_key, ""))
            target = _first_model_id(row.get(target_key, ""))
            if original is None or target is None or original == target:
                continue
            observations.append(
                TogetherModelObservation(
                    provider_model_id=original,
                    source=source,
                    deprecation_listing="current",
                    replacements=(target,),
                )
            )
    return tuple(observations)


def _deprecation_table_observations(
    body: str, *, source: TogetherSourceRef, listing: str
) -> tuple[TogetherModelObservation, ...]:
    observations: list[TogetherModelObservation] = []
    for table in pipe_tables(body):
        if not table:
            continue
        columns = {_header_key(key): key for key in table[0]}
        model_key = columns.get("model")
        date_key = columns.get("removal date")
        replacement_key = columns.get("recommended replacement")
        if model_key is None:
            continue
        retired = listing == "past"
        for row in table:
            model_id = _first_model_id(row.get(model_key, ""))
            if model_id is None:
                continue
            removal = _parse_iso_date(row.get(date_key, "")) if date_key else None
            replacement = _first_model_id(row.get(replacement_key, "")) if replacement_key else None
            observations.append(
                TogetherModelObservation(
                    provider_model_id=model_id,
                    source=source,
                    lifecycle=(
                        TogetherLifecycle.RETIRED if retired else TogetherLifecycle.DEPRECATED
                    ),
                    deprecation_listing=listing,
                    retirement_at=removal,
                    replacements=(
                        (replacement,) if replacement and replacement != model_id else None
                    ),
                    api_access=not retired,
                )
            )
    return tuple(observations)


def _parse_changelog(source: CapturedSource) -> tuple[TogetherModelObservation, ...]:
    if "changelog" not in source.body.lower() and "deprecat" not in source.body.lower():
        raise TogetherParseError("Together changelog is missing changelog documentation")
    ref = _source_ref(source)
    observations: list[TogetherModelObservation] = []
    for section in split_sections(source.body):
        heading = normalize_heading(section.title)
        if _skip_changelog_catalogue_section(heading):
            continue
        if "redirect" in heading:
            observations.extend(_redirect_observations(section.body, source=ref))
            continue
        if "deprecat" not in heading:
            continue
        observations.extend(_changelog_deprecation_observations(section.body, source=ref))
    return tuple(observations)


def _skip_changelog_catalogue_section(heading: str) -> bool:
    if "deprecat" in heading or "redirect" in heading:
        return False
    if "dedicated" in heading:
        return True
    return heading.startswith("new ")


def _changelog_deprecation_observations(
    body: str, *, source: TogetherSourceRef
) -> tuple[TogetherModelObservation, ...]:
    lowered = body.lower()
    retired = "no longer available" in lowered
    retirement = _first_english_date(body) or _first_iso_date(body)
    observations: list[TogetherModelObservation] = []
    for raw_line in body.split("\n"):
        line = raw_line.strip()
        if not line.startswith(("- ", "* ")):
            continue
        model_id = _first_model_id(line)
        if model_id is None:
            continue
        replacement = None
        match = _REPLACEMENT.search(line)
        if match is not None:
            replacement = _require_model_id(match.group(1))
        observations.append(
            TogetherModelObservation(
                provider_model_id=model_id,
                source=source,
                lifecycle=(TogetherLifecycle.RETIRED if retired else TogetherLifecycle.DEPRECATED),
                deprecation_listing="current",
                retirement_at=retirement,
                replacements=((replacement,) if replacement and replacement != model_id else None),
                api_access=not retired,
            )
        )
    observations.extend(_deprecation_table_observations(body, source=source, listing="current"))
    return tuple(_dedupe_keep_first(observations))


def _id_column(columns: dict[str, str]) -> str | None:
    for key in _ID_HEADERS:
        found = columns.get(key)
        if found is not None and key not in _SKIP_ID_HEADERS:
            return found
    return None


def _first_model_id(raw: str) -> str | None:
    for token in code_spans(raw):
        model_id = _require_model_id(token)
        if model_id is not None:
            return model_id
    text = _visible_text(raw)
    return _require_model_id(text)


def _require_model_id(value: str) -> str | None:
    token = value.strip().strip("*").strip()
    if not token or _PLACEHOLDER.search(token):
        return None
    if token.startswith("http"):
        return None
    if _MODEL_ID.fullmatch(token):
        return token
    return None


def _parse_yes_no(raw: str) -> bool | None:
    text = _visible_text(raw).lower()
    if not text or text in {"-", "—", "n/a"}:
        return None
    return _YES_NO.get(text)


def _parse_tokens(raw: str) -> int | None:
    text = _visible_text(raw).replace(",", "")
    if not text or text in {"-", "—"}:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmM])?(?:\s*tokens?)?", text)
    if match is None:
        return None
    return _tokens_from_parts(match.group(1), match.group(2))


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


def _tokens_from_match(match: re.Match[str]) -> int | None:
    return _tokens_from_parts(match.group("num"), match.group("unit"))


def _parse_model_page(source: CapturedSource) -> tuple[TogetherModelObservation, ...]:
    ref = _source_ref(source)
    body = _unwrap_markdown_emphasis(source.body)
    model_id = _model_page_primary_id(body)
    if model_id is None:
        raise TogetherParseError("Together model page is missing a serving model id")
    return (
        TogetherModelObservation(
            provider_model_id=model_id,
            source=ref,
            context_window_tokens=_first_token_match(body, _CONTEXT_WINDOW, _CONTEXT_LENGTH),
            max_input_tokens=_first_token_match(body, _INPUT_CAP),
            max_output_tokens=_first_token_match(body, _OUTPUT_UP_TO, _OUTPUT_CAP),
            api_access=True,
        ),
    )


def _unwrap_markdown_emphasis(body: str) -> str:
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", body)


def _model_page_primary_id(body: str) -> str | None:
    match = _PRIMARY_MODEL_ID.search(body)
    if match is None:
        return None
    return _require_model_id(match.group(1))


def _first_token_match(body: str, *patterns: re.Pattern[str]) -> int | None:
    for pattern in patterns:
        match = pattern.search(body)
        if match is not None:
            tokens = _tokens_from_match(match)
            if tokens is not None:
                return tokens
    return None


def _parse_iso_date(raw: str) -> str | None:
    text = _visible_text(raw)
    match = _ISO_DATE.fullmatch(text)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1)).isoformat()
    except ValueError:
        return None


def _first_iso_date(text: str) -> str | None:
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1)).isoformat()
    except ValueError:
        return None


def _first_english_date(text: str) -> str | None:
    match = _ENGLISH_DATE.search(text)
    if match is None:
        return None
    month = _MONTHS.get(match.group("month").lower())
    if month is None:
        return None
    try:
        return date(int(match.group("year")), month, int(match.group("day"))).isoformat()
    except ValueError:
        return None


def _dedupe_keep_first(
    observations: Sequence[TogetherModelObservation],
) -> tuple[TogetherModelObservation, ...]:
    seen: set[tuple[str, str]] = set()
    unique: list[TogetherModelObservation] = []
    for item in observations:
        key = (item.provider_model_id, item.source.source_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)


def _visible_text(raw: str) -> str:
    text = _LINK.sub(r"\1", raw or "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return " ".join(text.split())


def _header_key(raw: str) -> str:
    return normalize_heading(_visible_text(raw))
