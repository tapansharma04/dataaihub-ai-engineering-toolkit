"""Parse captured Fireworks Markdown into provider observations.

Fireworks AI serving source only. Origin/creator identities are not created.
Display names are never used as API model IDs. Customer deployment paths,
library URLs, and Hugging Face origin IDs are ignored except where embeddings
docs publish an HF-style serving ID. The List Models API is not a source.

Inference guides contribute documented example serving IDs. They are not a
complete Serverless or Model Library inventory. Pricing fields such as
``$x.xx/M Input`` are not token limits and are not parsed.

Lifecycle:
- "no longer enabled" / "Retired" with an ID → retired
- "deprecated from serverless" without shutdown → deprecated from that
  documented serving surface, not retired from Fireworks
- announced decommission/effective dates may be retirement_at
- "at least 2 weeks notice" is not a retirement date
- "legacy" BERT/sentence-transformers is a category, not LEGACY
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

from samyak.model.identity import IdentityKind
from samyak.model.markdown import code_spans, normalize_heading, pipe_tables, split_sections
from samyak.model.providers.fireworks.errors import FireworksParseError
from samyak.model.providers.fireworks.observations import (
    FireworksLifecycle,
    FireworksModelObservation,
    FireworksSourceRef,
)
from samyak.model.providers.fireworks.sources import CapturedSource, FireworksSourceType

_RESOURCE_ID = re.compile(r"^accounts/fireworks/(?:models|routers)/[a-z0-9][a-z0-9._-]*$")
_SHORT_FIREWORKS_ID = re.compile(r"^fireworks/[a-z0-9][a-z0-9._-]*$")
_HF_STYLE_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_PLACEHOLDER = re.compile(r"[<>]|ACCOUNT_ID|DEPLOYMENT_ID|YOUR_")
_MODEL_KWARG = re.compile(
    r"""(?:\bmodel\s*=\s*|["']model["']\s*:\s*)["']([^"']+)["']""",
)
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
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
_TOKEN = re.compile(r"(?P<num>[\d.]+)\s*(?P<unit>[kKmM])?\b")
_ALIAS_FOR = re.compile(
    r"`([^`]+)`\s+is an alias for\s+`([^`]+)`",
    re.IGNORECASE,
)


def parse_fireworks_sources(
    sources: Sequence[CapturedSource],
) -> tuple[FireworksModelObservation, ...]:
    """Parse captured official Fireworks Markdown. Does not fetch the network."""
    if not sources:
        raise FireworksParseError("Fireworks documentation sources are missing")
    observations: list[FireworksModelObservation] = []
    for source in sources:
        try:
            observations.extend(_parse_source(source))
        except FireworksParseError:
            raise
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise FireworksParseError(
                f"Fireworks documentation {source.source_id} could not be parsed"
            ) from exc
    if not observations:
        raise FireworksParseError("Fireworks documentation did not list any serving model ids")
    return tuple(observations)


def _parse_source(source: CapturedSource) -> tuple[FireworksModelObservation, ...]:
    if source.media_type != "text/markdown":
        raise FireworksParseError(
            f"Fireworks documentation {source.source_id} must be Markdown, not {source.media_type}"
        )
    if source.source_type is FireworksSourceType.EMBEDDINGS:
        return _parse_embeddings(source)
    if source.source_type is FireworksSourceType.SERVING_PATHS:
        return _parse_serving_paths(source)
    if source.source_type is FireworksSourceType.CHANGELOG:
        return _parse_changelog(source)
    if source.source_type is FireworksSourceType.TEXT_MODELS:
        return _parse_inference_guide(source, input_modalities=("text",), api_access=True)
    if source.source_type is FireworksSourceType.VISION_MODELS:
        return _parse_inference_guide(source, input_modalities=("text", "image"), api_access=True)
    if source.source_type is FireworksSourceType.TOOL_CALLING:
        return _parse_inference_guide(source, tool_calling=True, api_access=True)
    return _parse_model_page(source)


def _source_ref(source: CapturedSource) -> FireworksSourceRef:
    return FireworksSourceRef(
        source_id=source.source_id,
        source_kind=source.source_kind,
        source_url=source.source_url,
        content_hash=source.content_hash,
        retrieved_at=source.retrieved_at,
        source_type=source.source_type,
    )


def _parse_embeddings(source: CapturedSource) -> tuple[FireworksModelObservation, ...]:
    if "embedding" not in source.body.lower() and "rerank" not in source.body.lower():
        raise FireworksParseError("Fireworks embeddings page is missing embeddings documentation")
    ref = _source_ref(source)
    observations: list[FireworksModelObservation] = []
    for section in split_sections(source.body):
        hint = _embeddings_serverless_hint(f"{section.title}\n{section.body}")
        for table in pipe_tables(section.body):
            observations.extend(
                _embeddings_table_observations(table, source=ref, serverless_hint=hint)
            )
    observations.extend(_retired_embedder_observations(source.body, source=ref))
    for model_id in _embedding_example_ids(source.body):
        observations.append(
            FireworksModelObservation(
                provider_model_id=model_id,
                source=ref,
                documented_serverless=True,
                input_modalities=("text",),
                api_access=True,
            )
        )
    if not observations:
        raise FireworksParseError("Fireworks embeddings page did not list any serving model ids")
    return tuple(observations)


def _embeddings_table_observations(
    table: tuple[dict[str, str], ...],
    *,
    source: FireworksSourceRef,
    serverless_hint: bool | None = None,
) -> tuple[FireworksModelObservation, ...]:
    if not table:
        return ()
    columns = {_header_key(key): key for key in table[0]}
    id_key = columns.get("model id") or columns.get("model")
    if id_key is None:
        return ()
    name_key = columns.get("model") if columns.get("model") != id_key else None
    availability_key = columns.get("availability")
    context_key = columns.get("context")
    allow_hf = "model id" in columns
    observations: list[FireworksModelObservation] = []
    for row in table:
        model_ids = _model_ids_from_cell(row.get(id_key, ""), allow_hf=allow_hf)
        if not model_ids:
            continue
        display = _visible_text(row.get(name_key, "")) if name_key else ""
        serverless = False
        if availability_key is not None:
            availability = _visible_text(row.get(availability_key, "")).lower()
            serverless = "serverless" in availability
        elif serverless_hint is True:
            serverless = True
        context = _parse_tokens(row.get(context_key, "")) if context_key else None
        for model_id in model_ids:
            observations.append(
                FireworksModelObservation(
                    provider_model_id=model_id,
                    source=source,
                    display_name=display or None,
                    documented_serverless=serverless,
                    context_window_tokens=context,
                    input_modalities=("text",),
                    api_access=True if serverless else None,
                )
            )
    return tuple(observations)


def _embeddings_serverless_hint(text: str) -> bool | None:
    lowered = text.lower()
    if "not available on serverless" in lowered:
        return False
    if (
        "serverless only" in lowered
        or "still work on serverless" in lowered
        or "still serve on serverless" in lowered
    ):
        return True
    return None


def _retired_embedder_observations(
    body: str, *, source: FireworksSourceRef
) -> tuple[FireworksModelObservation, ...]:
    observations: list[FireworksModelObservation] = []
    for match in re.finditer(r"(?is)retired\.(.{0,800})", body):
        chunk = match.group(0)
        if "no longer enabled" not in chunk.lower():
            continue
        ids = _model_ids_from_cell(chunk, allow_hf=True)
        replacement_chunk = ""
        use_split = re.split(r"(?i)\buse\s+", chunk, maxsplit=1)
        prefix = use_split[0]
        if len(use_split) == 2:
            replacement_chunk = use_split[1]
        replacements = tuple(_model_ids_from_cell(replacement_chunk, allow_hf=True))
        retired_ids = tuple(
            item for item in _model_ids_from_cell(prefix, allow_hf=True) if item not in replacements
        )
        if not retired_ids:
            retired_ids = tuple(item for item in ids if item not in replacements)
        for model_id in retired_ids:
            observations.append(
                FireworksModelObservation(
                    provider_model_id=model_id,
                    source=source,
                    lifecycle=FireworksLifecycle.RETIRED,
                    replacements=replacements or None,
                    api_access=False,
                )
            )
    return tuple(observations)


def _parse_serving_paths(source: CapturedSource) -> tuple[FireworksModelObservation, ...]:
    lowered = source.body.lower()
    if "serving path" not in lowered and "fast" not in lowered:
        raise FireworksParseError(
            "Fireworks serving paths page is missing serving-path documentation"
        )
    ref = _source_ref(source)
    observations: list[FireworksModelObservation] = []
    for table in pipe_tables(source.body):
        if not table:
            continue
        columns = {_header_key(key): key for key in table[0]}
        id_key = columns.get("model id")
        name_key = columns.get("model")
        if id_key is None:
            continue
        for row in table:
            model_ids = _model_ids_from_cell(row.get(id_key, ""), allow_hf=False)
            display = _visible_text(row.get(name_key, "")) if name_key else ""
            for model_id in model_ids:
                observations.append(
                    FireworksModelObservation(
                        provider_model_id=model_id,
                        source=ref,
                        display_name=display or None,
                        documented_serverless=True,
                        api_access=True,
                    )
                )
    for model_id in _inference_example_ids(source.body):
        observations.append(
            FireworksModelObservation(
                provider_model_id=model_id,
                source=ref,
                documented_serverless=True,
                api_access=True,
            )
        )
    unique = _dedupe_keep_first(observations)
    if not unique:
        raise FireworksParseError("Fireworks serving paths page did not list any serving model ids")
    return unique


def _parse_changelog(source: CapturedSource) -> tuple[FireworksModelObservation, ...]:
    if "changelog" not in source.body.lower() and "deprecat" not in source.body.lower():
        raise FireworksParseError("Fireworks changelog is missing deprecation documentation")
    ref = _source_ref(source)
    observations: list[FireworksModelObservation] = []
    listing = "schedule"
    for section in split_sections(source.body):
        heading = normalize_heading(section.title)
        if heading.startswith("current"):
            listing = "current"
        elif heading.startswith("upcoming"):
            listing = "upcoming"
        elif heading.startswith("past"):
            listing = "past"
        for chunk in _changelog_units(section.body):
            if not _model_ids_from_cell(chunk, allow_hf=True):
                continue
            observations.extend(_changelog_chunk_observations(chunk, source=ref, listing=listing))
    return tuple(_dedupe_keep_first(observations))


def _changelog_units(body: str) -> tuple[str, ...]:
    units = tuple(part.strip() for part in re.split(r"\n\s*\n", body) if part.strip())
    return units if units else ((body,) if body.strip() else ())


def _changelog_chunk_observations(
    body: str, *, source: FireworksSourceRef, listing: str
) -> tuple[FireworksModelObservation, ...]:
    ids = _model_ids_from_cell(body, allow_hf=True)
    if not ids:
        return ()
    lowered = body.lower()
    lifecycle = None
    retirement = None
    if "no longer enabled" in lowered:
        lifecycle = FireworksLifecycle.RETIRED
        retirement = _first_english_date(body)
    elif (
        "no longer be available" in lowered
        or "decommissioned" in lowered
        or "deprecated from serverless" in lowered
        or re.search(r"\bis deprecated\b", lowered)
    ):
        lifecycle = FireworksLifecycle.DEPRECATED
        retirement = _first_english_date(body)
    if lifecycle is None:
        return ()
    replacements = _changelog_replacements(body)
    deprecated_ids = tuple(item for item in ids if item not in replacements)
    observations = [
        FireworksModelObservation(
            provider_model_id=model_id,
            source=source,
            lifecycle=lifecycle,
            deprecation_listing=listing,
            retirement_at=retirement,
            replacements=tuple(item for item in replacements if item != model_id) or None,
        )
        for model_id in deprecated_ids
    ]
    return tuple(observations)


def _changelog_replacements(body: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in re.finditer(r"(?i)migrate to\s+(.+?)(?:\.|$)", body):
        found.extend(_model_ids_from_cell(match.group(1), allow_hf=True))
    for match in re.finditer(r"(?i)use\s+`([^`]+)` instead", body):
        found.extend(_model_ids_from_cell(match.group(1), allow_hf=True))
    return tuple(dict.fromkeys(found))


def _parse_inference_guide(
    source: CapturedSource,
    *,
    input_modalities: tuple[str, ...] | None = None,
    tool_calling: bool | None = None,
    api_access: bool | None = None,
) -> tuple[FireworksModelObservation, ...]:
    """Record documented example serving IDs. Not an inventory table."""
    ref = _source_ref(source)
    return tuple(
        FireworksModelObservation(
            provider_model_id=model_id,
            source=ref,
            documented_serverless=True,
            input_modalities=input_modalities,
            tool_calling=tool_calling,
            api_access=api_access,
        )
        for model_id in _inference_example_ids(source.body)
    )


def _parse_model_page(source: CapturedSource) -> tuple[FireworksModelObservation, ...]:
    ref = _source_ref(source)
    ids = _inference_example_ids(source.body)
    if not ids:
        raise FireworksParseError("Fireworks model page is missing a serving model id")
    extras: list[FireworksModelObservation] = []
    for model_id in ids:
        alias_target, alias_ids = _alias_targets(source.body, model_id)
        kind = IdentityKind.ALIAS if alias_target is not None else IdentityKind.CANONICAL
        extras.append(
            FireworksModelObservation(
                provider_model_id=model_id,
                source=ref,
                identity_kind=kind,
                documented_serverless=True,
                api_access=True,
                aliases=alias_ids or None,
                resolves_to=alias_target,
            )
        )
        for alias_id in alias_ids:
            extras.append(
                FireworksModelObservation(
                    provider_model_id=alias_id,
                    source=ref,
                    identity_kind=IdentityKind.ALIAS,
                    resolves_to=model_id,
                )
            )
    return tuple(extras)


def _inference_example_ids(body: str) -> tuple[str, ...]:
    ids: list[str] = []
    for match in _MODEL_KWARG.finditer(body):
        model_id = _require_model_id(match.group(1), allow_hf=False)
        if model_id is not None:
            ids.append(model_id)
    for token in code_spans(body):
        model_id = _require_model_id(token, allow_hf=False)
        if model_id is not None:
            ids.append(model_id)
    return tuple(dict.fromkeys(ids))


def _embedding_example_ids(body: str) -> tuple[str, ...]:
    """API ``model`` values from embeddings examples, including Hugging Face ids."""
    ids: list[str] = []
    for match in _MODEL_KWARG.finditer(body):
        model_id = _require_model_id(match.group(1), allow_hf=True)
        if model_id is not None:
            ids.append(model_id)
    return tuple(dict.fromkeys(ids))


def _alias_targets(text: str, page_model_id: str) -> tuple[str | None, tuple[str, ...]]:
    aliases: list[str] = []
    resolves_to: str | None = None
    for match in _ALIAS_FOR.finditer(text):
        alias_id = _require_model_id(match.group(1), allow_hf=False)
        target_id = _require_model_id(match.group(2), allow_hf=False)
        if alias_id is None or target_id is None:
            continue
        if alias_id == page_model_id:
            resolves_to = target_id
        elif target_id == page_model_id:
            aliases.append(alias_id)
    return resolves_to, tuple(dict.fromkeys(aliases))


def _model_ids_from_cell(raw: str, *, allow_hf: bool) -> tuple[str, ...]:
    ids: list[str] = []
    for token in code_spans(raw):
        model_id = _require_model_id(token, allow_hf=allow_hf)
        if model_id is not None:
            ids.append(model_id)
    if not ids:
        for token in re.split(r"[\s,]+", _visible_text(raw)):
            model_id = _require_model_id(token.strip("`"), allow_hf=allow_hf)
            if model_id is not None:
                ids.append(model_id)
    return tuple(dict.fromkeys(ids))


def _require_model_id(value: str, *, allow_hf: bool) -> str | None:
    token = value.strip().strip("*").strip()
    if not token or _PLACEHOLDER.search(token):
        return None
    if token.startswith("accounts/") and "/deployments/" in token:
        return None
    if not token.startswith("accounts/fireworks/") and token.startswith("accounts/"):
        return None
    if _RESOURCE_ID.fullmatch(token) or _SHORT_FIREWORKS_ID.fullmatch(token):
        return token
    if allow_hf and _HF_STYLE_ID.fullmatch(token) and not token.startswith("http"):
        if token.startswith("fireworks/"):
            return token
        return token
    return None


def _parse_tokens(raw: str) -> int | None:
    """Parse a documented token count. Never used for pricing fields."""
    text = _visible_text(raw)
    if not text:
        return None
    match = _TOKEN.search(text)
    if match is None:
        return None
    try:
        value = float(match.group("num"))
    except ValueError:
        return None
    unit = (match.group("unit") or "").lower()
    if unit == "k":
        value *= 1_000
    elif unit == "m":
        value *= 1_000_000
    tokens = int(value)
    return tokens if tokens > 0 else None


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
    observations: Sequence[FireworksModelObservation],
) -> tuple[FireworksModelObservation, ...]:
    seen: set[tuple[str, str]] = set()
    unique: list[FireworksModelObservation] = []
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
