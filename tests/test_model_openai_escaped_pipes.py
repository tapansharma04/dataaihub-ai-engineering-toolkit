"""Escaped pipes in the archived OpenAI deprecations page. No network."""

from __future__ import annotations

import hashlib
from pathlib import Path

from samyak.model.catalog import CatalogOverlay
from samyak.model.facts import FactStatus
from samyak.model.providers.openai.adapter import catalog_from_openai_sources
from samyak.model.providers.openai.parse import parse_openai_sources
from samyak.model.providers.openai.sources import (
    DEPRECATIONS_URL,
    SOURCE_ID_DEPRECATIONS,
    captured_markdown,
)

FIXTURE = Path(__file__).parent / "fixtures" / "models" / "openai" / "deprecations-escaped-pipes.md"
# sha256 of archive blob openai-docs-deprecations in generation
# 2026-10-30T17-00-52.530312Z, confirmed against that manifest before copy.
ARCHIVE_SHA256 = "3fddf4c625d689e43e559c236bcdd675de8a3b3d035070ba205827d8ad0a2fc2"
RETRIEVED_AT = "2026-10-30T17:01:25.910647+00:00"
VERIFIED_AT = "2026-10-30T17:00:52+00:00"

# identity, aliases, replacements, deprecated_at, retirement_at
# Taken from the 27 escaped-pipe rows. The first code span is the row identity.
_ROWS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str, str], ...] = (
    ("gpt-5.4-nano", ("gpt-5.4-nano-2026-03-17",), ("gpt-6-luna",), "2026-10-01", "2027-04-01"),
    ("gpt-5.1", ("gpt-5.1-2025-11-13",), ("gpt-6-sol",), "2026-10-01", "2027-04-01"),
    (
        "gpt-4o-mini-tts",
        ("gpt-4o-mini-tts-2025-03-20",),
        ("gpt-realtime-2.1-mini",),
        "2026-10-01",
        "2027-01-06",
    ),
    (
        "gpt-4o-mini-tts",
        ("gpt-4o-mini-tts-2025-12-15",),
        ("gpt-realtime-2.1-mini",),
        "2026-10-01",
        "2027-01-06",
    ),
    (
        "gpt-4o-mini-transcribe",
        ("gpt-4o-mini-transcribe-2025-12-15",),
        ("gpt-live-transcribe", "gpt-transcribe"),
        "2026-08-26",
        "2027-02-26",
    ),
    (
        "gpt-realtime",
        ("gpt-realtime-2025-08-28",),
        ("gpt-realtime-2.1",),
        "2026-07-20",
        "2027-01-20",
    ),
    ("gpt-audio", ("gpt-audio-2025-08-28",), ("gpt-audio-1.5",), "2026-07-20", "2027-01-20"),
    (
        "gpt-realtime-mini",
        ("gpt-realtime-mini-2025-12-15",),
        ("gpt-realtime-2.1-mini",),
        "2026-07-20",
        "2027-01-20",
    ),
    (
        "gpt-audio-mini",
        ("gpt-audio-mini-2025-12-15",),
        ("gpt-audio-1.5",),
        "2026-07-20",
        "2027-01-20",
    ),
    ("gpt-5", ("gpt-5-2025-08-07",), ("gpt-5.6-sol",), "2026-06-11", "2026-12-11"),
    ("gpt-5-mini", ("gpt-5-mini-2025-08-07",), ("gpt-5.6-terra",), "2026-06-11", "2026-12-11"),
    ("gpt-5-nano", ("gpt-5-nano-2025-08-07",), ("gpt-5.6-luna",), "2026-06-11", "2026-12-11"),
    ("gpt-5-pro", ("gpt-5-pro-2025-10-06",), ("gpt-5.6-sol",), "2026-06-11", "2026-12-11"),
    ("o3", ("o3-2025-04-16",), ("gpt-5.6-sol",), "2026-06-11", "2026-12-11"),
    ("o3-pro", ("o3-pro-2025-06-10",), ("gpt-5.6-sol",), "2026-06-11", "2026-12-11"),
    (
        "gpt-image-1.5",
        ("gpt-image-1.5-2025-12-16",),
        ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"),
        "2026-06-02",
        "2026-12-01",
    ),
    (
        "gpt-3.5-turbo-0125",
        ("gpt-3.5-turbo", "gpt-3.5-turbo-completions"),
        ("gpt-5.6-terra",),
        "2026-04-22",
        "2026-10-23",
    ),
    (
        "gpt-4-0613",
        ("gpt-4", "gpt-4-0613-completions", "gpt-4-completions"),
        ("gpt-5.6-sol",),
        "2026-04-22",
        "2026-10-23",
    ),
    (
        "gpt-4-turbo",
        ("gpt-4-turbo-2024-04-09", "gpt-4-turbo-completions"),
        ("gpt-5.6-sol",),
        "2026-04-22",
        "2026-10-23",
    ),
    (
        "gpt-4.1-nano",
        ("gpt-4.1-nano-2025-04-14",),
        ("gpt-5.6-luna",),
        "2026-04-22",
        "2026-10-23",
    ),
    ("o1-2024-12-17", ("o1",), ("gpt-5.6-sol",), "2026-04-22", "2026-10-23"),
    ("o1-pro-2025-03-19", ("o1-pro",), ("gpt-5.6-sol",), "2026-04-22", "2026-10-23"),
    ("o3-mini-2025-01-31", ("o3-mini",), ("gpt-5.6-sol",), "2026-04-22", "2026-10-23"),
    ("o4-mini-2025-04-16", ("o4-mini",), ("gpt-5.6-terra",), "2026-04-22", "2026-10-23"),
    (
        "computer-use-preview-2025-03-11",
        ("computer-use-preview",),
        ("gpt-5.6-terra",),
        "2026-04-22",
        "2026-07-23",
    ),
    (
        "o3-deep-research-2025-06-26",
        ("o3-deep-research",),
        ("gpt-5.6-sol",),
        "2026-04-22",
        "2026-07-23",
    ),
    (
        "o4-mini-deep-research-2025-06-26",
        ("o4-mini-deep-research",),
        ("gpt-5.6-sol",),
        "2026-04-22",
        "2026-07-23",
    ),
)


def _source():
    return captured_markdown(
        source_id=SOURCE_ID_DEPRECATIONS,
        source_url=DEPRECATIONS_URL,
        body=FIXTURE.read_text(encoding="utf-8"),
        retrieved_at=RETRIEVED_AT,
    )


def _replacement_ids(record) -> list[str]:
    assert record.replacement.status is FactStatus.KNOWN
    assert record.replacement.value is not None
    return [item.provider_model_id for item in record.replacement.value]


def test_fixture_matches_the_archived_deprecations_blob() -> None:
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest == ARCHIVE_SHA256
    escaped_rows = [
        line
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.startswith("|") and "\\|" in line
    ]
    assert len(escaped_rows) == 27
    assert len(_ROWS) == 27


def test_each_escaped_pipe_row_keeps_identity_alias_and_successor() -> None:
    observations = parse_openai_sources((_source(),))
    for identity, aliases, replacements, deprecated_at, retirement_at in _ROWS:
        matches = [
            item
            for item in observations
            if item.provider_model_id == identity
            and item.aliases == aliases
            and item.replacements == replacements
            and item.deprecated_at == deprecated_at
            and item.retirement_at == retirement_at
        ]
        assert matches, identity
    alias_ids = {alias for _identity, aliases, *_rest in _ROWS for alias in aliases}
    identities = {item.provider_model_id for item in observations}
    assert alias_ids.isdisjoint(identities)


def test_known_models_use_the_documented_successor() -> None:
    catalog = catalog_from_openai_sources(
        (_source(),),
        generated_at=VERIFIED_AT,
        verified_at=VERIFIED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    records = {record.provider_model_id: record for record in catalog.models}
    expected = {
        "gpt-5": ["gpt-5.6-sol"],
        "gpt-5-mini": ["gpt-5.6-terra"],
        "gpt-5-nano": ["gpt-5.6-luna"],
        "gpt-5-pro": ["gpt-5.6-sol"],
        "gpt-5.1": ["gpt-6-sol"],
        "gpt-5.4-nano": ["gpt-6-luna"],
        "o3": ["gpt-5.6-sol"],
        "o3-pro": ["gpt-5.6-sol"],
        "gpt-4o-mini-transcribe": ["gpt-live-transcribe", "gpt-transcribe"],
        "gpt-4o-mini-tts": ["gpt-realtime-2.1-mini"],
    }
    for model_id, successors in expected.items():
        assert _replacement_ids(records[model_id]) == successors
        assert model_id in records
    tts = records["gpt-4o-mini-tts"]
    assert tts.replacement.status is FactStatus.KNOWN
    assert tts.aliases.status is FactStatus.CONFLICT
    alias_claims = {claim.value for claim in tts.aliases.claims}
    assert alias_claims == {
        ("gpt-4o-mini-tts-2025-03-20",),
        ("gpt-4o-mini-tts-2025-12-15",),
    }
    gpt5 = records["gpt-5"]
    assert gpt5.aliases.status is FactStatus.KNOWN
    assert gpt5.aliases.value == ("gpt-5-2025-08-07",)
    assert gpt5.deprecated_at.value == "2026-06-11"
    assert gpt5.retirement_at.value == "2026-12-11"
    snapshots = [
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
        "gpt-5-pro-2025-10-06",
        "gpt-5.1-2025-11-13",
        "gpt-5.4-nano-2026-03-17",
        "o3-2025-04-16",
        "o3-pro-2025-06-10",
        "gpt-4o-mini-transcribe-2025-12-15",
        "gpt-4o-mini-tts-2025-03-20",
        "gpt-4o-mini-tts-2025-12-15",
    ]
    assert set(snapshots).isdisjoint(records)
