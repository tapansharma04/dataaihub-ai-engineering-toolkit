"""Orchestrate discovery, extraction, and read-only catalog lifecycle rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from samyak.__version__ import __version__
from samyak.app.catalog_view import load_catalog_view
from samyak.app.discovery import DiscoveredAppFile, discover_project, validate_project_root
from samyak.app.extractors.json_config import extract_json_observations
from samyak.app.extractors.python_ast import extract_python_observations
from samyak.app.extractors.toml_config import extract_toml_observations
from samyak.app.inventory import build_inventory
from samyak.app.models import (
    CAPABILITY_NAME,
    NOTICE_FILE_PARSE_FAILED,
    PRODUCT_NAME,
    REPORT_SCHEMA_VERSION,
    AppConfig,
    AppNotice,
    AppReport,
    AppSummary,
)
from samyak.app.observations import Observation, sort_observations
from samyak.app.rules.model_lifecycle import apply_model_lifecycle


@dataclass(frozen=True, slots=True)
class AppExtraction:
    """Internal result: observations plus the report envelope."""

    report: AppReport
    observations: tuple[Observation, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = self.report.to_dict()
        payload["observations"] = [item.to_dict() for item in self.observations]
        return payload


def extract_application(
    project_path: str | Path,
    config: AppConfig | None = None,
) -> AppExtraction:
    """Discover a project tree, extract observations, and apply lifecycle rules.

    Loads the local Model Intelligence overlay read-only. Does not refresh the
    catalog, call provider adapters, or use the network. One file is read at a
    time; source text is not retained.
    """
    cfg = config or AppConfig()
    root = validate_project_root(Path(project_path))
    discovery = discover_project(root, cfg)

    observations: list[Observation] = []
    notices = list(discovery.notices)
    parse_failures = 0
    analyzed_by_ext: dict[str, int] = {}
    parse_notices = 0

    for item in discovery.files:
        try:
            extracted = _extract_file(item)
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as exc:
            parse_failures += 1
            if parse_notices < cfg.max_notices_per_code:
                notices.append(_parse_notice(item.relative_path, exc))
                parse_notices += 1
            continue
        observations.extend(extracted)
        analyzed_by_ext[item.extension] = analyzed_by_ext.get(item.extension, 0) + 1

    ordered = sort_observations(observations)
    view, catalog_context, catalog_notices = load_catalog_view()
    notices.extend(catalog_notices)
    findings, rule_notices = apply_model_lifecycle(
        ordered,
        view,
        max_notices_per_code=cfg.max_notices_per_code,
    )
    notices.extend(rule_notices)
    notices.sort(key=lambda item: (item.code, item.path or "", item.message))
    inventory = build_inventory(ordered, view)
    summary = AppSummary(
        project_root=root.name,
        analyzed_files=sum(analyzed_by_ext.values()),
        skipped_oversized=discovery.skipped_oversized,
        skipped_binary=discovery.skipped_binary,
        parse_failures=parse_failures,
        truncated=discovery.truncated,
        analyzed_by_extension=dict(sorted(analyzed_by_ext.items())),
    )
    report = AppReport(
        product=PRODUCT_NAME,
        capability=CAPABILITY_NAME,
        version=__version__,
        report_schema_version=REPORT_SCHEMA_VERSION,
        summary=summary,
        catalog=catalog_context,
        inventory=inventory,
        notices=tuple(notices),
        findings=findings,
        config=cfg.to_dict(),
    )
    return AppExtraction(report=report, observations=ordered)


def _extract_file(item: DiscoveredAppFile) -> tuple[Observation, ...]:
    raw = item.absolute_path.read_bytes()
    try:
        if item.extension == ".py":
            text = _decode_text(raw)
            return extract_python_observations(item.relative_path, text)
        if item.extension == ".json":
            text = _decode_text(raw)
            return extract_json_observations(item.relative_path, text)
        if item.extension == ".toml":
            return extract_toml_observations(item.relative_path, raw)
    finally:
        del raw
    return ()


def _decode_text(raw: bytes) -> str:
    return raw.decode("utf-8-sig")


def _parse_notice(relative_path: str, exc: BaseException) -> AppNotice:
    return AppNotice(
        code=NOTICE_FILE_PARSE_FAILED,
        message=f"Failed to parse {relative_path}: {exc.__class__.__name__}.",
        path=relative_path,
    )
