"""Text and JSON presentation for Application Readiness reports.

Renders an ``AppReport`` only. Does not re-run analysis and does not expose
the internal observation IR.
"""

from __future__ import annotations

import json

from samyak.app.models import (
    AppFinding,
    AppInventory,
    ApplicationIdentityState,
    AppNotice,
    AppReport,
    CatalogContext,
    CatalogEvidence,
    InventoryOperation,
    MatchKind,
    ModelDependency,
    SourceLocation,
    UnboundReason,
    UnresolvedOperation,
    public_identity_value,
)

REPORT_FORMATS = ("text", "json")


def render_app_report(report: AppReport, fmt: str) -> str:
    """Render ``report`` as ``text`` or ``json``."""
    if fmt == "json":
        return render_json_report(report)
    if fmt == "text":
        return render_text_report(report)
    raise ValueError(f"Unsupported report format: {fmt}")


def render_json_report(report: AppReport) -> str:
    """Serialize the public AppReport envelope. Observations are not included."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def render_text_report(report: AppReport) -> str:
    """Render an evidence-first terminal report. Not a readiness score."""
    summary = report.summary
    lines: list[str] = [
        "Samyak Application Analysis",
        "===========================",
        "",
        f"Product:                  {report.product}",
        f"Capability:               {report.capability}",
        f"Version:                  {report.version}",
        "",
        "Project",
        "-------",
        f"Root:                     {summary.project_root}",
        f"Analyzed files:           {summary.analyzed_files}",
        f"Skipped oversized:        {summary.skipped_oversized}",
        f"Skipped binary:           {summary.skipped_binary}",
        f"Parse failures:           {summary.parse_failures}",
        f"Truncated:                {_yes_no(summary.truncated)}",
        f"By extension:             {_extension_summary(summary.analyzed_by_extension)}",
        "",
    ]
    lines.extend(_render_catalog(report.catalog))
    lines.extend(_render_inventory(report.inventory))
    lines.extend(_render_findings(report.findings, catalog_present=report.catalog.present))
    lines.extend(_render_notices(report.notices))
    counts = report.severity_counts()
    lines.extend(
        [
            "Finding counts",
            "--------------",
            f"High:      {counts['HIGH']}",
            f"Medium:    {counts['MEDIUM']}",
            f"Low:       {counts['LOW']}",
            f"Info:      {counts['INFO']}",
            "",
            "This inspection is local and offline. It uses the local model catalog",
            "if present and does not refresh it. Finding counts are inventory, not",
            "a readiness score.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_catalog(catalog: CatalogContext) -> list[str]:
    lines = [
        "Catalog",
        "-------",
        f"Present:                  {_yes_no(catalog.present)}",
    ]
    if catalog.present:
        providers = ", ".join(catalog.providers) if catalog.providers else "(none)"
        lines.extend(
            [
                f"Freshness:                {catalog.freshness_status}",
                f"Generated at:             {catalog.generated_at}",
                f"Overlay:                  {catalog.overlay}",
                f"Providers:                {providers}",
            ]
        )
    lines.append("")
    return lines


_MAX_COMMON_EXPRESSIONS = 3


def _render_inventory(inventory: AppInventory) -> list[str]:
    lines = [
        "Inventory",
        "---------",
        "",
    ]
    lines.extend(_render_inventory_summary(inventory))
    if not inventory.dependencies and not inventory.unresolved_operations:
        lines.extend(
            [
                "No bound model dependencies.",
                "No unresolved operations.",
                "",
            ]
        )
        return lines
    lines.extend(_render_models(inventory.dependencies))
    lines.extend(_render_unresolved(inventory.unresolved_operations))
    return lines


def _render_inventory_summary(inventory: AppInventory) -> list[str]:
    established = 0
    candidate = 0
    resolved = 0
    ambiguous = 0
    for dependency in inventory.dependencies:
        state = dependency.application_identity
        if state is ApplicationIdentityState.ESTABLISHED:
            established += 1
        elif state is ApplicationIdentityState.CANDIDATE:
            candidate += 1
        elif state is ApplicationIdentityState.BOUND:
            resolved += 1
        elif state is ApplicationIdentityState.AMBIGUOUS:
            ambiguous += 1
    unresolved_total = len(inventory.unresolved_operations)
    lines = [
        "Application Inventory",
        f"  Model dependencies:     {len(inventory.dependencies)}",
        f"    Established:          {established}",
        f"    Candidate:            {candidate}",
        f"    Resolved:             {resolved}",
    ]
    if ambiguous:
        lines.append(f"    Ambiguous:            {ambiguous}")
    lines.append(f"  Unresolved operations:  {unresolved_total}")
    for phrase, group in _unresolved_groups(inventory.unresolved_operations):
        lines.append(f"    {len(group)} × {phrase}")
    lines.append("")
    return lines


def _render_models(dependencies: tuple[ModelDependency, ...]) -> list[str]:
    lines = ["Models"]
    if not dependencies:
        lines.extend(["  (none)", ""])
        return lines
    for dependency in dependencies:
        lines.append(f"  {dependency.identity_label}")
        lines.append(f"    Identity:     {public_identity_value(dependency.application_identity)}")
        if (
            dependency.provider_id is None
            and dependency.application_identity is not ApplicationIdentityState.ESTABLISHED
        ):
            lines.append("    Provider:     not established from source")
        citation = dependency.catalog_evidence
        if citation is not None:
            lines.extend(_format_inventory_catalog(dependency, citation))
        count = len(dependency.operations)
        label = "location" if count == 1 else "locations"
        lines.append(f"    Used:         {count} {label}")
        for operation in dependency.operations:
            lines.extend(_format_inventory_operation_lines(operation))
        lines.append("")
    return lines


def _format_inventory_catalog(dependency: ModelDependency, citation: CatalogEvidence) -> list[str]:
    lines: list[str] = []
    if citation.match_kind is MatchKind.MODEL_ID_UNIQUE and citation.candidate_samyak_ids:
        listed = ", ".join(citation.candidate_samyak_ids)
        lines.append(f"    Catalog:      unique overlay match {listed}")
    elif citation.match_kind is MatchKind.MODEL_ID_AMBIGUOUS and citation.candidate_samyak_ids:
        listed = ", ".join(citation.candidate_samyak_ids)
        lines.append(f"    Catalog:      ambiguous overlay matches {listed}")
    elif citation.samyak_id is not None:
        lines.append(f"    Catalog:      {citation.samyak_id}")
    elif citation.match_kind is MatchKind.NOT_IN_CATALOG:
        if dependency.provider_id is not None:
            lines.append("    Catalog:      not in provider catalog")
        else:
            lines.append("    Catalog:      not classified")
    elif citation.match_kind is MatchKind.PROVIDER_NOT_IN_CATALOG:
        lines.append("    Catalog:      provider has no overlay records")
    if citation.lifecycle_value is not None:
        lines.append(f"    Lifecycle:    {citation.lifecycle_value}")
    elif citation.lifecycle_status is not None:
        lines.append(f"    Lifecycle:    {citation.lifecycle_status}")
    elif dependency.application_identity is ApplicationIdentityState.CANDIDATE:
        lines.append("    Lifecycle:    not applied")
    return lines


def _format_inventory_operation_lines(operation: InventoryOperation) -> list[str]:
    pointer = _location_pointer(operation.path, operation.line_start, operation.line_end)
    slot = operation.slot or "model"
    if operation.expression is not None:
        shown = f"{slot}={operation.expression}"
    else:
        shown = f"{slot}={operation.detected_model_id!r}"
    lines = [f"      {pointer}  {shown}"]
    source = operation.resolution_source
    if source is None:
        return lines
    src_ptr = _location_pointer(source.path, source.line_start, source.line_end)
    name = source.symbol or operation.expression or ""
    lines.append(f"                 via {src_ptr}  {name}={source.detected_value!r}")
    return lines


def _render_unresolved(operations: tuple[UnresolvedOperation, ...]) -> list[str]:
    lines = ["Unresolved operations"]
    if not operations:
        lines.extend(["  (none)", ""])
        return lines
    for phrase, group in _unresolved_groups(operations):
        lines.append(f"  {len(group)} × {phrase}")
        common = _common_expressions(group)
        if common:
            listed = ", ".join(_format_common_expression(item, count) for item, count in common)
            lines.append(f"      common: {listed}")
    lines.append("")
    return lines


def _unresolved_groups(
    operations: tuple[UnresolvedOperation, ...],
) -> list[tuple[str, tuple[UnresolvedOperation, ...]]]:
    phrase_order: list[str] = []
    buckets: dict[str, list[UnresolvedOperation]] = {}
    for reason in UnboundReason:
        phrase = _unbound_reason_phrase(reason)
        if phrase not in buckets:
            buckets[phrase] = []
            phrase_order.append(phrase)
    for item in operations:
        buckets[_unbound_reason_phrase(item.unbound_reason)].append(item)
    return [(phrase, tuple(buckets[phrase])) for phrase in phrase_order if buckets[phrase]]


def _common_expressions(
    operations: tuple[UnresolvedOperation, ...],
) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in operations:
        slot = item.slot or "model"
        shown = f"{slot}={item.expression}"
        counts[shown] = counts.get(shown, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:_MAX_COMMON_EXPRESSIONS]


def _format_common_expression(shown: str, count: int) -> str:
    if count == 1:
        return shown
    return f"{shown} ({count})"


def _location_pointer(path: str, line_start: int | None, line_end: int | None) -> str:
    if line_start is None:
        return path
    if line_end is not None and line_end != line_start:
        return f"{path}:{line_start}-{line_end}"
    return f"{path}:{line_start}"


def _unbound_reason_phrase(reason: UnboundReason) -> str:
    """Human-readable text for an unbound reason. JSON still uses the enum value."""
    if reason is UnboundReason.DYNAMIC_EXPRESSION:
        return "model identity is determined at runtime"
    if reason is UnboundReason.DYNAMIC_CALL:
        return "model identity comes from a runtime call"
    if reason is UnboundReason.DYNAMIC_SUBSCRIPT:
        return "model identity comes from a runtime lookup"
    return "model identity could not be resolved from this file"


def _render_notices(notices: tuple[AppNotice, ...]) -> list[str]:
    lines = [
        "Notices",
        "-------",
    ]
    if not notices:
        lines.extend(["", "No analysis notices.", ""])
        return lines
    for notice in notices:
        location = f"  ({notice.path})" if notice.path else ""
        lines.extend(["", f"{notice.code}{location}", f"      {notice.message}"])
    lines.append("")
    return lines


def _render_findings(findings: tuple[AppFinding, ...], *, catalog_present: bool) -> list[str]:
    lines = [
        "Findings",
        "--------",
    ]
    if not findings:
        if catalog_present:
            empty = "No findings."
        else:
            empty = (
                "No findings were emitted. Lifecycle rules were not applied "
                "because the local catalog was not available."
            )
        lines.extend(["", empty, ""])
        return lines
    for finding in findings:
        lines.extend(_render_finding(finding))
    lines.append("")
    return lines


def _render_finding(finding: AppFinding) -> list[str]:
    lines = [
        "",
        f"{finding.severity.value}  {finding.title}",
        f"      Code: {finding.code}",
        "",
    ]
    lines.extend(_format_source(finding.evidence, expression=finding.reference_expression))
    if finding.resolution_source is not None:
        lines.extend(_format_resolution(finding.resolution_source))
    if finding.catalog_evidence is not None:
        lines.extend(_format_catalog(finding.catalog_evidence))
    lines.extend(
        [
            "",
            f"      {finding.message}",
            "",
            "      Why it matters:",
            f"      {finding.why_it_matters}",
            "",
            "      Recommendation:",
            f"      {finding.recommendation}",
        ]
    )
    return lines


def _format_source(location: SourceLocation, *, expression: str | None = None) -> list[str]:
    pointer = _location_pointer(location.path, location.line_start, location.line_end)
    slot = location.config_key or location.symbol
    if expression is not None:
        detail = f"{slot}={expression}" if slot else expression
    else:
        detail = f"{slot}={location.detected_value!r}" if slot else repr(location.detected_value)
    return [
        f"      Source: {pointer}  ({location.evidence_type.value})",
        f"              {detail}",
    ]


def _format_resolution(location: SourceLocation) -> list[str]:
    pointer = _location_pointer(location.path, location.line_start, location.line_end)
    name = location.symbol or ""
    detail = f"{name}={location.detected_value!r}" if name else repr(location.detected_value)
    return [
        f"      Resolution: {pointer}  ({location.evidence_type.value})",
        f"                  {detail}",
    ]


def _format_catalog(evidence: CatalogEvidence) -> list[str]:
    identity = (
        evidence.samyak_id
        if evidence.samyak_id is not None
        else "(not an established catalog identity)"
    )
    lines = [
        f"      Catalog: {identity}",
        f"               match: {evidence.match_kind.value}",
    ]
    if evidence.provider_id is not None:
        lines.append(f"               provider: {evidence.provider_id}")
    if evidence.identity_kind is not None:
        lines.append(f"               identity: {evidence.identity_kind}")
    if evidence.lifecycle_status is not None:
        lifecycle = evidence.lifecycle_status
        if evidence.lifecycle_value is not None:
            lifecycle = f"{lifecycle} {evidence.lifecycle_value}"
        lines.append(f"               lifecycle: {lifecycle}")
    if evidence.catalog_freshness:
        status = evidence.catalog_freshness.get("status")
        if status:
            lines.append(f"               freshness: {status}")
    if evidence.candidate_samyak_ids:
        lines.append("               candidates: " + ", ".join(evidence.candidate_samyak_ids))
    if evidence.replacement_samyak_ids:
        lines.append("               replacement: " + ", ".join(evidence.replacement_samyak_ids))
    return lines


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _extension_summary(analyzed_by_extension: dict[str, int]) -> str:
    if not analyzed_by_extension:
        return "(none)"
    return ", ".join(
        f"{extension}={count}" for extension, count in sorted(analyzed_by_extension.items())
    )
