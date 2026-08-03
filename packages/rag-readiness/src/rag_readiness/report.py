"""Human-readable and JSON report rendering."""

from __future__ import annotations

import json
from typing import Any

from rag_readiness.models import AnalysisReport, Finding, Severity


def render_text_report(report: AnalysisReport) -> str:
    """Render a clean terminal-oriented text report."""
    summary = report.summary
    lines: list[str] = [
        "RAG Data Readiness Report",
        "=========================",
        "",
        "Corpus",
        "------",
        f"Root:                     {summary.corpus_root}",
        f"Files discovered:         {summary.total_discovered_files}",
        f"Supported files:          {summary.supported_files}",
        f"Unsupported files:        {summary.unsupported_files}",
        f"Documents analyzed:       {summary.analyzed_documents}",
        f"Load errors:              {summary.load_errors}",
        f"Total characters:         {summary.total_characters}",
        f"Total bytes:              {summary.total_bytes}",
        f"Average characters:       {_fmt_float(summary.average_characters)}",
        f"Median characters:        {_fmt_float(summary.median_characters)}",
        f"Min characters:           {_fmt_optional_int(summary.min_characters)}",
        f"Max characters:           {_fmt_optional_int(summary.max_characters)}",
        "",
        "Findings",
        "--------",
    ]

    if not report.findings:
        lines.append("")
        lines.append("No findings. The analyzed corpus did not trigger v0.1 checks.")
    else:
        for finding in report.findings:
            lines.extend(_render_finding(finding))

    counts = report.severity_counts()
    lines.extend(
        [
            "",
            "Summary",
            "-------",
            f"High:      {counts[Severity.HIGH.value]}",
            f"Medium:    {counts[Severity.MEDIUM.value]}",
            f"Low:       {counts[Severity.LOW.value]}",
            f"Info:      {counts[Severity.INFO.value]}",
            "",
            "Note: Findings highlight corpus characteristics that may affect RAG systems.",
            "They do not predict or guarantee downstream retrieval or answer quality.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_json_report(report: AnalysisReport) -> str:
    """Serialize the report to stable, pretty-printed JSON."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=False) + "\n"


def report_as_dict(report: AnalysisReport) -> dict[str, Any]:
    """Return the JSON-serializable report dictionary."""
    return report.to_dict()


def _render_finding(finding: Finding) -> list[str]:
    sample = finding.evidence.get("sample_paths") or list(finding.affected_documents[:5])
    lines = [
        "",
        f"{finding.severity.value}  {finding.title}",
        "",
        f"      {finding.message}",
        "",
        "      Why it matters:",
        f"      {finding.why_it_matters}",
        "",
        "      Recommendation:",
        f"      {finding.recommendation}",
    ]
    if sample:
        shown = ", ".join(str(path) for path in sample[:5])
        extra = len(finding.affected_documents) - min(5, len(sample))
        suffix = f" (+{extra} more)" if extra > 0 else ""
        lines.extend(["", f"      Examples: {shown}{suffix}"])
    return lines


def _fmt_float(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.1f}"


def _fmt_optional_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)
