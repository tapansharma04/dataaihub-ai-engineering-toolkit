# Samyak

AI engineering tooling for building reliable AI applications.

**Samyak is an open-source project by [DataAIHub](https://www.dataaihub.co).**

The current release provides **Corpus Intelligence** for analyzing document collections before they are used in AI and RAG systems.

## Corpus Intelligence

Analyze your document corpus before it becomes a RAG problem.

Many AI/RAG issues start in the source corpus — empty files, duplicates, tiny stubs, oversized dumps, noisy extraction — before retrieval, embeddings, or generation are involved.

Corpus Intelligence inspects a local document directory, collects evidence, explains why findings matter, and recommends practical next actions. It identifies characteristics that **may** affect downstream systems. It does **not** guarantee or predict retrieval or answer quality.

## Who it is for

- AI / ML engineers preparing document corpora for RAG or other retrieval workflows
- Data engineers validating knowledge-base handoffs
- Platform and QA teams adding corpus checks to local workflows or CI

## Features (v0.1)

- Corpus inventory (supported / unsupported / analyzed, by format)
- Empty and whitespace-only document detection
- Very small / low-information document signals
- Very large document heuristics
- Exact duplicate detection (normalized content hashing)
- Lightweight text-quality / noise indicators
- Simple chunkability indicators (not a chunking engine)
- **PDF** text-layer extraction with page-level stats (no OCR)
- PDF-specific checks (OCR-likely pages, headers/footers, complexity, table-like text)
- **HTML** local parsing (scripts/styles removed; no network fetches)
- HTML-specific checks (boilerplate, low content ratio, code-heavy pages)
- Human-readable CLI report and JSON output
- Incremental corpus processing (documents are not all held in memory at once)
- Progress on stderr for large corpora (`--no-progress` to disable)
- Fully local analysis — no network, no API keys, no telemetry

## Privacy / local processing

Samyak:

- processes documents on your machine
- makes **no** network or API calls
- sends **no** document content externally
- includes **no** telemetry
- requires **no** account and **no** API key

Treat corpus files as untrusted input: contents are never executed or interpreted as instructions.

## Supported formats (v0.1)

| Extension | Support |
| --- | --- |
| `.txt` | Yes |
| `.md` | Yes |
| `.pdf` | Yes (text layer via `pypdf`; no OCR) |
| `.html` / `.htm` | Yes (local files via BeautifulSoup; no crawling) |
| Other formats | Discovered and reported as unsupported (analysis continues) |

## Installation

Install from PyPI:

```bash
pip install samyak
```

Requires Python 3.12+.

The CLI command is:

```bash
samyak
```

## Quick start

```bash
samyak corpus ./documents
```

JSON:

```bash
samyak corpus ./documents --format json
```

Write a report file:

```bash
samyak corpus ./documents --format json --output report.json
```

Help and version:

```bash
samyak --help
samyak corpus --help
samyak --version
```

Try the bundled sample corpus:

```bash
samyak corpus examples/sample-corpus
```

## Checks implemented

| Check | What it looks for |
| --- | --- |
| Inventory | Discovered, supported, unsupported, analyzed counts and size stats |
| Empty documents | No meaningful text after stripping whitespace |
| Very small documents | Meaningful text below a documented character threshold (default: 100) |
| Very large documents | Meaningful text at/above a documented threshold (default: 100,000) |
| Exact duplicates | Identical normalized textual content (SHA-256 of normalized text) |
| Text quality | Conservative heuristics: high symbol ratio, excessive whitespace, repeated lines |
| Chunkability | Long uninterrupted blocks; documents dominated by very short lines |
| PDF (format-specific) | OCR-likely/text-poor pages, uneven page text, repeated headers/footers, extraction anomalies, large/complex PDFs, table-like text (heuristic) |
| HTML (format-specific) | Boilerplate domination, low main-content ratio, large pages, code-heavy pages, repeated navigation-like lines |

Default thresholds are documented in code (`AnalysisConfig`) and included in JSON output under `config`. Override size thresholds via `--small-chars` / `--large-chars`.

## Example output (text)

```text
Samyak Corpus Intelligence Report
=================================

Product:                  samyak
Capability:               corpus
Version:                  0.1.0

Corpus
------
Documents analyzed:       ...
Unsupported files:        ...

Findings
--------

HIGH  Exact duplicate documents

      2 documents belong to 1 exact-duplicate group(s).

      Why it matters:
      Duplicate content can produce redundant retrieval results...

      Recommendation:
      Review and remove duplicate documents before indexing...

Summary
-------
High:      ...
Medium:    ...
Low:       ...
Info:      ...
```

## JSON output

JSON includes:

- `product` / `capability` / `version` (product identity)
- `config` (thresholds used)
- `summary` (inventory and size statistics)
- `findings` (code, category, severity, message, why_it_matters, recommendation, evidence, affected_documents)
- `severity_counts`

Example identity fields:

```json
{
  "product": "samyak",
  "capability": "corpus",
  "version": "0.1.0"
}
```

Paths in reports are **relative to the corpus root** whenever possible.

Exit codes (v0.1):

- `0` — analysis completed
- non-zero — invalid path, I/O failure, or other runtime error

Findings (including `HIGH`) do **not** fail the process by themselves. v0.1 reports findings but does not fail CI based on finding severity.

### CI usage

```bash
samyak corpus ./documents --format json --output corpus-report.json
```

Keep stdout JSON-only for piping; progress messages go to stderr. Use `--no-progress` when you want silent stderr.

## Library usage

```python
from samyak import analyze_corpus

report = analyze_corpus("./documents")
print(report.product, report.capability, report.version)
print(report.severity_counts())
for finding in report.findings:
    print(finding.severity, finding.title, finding.recommendation)
```

## Validation & scale characteristics

Corpus Intelligence is validated beyond unit tests using public corpora and controlled defect injection. This repository does **not** redistribute third-party dataset files.

Documents are processed **incrementally**: peak memory for multi-document corpora is driven primarily by the largest active document plus compact corpus metadata (hashes, counters, bounded finding samples), not by total corpus text size.

Validated on Apple M1, 16 GB RAM, macOS arm64, Python 3.12 (release-candidate measurements for v0.1.0):

| Scenario | Approximate result |
| --- | --- |
| Multi-GB text corpora (≈250 MB → ≈5 GB) | Completes with low hundreds of MB peak RSS (not proportional to full corpus size) |
| Many small files (up to 100,000 docs) | Completes; runtime scales primarily with file count |
| Very large individual files (10–250 MB) | Peak RSS scales with the active file (~4.5–8× input size after optimization) |

Controlled detection validation covers empty documents, exact duplicates, size outliers, fragmentation/noise signals, PDF OCR-likely/text-poor pages, and HTML code-heavy pages. Performance varies by hardware, storage, and document formats. PDF analysis is substantially more expensive than plain text. These figures are measured characteristics, not guarantees.

## Limitations

- No OCR — scanned/image-only PDFs are *detected* but not converted to text
- PDF table detection is a conservative text heuristic, not a table extractor
- HTML main-content / boilerplate detection is heuristic; site structure varies widely
- No DOCX or other office formats yet
- No near-duplicate / semantic duplicate detection
- No embeddings, vector databases, or LLM judges
- Findings identify characteristics that **may** cause ingestion/chunking/retrieval problems; they do **not** predict downstream answer quality
- Thresholds may need adjustment for specialized document collections
- No single readiness score — findings are evidence-based, not oracles
- Very large *individual* documents still require memory proportional to that file
- Finding path lists in reports are bounded samples; full counts remain in evidence
- Multi-gigabyte *corpora* are processed incrementally and do not require holding all document text in RAM

## Samyak by DataAIHub

| Project | Role |
| --- | --- |
| [DataAIHub](https://www.dataaihub.co) | Knowledge, ecosystem, research, guides, comparisons, and discovery |
| [DataAIHub Cookbook](https://github.com/tapansharma04/dataaihub-cookbook) | Practical, runnable examples for learning AI engineering patterns |
| **Samyak** (this repository) | Open-source AI engineering product; current capability: Corpus Intelligence |

## Development

Requirements: Python 3.12+

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests
ruff format --check src tests
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
