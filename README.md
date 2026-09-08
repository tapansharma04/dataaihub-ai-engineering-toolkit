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

## Features (v0.1.1)

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
- Human-readable CLI report, JSON output, and self-contained HTML reports
- Incremental corpus processing (documents are not all held in memory at once)
- Progress on stderr for large corpora (`--no-progress` to disable)
- Fully local analysis — no network, no API keys, no telemetry
- Optional local run history (`--save`) and loopback viewer (`samyak view`)
- Baseline → current comparison of two saved runs

## Privacy / local processing

Samyak:

- processes documents on your machine
- makes **no** network or API calls
- sends **no** document content externally
- includes **no** telemetry
- requires **no** account and **no** API key

Treat corpus files as untrusted input: contents are never executed or interpreted as instructions.

Saved runs (when you pass `--save`) stay in a local cache directory on this machine. Samyak does not upload them. Run-history metadata may include the local corpus directory path so you can tell identically named folders apart; that path is not added to analysis reports.

## Local Run History

Run History currently lets you save previous analysis runs, reopen them, and compare two saved runs. Comparison is available in the local viewer; it is not a hosted service.

`--save` is **opt-in**. Without it, `samyak corpus PATH` behaves as in v0.1: text, JSON, and HTML reports are unchanged, and nothing is written to the run cache.

```bash
samyak corpus ./documents --save
samyak view
```

`samyak view` starts a local HTTP server on **127.0.0.1** (default port **15500**) and prints a URL such as `http://127.0.0.1:15500/`. The dashboard lists recent saved runs (corpus **basename**, timestamp, run ID, counts, and status). Opening a run shows that run’s metadata — corpus name and the local **corpus path** as separate fields — plus the existing Samyak HTML report, reconstructed from stored JSON. You do not need the original corpus files to still exist, and you do not need to rerun analysis.

To compare two runs, select exactly two rows on the history page and choose **Compare selected runs**. The earlier saved run is the **baseline**; the later run is **current**. Deltas are baseline → current. The comparison page summarizes inventory changes (files discovered/supported/unsupported/analyzed, finding counts, load/discovery errors) and groups findings as new, no longer detected, changed, or unchanged. A finding that is no longer detected was present in the baseline snapshot and absent from the current snapshot; that is not proof the underlying issue was fixed.

Findings are matched by their stable finding **code** (for example `EMPTY_DOCUMENTS`), not by rendered text or list order. Changed findings show what differed (severity, affected documents, recommendation, evidence). If analysis configuration differs, the page shows each changed setting as `baseline → current` so threshold changes are not mistaken for corpus changes.

Comparison reads the saved `AnalysisReport` snapshots. Runs with different report schemas or analysis capabilities cannot be compared. Two runs from different saved corpus locations can be compared, but the page labels that clearly — matching folder names is not treated as the same corpus.

```bash
samyak view --port 15500
samyak view --no-open
```

Data location (on this machine only):

- Default on macOS and Linux: `~/.cache/samyak/`
- Default on Windows: `%LOCALAPPDATA%\samyak\`
- Override: `SAMYAK_CACHE_DIR`

Each run is stored as `runs/<run-id>/report.json` and `runs/<run-id>/metadata.json`.

- `report.json` is the existing `AnalysisReport` JSON (document paths stay **corpus-relative**).
- `metadata.json` is local run-history metadata: timestamps, counts, a corpus **basename**, and the **local corpus directory path** from when `--save` was used. That path is machine-local metadata for distinguishing folders with the same name. It is not included in stdout, `--format json`, `--format html`, or `--output` files.

The viewer binds to loopback only and makes no network requests outside this machine. Nothing is uploaded.

## Supported formats (v0.1)

| Extension | Support |
| --- | --- |
| `.txt` | Yes |
| `.md` | Yes |
| `.pdf` | Yes (text layer via `pypdf`; no OCR) |
| `.html` / `.htm` | Yes (local files via BeautifulSoup; no crawling) |
| Other formats | Discovered and reported as unsupported (analysis continues) |

Point Samyak at a **document corpus directory**, not a source-code repository. Discovery is recursive and will include unsupported files it finds (for example under `.git` or `node_modules`) as inventory, not as a project linter.

## Installation

Install from PyPI to use Samyak:

```bash
pip install samyak
```

Requires Python 3.12+.

The CLI command is:

```bash
samyak
```

## Quick start

Run Corpus Intelligence on a local document directory:

```bash
samyak corpus ./documents
```

JSON:

```bash
samyak corpus ./documents --format json
```

HTML (self-contained file; open locally, no network required):

```bash
samyak corpus ./documents --format html --output report.html
```

Save a run for the local viewer:

```bash
samyak corpus ./documents --save
```

Write a JSON report file:

```bash
samyak corpus ./documents --format json --output report.json
```

Help and version:

```bash
samyak --help
samyak corpus --help
samyak view --help
samyak --version
```

The `examples/sample-corpus` directory is in this Git repository (and the source distribution). It is **not** included when you `pip install samyak`. After installing Samyak, clone the repository and analyze the example directory:

```bash
pip install samyak
git clone https://github.com/tapansharma04/dataaihub-ai-engineering-toolkit.git
samyak corpus dataaihub-ai-engineering-toolkit/examples/sample-corpus
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
| Load failures | Supported files that could not be parsed (analysis continues) |
| Discovery access failures | Paths that could not be listed or inspected (analysis continues) |

Default thresholds are documented on `AnalysisConfig` and included in JSON output under `config`. Override size thresholds via `--small-chars` / `--large-chars`.

## Example output (text)

```text
Samyak Corpus Intelligence Report
=================================

Product:                  samyak
Capability:               corpus
Version:                  0.1.1

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
  "version": "0.1.1"
}
```

Paths in reports are **relative to the corpus root** whenever possible.

### HTML output

`--format html` renders the same `AnalysisReport` as a standalone HTML page. Open the file in a browser; it requires no network access, JavaScript framework, or external assets.

```bash
samyak corpus ./documents --format html --output report.html
```

Omit `--output` to print HTML to stdout (same contract as text and JSON).

Exit codes (v0.1):

- `0` — analysis completed (findings do not change the exit code)
- `2` — invalid path, invalid configuration, missing subcommand, or I/O failure writing `--output`

Findings (including `HIGH`) do **not** fail the process by themselves. v0.1 reports findings but does not fail CI based on finding severity.

### CI usage

```bash
samyak corpus ./documents --format json --output corpus-report.json
```

Keep stdout JSON-only for piping; progress messages go to stderr. Use `--no-progress` when you want silent stderr.

## Library usage

```python
from samyak import AnalysisConfig, analyze_corpus

report = analyze_corpus("./documents")
print(report.product, report.capability, report.version)
print(report.severity_counts())
for finding in report.findings:
    print(finding.severity, finding.title, finding.recommendation)

report = analyze_corpus(
    "./documents",
    config=AnalysisConfig(small_document_chars=50),
)
```

## Validation & scale characteristics

Corpus Intelligence is validated beyond unit tests using public corpora and controlled defect injection. This repository does **not** redistribute third-party dataset files.

Documents are processed **incrementally**: peak memory for multi-document corpora is driven primarily by the largest active document plus compact corpus metadata (hashes, counters, bounded finding samples), not by total corpus text size.

Measured on Apple M1, 16 GB RAM, macOS arm64, Python 3.12 for v0.1.0:

| Scenario | Approximate result |
| --- | --- |
| Multi-GB text corpora (≈250 MB → ≈5 GB) | Completes with low hundreds of MB peak RSS (not proportional to full corpus size) |
| Many small files (up to 100,000 docs) | Completes; runtime scales primarily with file count |
| Very large individual files (10–250 MB) | Peak RSS scales with the active file (typically several times the file size) |

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
- Very large *individual* documents still require memory proportional to that file (PDF and HTML included)
- Discovery materializes the list of discovered files; duplicate tracking is proportional to file count
- Finding path lists in reports are bounded samples; full counts remain in evidence
- Multi-gigabyte *corpora* are processed incrementally and do not require holding all document text in RAM
- Local run history is opt-in (`--save`); there is no automatic cleanup yet

## Samyak by DataAIHub

| Project | Role |
| --- | --- |
| [DataAIHub](https://www.dataaihub.co) | Knowledge, ecosystem, research, guides, comparisons, and discovery |
| [DataAIHub Cookbook](https://github.com/tapansharma04/dataaihub-cookbook) | Practical, runnable examples for learning AI engineering patterns |
| **[Samyak](https://github.com/tapansharma04/dataaihub-ai-engineering-toolkit)** (this repository) | Open-source AI engineering product; current capability: Corpus Intelligence |

## Development

Requirements: Python 3.12+

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests
ruff format --check src tests
```

Tests require the package to be installed (editable is fine) so distribution metadata can be inspected.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
