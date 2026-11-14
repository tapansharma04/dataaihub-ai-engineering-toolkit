# Samyak

AI engineering tooling for building reliable AI applications.

**Samyak is an open-source project by [DataAIHub](https://www.dataaihub.co).**

The current release provides **Corpus Intelligence** for analyzing document collections before they are used in AI and RAG systems, and **Model Intelligence** for a local catalog of documented API models from official provider documentation.

## Corpus Intelligence

Analyze your document corpus before it becomes a RAG problem.

Many AI/RAG issues start in the source corpus — empty files, duplicates, tiny stubs, oversized dumps, noisy extraction — before retrieval, embeddings, or generation are involved.

Corpus Intelligence inspects a local document directory, collects evidence, explains why findings matter, and recommends practical next actions. It identifies characteristics that **may** affect downstream systems. It does **not** guarantee or predict retrieval or answer quality.

## Model Intelligence

Keep a local catalog of provider-documented API models, verified from official documentation rather than guessed from marketing pages or live API probes.

v0.4.0 covers these serving sources: **anthropic**, **fireworks**, **google**, **openai**, and **together**. Refresh one provider, or refresh all of them sequentially with `samyak model update all`. Other providers' records stay in the overlay.

```bash
samyak model update openai
samyak model update all
samyak view
```

`samyak model update` is Samyak's **only network command**. It fetches official documentation over HTTPS and writes `models/catalog.json` under the local Samyak cache. No API key, account, or provider SDK is required. Browse the catalog in `samyak view` afterward; the viewer itself stays on loopback and does not refresh the catalog.

There is no `samyak model list` or `samyak model show`. Model Intelligence is not a pricing, ranking, or recommendation product.

## Offline Model Catalog and Comparison

Acquire provider documentation into the local catalog, then start the loopback-only viewer:

```bash
samyak model update all
samyak view
```

`samyak model update` is the supported acquisition command; it is the only
networked model workflow. `samyak view` opens the local workspace, where the
**Model Catalog** is available after acquisition. The viewer uses the saved
catalog only: it does not refresh providers or send catalog data elsewhere.

In the catalog, select two or three model checkboxes to open a comparison. The
comparison preserves the order in which models were selected. Remove an
individual selection with its checkbox, or use **Clear selection** to reset the
comparison. The three-model maximum is enforced in the browser. Explorer
filters are client-side and can hide catalog rows, but they do not discard an
existing comparison selection.

Comparisons are strict by default: only `current_documented` records are
eligible. Enable the explicit legacy or historical documentation options only
when you need to inspect those records; an opt-in does not certify current
availability. Each comparison includes provider and documentation-disposition
rows, stable model IDs, display names, and detail links.

Facts retain their recorded evidence. Expand **Evidence** to inspect source
identifiers and verification timestamps. Conflicting facts remain separate
claims with their own values and sources; Samyak does not resolve them. Context
windows are directly comparable only when their verified measurement kinds are
compatible. An unknown availability scope is not a claim about account,
region, deployment, or API access. On narrow screens, scroll the comparison
workspace horizontally to inspect all selected columns.

## Who it is for

- AI / ML engineers preparing document corpora for RAG or other retrieval workflows
- Engineers choosing among documented API models from Anthropic, Fireworks, Google, OpenAI, or Together
- Data engineers validating knowledge-base handoffs
- Platform and QA teams adding corpus checks to local workflows or CI

## Features (v0.4.0)

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
- Corpus analysis is fully local — no network, no API keys, no telemetry
- Optional local run history (`--save`) and loopback viewer (`samyak view`)
- Baseline → current comparison of two saved runs
- Local model catalog from official Anthropic, Fireworks, Google, OpenAI, and Together documentation
- `samyak model update` for one serving source, or `samyak model update all` (the only network commands; no API key)
- Offline model-catalog browsing in `samyak view` after a refresh
- `samyak model history` to inspect previously observed documented lifecycle changes (local; no network)
- `samyak model query` for evidence-aware local capability queries over the saved model catalog

## Privacy / local processing

Samyak:

- processes documents on your machine
- sends **no** document content externally
- includes **no** telemetry
- requires **no** account and **no** API key
- runs Corpus Intelligence and `samyak view` with **no** network or API calls
- uses the network **only** for `samyak model update`, which fetches official provider documentation and writes a local catalog overlay

Treat corpus files as untrusted input: contents are never executed or interpreted as instructions.

Saved runs (when you pass `--save`) stay in a local cache directory on this machine. Samyak does not upload them. Run-history metadata may include the local corpus directory path so you can tell identically named folders apart; that path is not added to analysis reports.

## Local Run History

Run History currently lets you save previous analysis runs, reopen them, and compare two saved runs. Comparison is available in the local viewer; it is not a hosted service.

`--save` is **opt-in**. Without it, `samyak corpus PATH` behaves as in v0.1: text, JSON, and HTML reports are unchanged, and nothing is written to the run cache.

```bash
samyak corpus ./documents --save
samyak view
```

`samyak view` starts a local HTTP server on **127.0.0.1** (default port **15500**) and prints a URL such as `http://127.0.0.1:15500/`. The home page is a local workspace: saved corpus runs and, after a model refresh, the Model Intelligence catalog. The dashboard lists recent saved runs (corpus **basename**, timestamp, run ID, counts, and status). Opening a run shows that run’s metadata — corpus name and the local **corpus path** as separate fields — plus the existing Samyak HTML report, reconstructed from stored JSON. You do not need the original corpus files to still exist, and you do not need to rerun analysis.

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

The model catalog overlay is `models/catalog.json` in the same cache directory. `catalog.json.bak` is a best-effort previous copy and is not the live catalog.

The viewer binds to loopback only and makes no network requests outside this machine. Nothing is uploaded. Refreshing the model catalog requires `samyak model update`, not the viewer.

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

Refresh the local model catalog (uses the network; no API key):

```bash
samyak model update openai
samyak model update all
samyak view
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
samyak model --help
samyak model update --help
samyak model history --help
samyak model query --help
samyak --version
```

### Model capability queries

`samyak model query` is read-only and uses verified facts from the local
catalog. By default it includes only records explicitly marked
`current_documented`; use `--include-legacy` to additionally inspect legacy
records, or repeat `--documentation` to include a named historical category.
Those options do not certify current availability.

```bash
samyak model query --tool-calling true
samyak model query --input-modality text --input-modality image --format json
samyak model query --min-context 100000 --context-kind input
```

Add `--explain` to show evidence-safe predicate explanations. JSON output gains
an additive `explanations` field only with that flag. Exclusion reason counts
can overlap: an excluded record may fail more than one predicate. Missing
verified evidence is not a verified negative, and zero verified matches does
not prove that no model is capable or currently available to an account.

Unknown, unverified, and conflicting facts do not match either boolean value.
Context limits are compared only within the explicitly selected kind; combined
and input windows are not converted. A matching capability remains evidence
about documented provider behavior, not a guarantee for a particular account,
region, or deployment surface.

### Model comparisons

Compare two or three explicit stable model IDs from the saved catalog:

```bash
samyak model compare google:gemini-2.5-flash google:gemini-2.5-pro
samyak model compare google:gemini-2.5-flash anthropic:claude-fable-5 fireworks:accounts/fireworks/models/kimi-k2p5 --format json
```

Comparison is strict-current-documentation by default; use `--include-legacy`
or repeated `--documentation` only to inspect a named uncertified category.
Cells retain verified yes/no, unknown, not-verified, and conflicting evidence
states with provenance. Context windows are directly comparable only when their
verified kinds match; the command does not choose a best model.

### Model catalog capability explorer

The local Model Catalog page includes a collapsed capability explorer. It
refines only the rows selected by the page's URL search, provider, and
lifecycle filters; it never expands that server-rendered universe. The browser
uses each record's stable `samyak_id` to map matching facts back to a rendered
row, so display names and row order do not affect filtering.

The default documentation selection is `current_documented`. Legacy,
historical-reference, and not-observed records require an explicit checkbox.
Capability filters match only verified known facts: unknown, unverified, and
conflicting facts match neither `true` nor `false`. Multiple input or output
modalities are conjunctive. A context threshold also requires its matching
context kind; kinds are never converted.

The match count is reported against the documentation-eligible records in the
already URL-filtered scope. Clear restores current-documentation-only browser
filters without changing URL filters. The page and its embedded fact index work
offline; they reflect the saved catalog's freshness and do not fetch providers.

The explorer's **Why this result** panel can inspect one model from that
server-scoped page at a time. It displays the saved fact state, source IDs,
verification timestamps, and a link to its local evidence details. The browser
does not fetch providers or infer missing capabilities; a zero verified match
is explicitly not evidence that no capable model exists.

### Application Readiness

`samyak app check PATH` inspects a local application project for model
references, AI operations whose model identity cannot be established, and
lifecycle findings from the saved Model Intelligence catalog. It analyzes
Python, JSON, and TOML files only; generated, vendored, virtual-environment,
and version-control directories are skipped.

```bash
samyak app check ./my-ai-service
samyak app check ./my-ai-service --format json --output app-readiness.json
```

The supported formats are `text` and `json`. The command reads source files
one at a time and reads the local catalog when it is present. It is offline and
read-only: it does not execute application code, inspect provider accounts,
call provider APIs, refresh the catalog, or inspect unsupported file types.
Binary, oversized, unparsable, and out-of-scope files are reported as notices
or skipped; they are not silently treated as evidence.

Catalog matches can be established, resolved, ambiguous, or absent. A finding
about lifecycle reflects the saved catalog evidence, not a guarantee about an
account, region, deployment, runtime path, or application readiness. Finding
counts are inventory, not a readiness score.

### Model export

`samyak model export` writes a versioned JSON wrapper around the local catalog
and, when present, its lifecycle history. It is read-only and offline: it does
not refresh or modify either local file.

```bash
samyak model export > model-export.json
samyak model export --output model-export.json
samyak model export --require-complete --output model-export.json
```

JSON is the only export format. Without `--output`, JSON is written to stdout;
with `--output`, the destination is overwritten, except that Samyak refuses to
replace its catalog or lifecycle-history files. `--require-complete` exits with
status 2 and writes nothing when the snapshot is not complete enough for
publication.

The wrapper has a versioned contract (`export_schema_version`), carries the
catalog's documentation dispositions and fact provenance, and preserves
unknown, unverified, and conflicting facts rather than resolving them. A
conflicting fact retains its individual claims and their provenance. Consumers
must use `publication.status` and `publication.reasons`; a saved catalog is not
by itself a claim that all provider coverage is complete. See
[`docs/model-export-contract.md`](docs/model-export-contract.md) for the
consumer contract.

### Archived-evidence diagnostic

`samyak model diagnose-overrides` is an advanced, read-only diagnostic for
completed source-archive generations created with `samyak model update
--archive-sources`. It re-parses selected archived evidence and reports
supporting observations, source-rank overrides, and conflicts. It never
refreshes providers or writes the catalog, history, or archive. It is for
reproducibility investigation, not normal catalog use or publication.

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
Version:                  0.4.0

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
  "version": "0.4.0"
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

- `0` — analysis completed (findings do not change the exit code); model catalog update committed
- `2` — invalid path, invalid configuration, missing subcommand, I/O failure writing `--output`, unknown model provider, or model catalog update failed

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
- Model Intelligence currently covers official Anthropic, Fireworks, Google (Gemini API), OpenAI, and Together documentation — not partner platforms, other serving sources, pricing, or recommendations
- Model facts are taken from captured documentation, not from probing provider APIs
- A provider refresh replaces that provider's records; it does not invent missing detail-page facts when a model page cannot be retrieved

## Samyak by DataAIHub

| Project | Role |
| --- | --- |
| [DataAIHub](https://www.dataaihub.co) | Knowledge, ecosystem, research, guides, comparisons, and discovery |
| [DataAIHub Cookbook](https://github.com/tapansharma04/dataaihub-cookbook) | Practical, runnable examples for learning AI engineering patterns |
| **[Samyak](https://github.com/tapansharma04/dataaihub-ai-engineering-toolkit)** (this repository) | Open-source AI engineering product; current capabilities: Corpus Intelligence and Model Intelligence |

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
