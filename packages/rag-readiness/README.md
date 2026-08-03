# RAG Data Readiness Analyzer

Analyze a local document corpus **before** you embed or index it for a RAG system.

Many RAG problems originate in the source corpus — empty files, duplicates, tiny stubs, oversized dumps, noisy extraction — before retrieval, embeddings, or generation are involved.

`rag-readiness` inspects a directory of documents, collects evidence, explains why findings matter for RAG, and recommends practical next actions.

It identifies corpus characteristics that **may** affect a RAG system. It does **not** guarantee or predict downstream RAG quality.

## Features (v0.1)

- Corpus inventory (supported / unsupported / analyzed)
- Empty and whitespace-only document detection
- Very small / low-information document heuristics
- Very large document heuristics
- Exact duplicate detection (normalized content hashing)
- Lightweight text-quality / noise indicators
- Simple chunkability indicators (not a chunking engine)
- Human-readable CLI report and JSON output
- Fully local analysis — no network, no API keys, no telemetry

## Privacy / local processing

This utility:

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
| Other formats | Discovered and reported as unsupported (analysis continues) |

## Installation

From this monorepo (development / local install):

```bash
cd packages/rag-readiness
python -m pip install -e ".[dev]"
```

> **PyPI note:** The distribution name in `pyproject.toml` is currently the candidate
> `dataaihub-rag-readiness`. Verify name availability before publishing. This package
> is **not** claimed to be on PyPI until it is actually published.

The CLI command is:

```bash
rag-readiness
```

## Quick start

```bash
rag-readiness ./documents
```

JSON:

```bash
rag-readiness ./documents --format json
```

Write a report file:

```bash
rag-readiness ./documents --format json --output report.json
```

Help and version:

```bash
rag-readiness --help
rag-readiness --version
```

Try the bundled sample corpus:

```bash
rag-readiness examples/sample-corpus
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

Default thresholds are documented in code (`AnalysisConfig`) and included in JSON output under `config`. Override size thresholds via `--small-chars` / `--large-chars`.

## Example output (text)

```text
RAG Data Readiness Report
=========================

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

- `utility` / `version`
- `config` (thresholds used)
- `summary` (inventory and size statistics)
- `findings` (code, category, severity, message, why_it_matters, recommendation, evidence, affected_documents)
- `severity_counts`

Paths in reports are **relative to the corpus root** whenever possible.

Exit codes (v0.1):

- `0` — analysis completed
- non-zero — invalid path, I/O failure, or other runtime error

Findings (including `HIGH`) do **not** fail the process by themselves. Threshold-based CI failure may be added later if needed.

## Library usage

```python
from rag_readiness import analyze_corpus

report = analyze_corpus("./documents")
print(report.severity_counts())
for finding in report.findings:
    print(finding.severity, finding.title, finding.recommendation)
```

## Limitations

- Only `.txt` and `.md` are analyzed in v0.1
- No PDF/DOCX/HTML/OCR support yet
- No near-duplicate / semantic duplicate detection
- No embeddings, vector databases, or LLM judges
- Heuristics are conservative and may flag legitimate technical content for review
- No single “RAG readiness score” — findings are evidence-based, not oracles

## Validation

`rag-readiness` is validated with synthetic fixtures (unit tests) and public
information-retrieval / document corpora used during development.

Public datasets referenced for development and validation include:

- [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) (also packaged in [BEIR](https://github.com/beir-cellar/beir))
- [ArguAna](https://github.com/beir-cellar/beir) / [args.me](https://zenodo.org/records/3734893) (argument retrieval corpus; BEIR packaging)

These references identify public sources used for local validation. This repository
does **not** redistribute third-party dataset files.

## Future improvements

Improvements considered for this utility only:

- Additional document formats (for example PDF text layer)
- Richer metadata analysis
- Improved duplicate detection (near-duplicates)
- Configurable policy profiles
- Optional CI gates based on severity thresholds
- Additional corpus-quality checks with clear RAG rationale

## Development

Requirements: Python 3.12+

```bash
cd packages/rag-readiness
python -m pip install -e ".[dev]"
pytest
ruff check src tests
ruff format --check src tests
```

## License

MIT — see the repository root `LICENSE` file.
