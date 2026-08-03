# DataAIHub AI Engineering Toolkit

Open-source utilities for building, testing, diagnosing, evaluating, and operating production AI systems.

Part of the [DataAIHub](https://www.dataaihub.co) ecosystem.

## The DataAIHub ecosystem

| Project | Role |
| --- | --- |
| [DataAIHub](https://www.dataaihub.co) | Knowledge, ecosystem, research, guides, comparisons, and discovery |
| [DataAIHub Cookbook](https://github.com/tsharma82/dataaihub-cookbook) | Practical, runnable examples for learning how to build AI engineering systems and patterns |
| **DataAIHub AI Engineering Toolkit** (this repository) | Production-oriented utilities that help engineers analyze, test, diagnose, validate, and operate AI systems |

This toolkit is **not** a tutorial collection. Guides belong on DataAIHub; examples belong in the Cookbook; this repository ships tools you can run against real systems and artifacts.

## Utilities

### RAG Data Readiness Analyzer

Analyze a local document corpus before embedding or indexing for a RAG system. Identifies empty documents, duplicates, size outliers, lightweight text-quality issues, and simple chunkability concerns — entirely offline.

- Package: [`packages/rag-readiness`](packages/rag-readiness)
- Status: **v0.1.0** available for local install from this repository (not published to PyPI yet)
- CLI: `rag-readiness`

```bash
cd packages/rag-readiness
python -m pip install -e .
rag-readiness ./documents
```

As additional utilities are released, they will be listed here.

## Repository layout

This repository is a monorepo for independently publishable utilities. Each utility lives under `packages/` with its own packaging metadata, tests, and docs, and can be versioned and released on its own schedule.

Utilities may be implemented in different languages over time (for example Python packages for PyPI, or JavaScript/TypeScript packages for npm). Packaging and tooling are added only when a concrete utility needs them.

## Design principles

- **Quality over quantity.** Prefer a small number of high-quality utilities over many shallow ones.
- Solve real production AI engineering problems.
- Be useful independently of DataAIHub.
- Prefer deterministic, local analysis when an LLM or paid API is unnecessary.
- Be automation- and CI-friendly where appropriate.
- Be explainable, testable, and production-oriented.
- Prefer local/private analysis for enterprise data whenever practical.

## Contributing

Contribution guidelines will be published as the first utilities land. Issues and pull requests that improve released utilities, documentation, tests, and examples are welcome.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
