# Product overview

Retrieval-augmented generation systems answer questions using a knowledge base.
This document explains what RAG is at a high level and why corpus quality matters
before you spend time on embeddings, vector indexes, or prompt engineering.

## Why corpus quality matters

If the source documents are empty, duplicated, or poorly structured, retrieval
quality suffers regardless of which embedding model or vector database you choose.

## Key ideas

- Prefer curated, deduplicated source documents.
- Keep metadata such as title and source when available.
- Review unusually large files before applying naive chunking.
- Exclude documents that contain no meaningful text.

## Summary

Analyze the corpus first. Embedding comes later.
