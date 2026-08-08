"""Normalized content hashing for exact duplicate detection."""

from __future__ import annotations

import hashlib


def normalize_for_hash(text: str) -> str:
    """Normalize text for exact-duplicate comparison.

    Collapses whitespace runs (including newlines) to a single space so
    trivial formatting-only differences still count as duplicates.
    """
    return " ".join(text.split())


def content_hash_normalized(normalized: str) -> str:
    """SHA-256 hex digest of an already-normalized string."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """Return SHA-256 of whitespace-normalized text.

    Semantically equivalent to hashing ``" ".join(text.split())``, but streams
    tokens into the hasher so a second full-size normalized copy is not created.
    """
    hasher = hashlib.sha256()
    n = len(text)
    i = 0
    first = True
    while True:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i + 1
        while j < n and not text[j].isspace():
            j += 1
        if not first:
            hasher.update(b" ")
        else:
            first = False
        hasher.update(text[i:j].encode("utf-8"))
        i = j
    return hasher.hexdigest()


def normalized_is_empty(text: str) -> bool:
    """Return True when ``normalize_for_hash(text)`` would be empty."""
    return not any(not ch.isspace() for ch in text)
