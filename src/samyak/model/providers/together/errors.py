"""Together AI adapter errors. Not part of the public Samyak API."""

from __future__ import annotations

from samyak.model.errors import ModelCatalogError


class TogetherAdapterError(ModelCatalogError):
    """Failure while reading Together documentation into catalog records."""


class TogetherParseError(TogetherAdapterError):
    """Official Together documentation could not be parsed.

    Raised when a required identity field is missing or the document structure
    is not the labeled Markdown this adapter understands. Messages name the
    document, not parser internals.
    """


class TogetherFetchError(TogetherAdapterError):
    """Official Together documentation could not be retrieved or captured."""
