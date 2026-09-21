"""Google Gemini API adapter errors. Not part of the public Samyak API."""

from __future__ import annotations

from samyak.model.errors import ModelCatalogError


class GoogleAdapterError(ModelCatalogError):
    """Failure while reading Google Gemini API documentation into catalog records."""


class GoogleParseError(GoogleAdapterError):
    """Official Google Gemini API documentation could not be parsed.

    Raised when a required identity field is missing or the document structure
    is not the labeled Markdown this adapter understands. Messages name the
    document, not parser internals.
    """


class GoogleFetchError(GoogleAdapterError):
    """Official Google Gemini API documentation could not be retrieved or captured."""
