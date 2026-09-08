"""Compare two saved corpus analysis runs.

Internal package — not part of the public Samyak API.

The analysis engine does not depend on this package. Comparison reads
reconstructed ``AnalysisReport`` snapshots from the local run store.
"""

from __future__ import annotations

from samyak.comparison.compare import compare_runs
from samyak.comparison.errors import IncompatibleRunsError
from samyak.comparison.models import RunComparison

__all__ = [
    "IncompatibleRunsError",
    "RunComparison",
    "compare_runs",
]
