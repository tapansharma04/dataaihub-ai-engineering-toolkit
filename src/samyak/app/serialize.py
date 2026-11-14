"""Internal debug serialization of an AppExtraction.

This is not the public AppReport JSON. CLI output uses
``samyak.app.report.render_app_report``, which never includes observations.
"""

from __future__ import annotations

import json

from samyak.app.pipeline import AppExtraction


def extraction_debug_json(extraction: AppExtraction) -> str:
    """Serialize the internal extraction (report envelope plus observations)."""
    return json.dumps(extraction.to_dict(), indent=2, sort_keys=False, ensure_ascii=False) + "\n"
