"""Internal comparison result types.

These are not part of the public Samyak API.
"""

from __future__ import annotations

from dataclasses import dataclass

from samyak.corpus.models import Finding
from samyak.store.models import StoredRun

STATUS_NEW = "new"
STATUS_NO_LONGER_DETECTED = "no_longer_detected"
STATUS_CHANGED = "changed"
STATUS_UNCHANGED = "unchanged"

CORPUS_SAME_LOCATION = "same_location"
CORPUS_CROSS = "cross_corpus"
CORPUS_SAME_NAME_UNCONFIRMED = "same_name_unconfirmed"
CORPUS_DIFFERENT_NAME_UNCONFIRMED = "different_name_unconfirmed"


@dataclass(frozen=True, slots=True)
class MetricChange:
    """A numeric inventory field compared across two runs."""

    label: str
    before: int
    after: int

    @property
    def changed(self) -> bool:
        return self.before != self.after


@dataclass(frozen=True, slots=True)
class ConfigChange:
    """One analysis-configuration field that differs between baseline and current."""

    key: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class FieldChange:
    """One finding field that differs between baseline and current."""

    field: str
    before: str
    after: str
    added_paths: tuple[str, ...] = ()
    removed_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FindingMatch:
    """A finding identified by ``code``, placed relative to the baseline run."""

    code: str
    status: str
    before: Finding | None
    after: Finding | None
    changes: tuple[FieldChange, ...] = ()

    @property
    def title(self) -> str:
        finding = self.after or self.before
        return finding.title if finding is not None else self.code

    @property
    def severity(self) -> str | None:
        finding = self.after or self.before
        return finding.severity.value if finding is not None else None


@dataclass(frozen=True, slots=True)
class CorpusRelation:
    """How the two runs relate as saved corpus locations."""

    kind: str
    left_label: str | None
    right_label: str | None
    left_path: str | None
    right_path: str | None


@dataclass(frozen=True, slots=True)
class RunComparison:
    """Baseline run versus current run. Deltas are baseline → current."""

    baseline: StoredRun
    current: StoredRun
    corpus: CorpusRelation
    version_note: str | None
    config_changes: tuple[ConfigChange, ...]
    metrics: tuple[MetricChange, ...]
    new: tuple[FindingMatch, ...]
    no_longer_detected: tuple[FindingMatch, ...]
    changed: tuple[FindingMatch, ...]
    unchanged: tuple[FindingMatch, ...]

    @property
    def config_changed(self) -> bool:
        return bool(self.config_changes)

    @property
    def finding_counts(self) -> tuple[int, int, int, int]:
        return (
            len(self.new),
            len(self.no_longer_detected),
            len(self.changed),
            len(self.unchanged),
        )
