# Model Intelligence export contract

This v0.4.0 contract is not part of the published Samyak `v0.3.0` release
(`ffd0d8ceeba2fca6d65560b609c1fc7ea6e1fb47`). It describes the JSON artifact
produced by `samyak model export`. DataAIHub imports that file. It does not
import `samyak.model` and does not run Samyak while serving the tracker.

The wrapper has `export_schema_version` `1`. The embedded catalog keeps
`catalog_schema_version` `1`, and an embedded history object keeps
`lifecycle_history_schema_version` `1`. The wrapper does not redefine those
schemas.

## Stable wrapper fields

| Field | Type | Meaning |
| --- | --- | --- |
| `product` | string | Always `samyak`. |
| `capability` | string | Always `model_export`. This is the wrapper, not the catalog's `model` capability. |
| `export_schema_version` | integer | `1` for this contract. Reject any other value. |
| `exporter_version` | string | Samyak package version that wrote the file. It is not a schema version. |
| `publication.status` | string | `complete` or `incomplete`. |
| `publication.reasons` | array of strings | Empty when complete. Sorted unique reason codes when incomplete. |
| `coverage` | array of objects | One row per supported provider, then any extra provider ids sorted. |
| `catalog` | object | Exact catalog schema v1 document. |
| `history` | object or `null` | Exact history schema v1 document, or `null` when no history file existed. |

`coverage` row fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `provider_id` | string | Provider id. |
| `present` | boolean | The catalog contains at least one model for that provider. |
| `refreshed_with_snapshot` | boolean | Every provenanced fact for that provider has `verified_at` equal to `catalog.freshness.generated_at`. |
| `optional_source_failures` | array of strings | Notice codes for optional-source failures attributed to that provider. |
| `coverage_notices` | array of strings | Other provider notices, including `DOCUMENTED_SUBSET:<provider>`. |

A `DOCUMENTED_SUBSET:` notice records that the provider published a subset.
It is coverage information. It is not an optional-source failure and it does
not make the export incomplete.

## Publication

`complete` means every provider in the exporter's supported set is present,
no optional-source failure notice is present, and every provenanced fact's
`verified_at` equals `catalog.freshness.generated_at`. `not_verified` facts
have no provenance and do not block publication. For a `conflict` fact, every
claim must carry that same `verified_at`.

Reason codes:

- `provider_missing:<provider_id>` — that supported provider has no model row.
  Absence is not reported as a source failure.
- `optional_source_failed:<notice_code>` — the catalog notice is
  `PARTIAL_MODEL_PAGES`, `PARTIAL_MODEL_PAGES:<provider>`,
  `SOURCE_FETCH_FAILED:...`, or `SOURCE_PARSE_FAILED:...`.
- `facts_not_from_this_snapshot:<provider_id>` — a provenanced fact for that
  provider was not verified at `catalog.freshness.generated_at`.
- `history_write_failed` — the catalog has a history-write marker. The code
  is `HISTORY_WRITE_FAILED` or `HISTORY_WRITE_FAILED:<provider_id>:<observed_at>`.
  The timestamp is the remainder after the provider field and may itself
  contain colons. Each marker names one catalog update whose lifecycle events
  were not appended. The catalog commit for that update succeeded. The history
  file may be absent, empty, or missing those events. This is not the same as
  `history: null`. An absent history file without a marker is an intentionally
  missing log, and an existing history object with `"events": []` is an empty
  log.

A marker stays on the catalog through later provider updates, process
restarts, and export. A successful history append for the same or another
provider does not remove it, including a later provider in `update all`.
The omitted events are not retained. Comparing the current catalog with an
older snapshot does not recover them, so publication stays incomplete while
any marker is present. Export adds `history_write_failed` once, however many
markers are stored.

`catalog.freshness.status` is separate. `as_of` means the freshness record
was written from the snapshot clock. It does not mean the export is
publishable. Consumers must use `publication.status` and `publication.reasons`.

`--require-complete` is a producer option. It exits `2` without writing JSON
when publication is incomplete, and it leaves an existing output file
unchanged. A file written without that flag can still be incomplete. The
importer must reject `incomplete` itself.

## Null, missing, and unknown

- `history: null` means the history file was absent. It is not an empty log.
- A history object with `"events": []` means the file existed and contained no
  events. The first successful snapshot for a provider writes no events, so an
  empty log does not mean the catalog is empty.
- JSON `null` on a fact `value` means that fact status has no single value.
  It does not mean the value is false, zero, or unsupported.
- Fact status `unknown` means the source was checked and did not state the
  fact. `not_applicable` means the fact does not apply to that model.
  `not_verified` means Samyak did not verify it. `conflict` means eligible
  sources disagreed, as defined in "How a fact value is selected". The values
  are in `claims`, and `value` is null.
- `observed_at` in catalog provenance is null. `verified_at` is the Samyak
  refresh clock, not the provider's publication time.
- A content hash is null when the fact was derived rather than copied from
  one source document.

## Catalog facts a consumer must read

Identity is `provider_id:provider_model_id`, split on the first colon. The
serving provider is `provider_id`. That id is not the organization that
trained the model.

`identity_kind` is `canonical` or `alias`. An alias is its own model row.
`resolves_to.value` points at the canonical identity. A canonical row does
not use `resolves_to` as its own id.

Lifecycle `value` is `active`, `legacy`, `deprecated`, or `retired`. `legacy`
is an Anthropic-specific lifecycle value. `retirement_at` is the documented
removal date. There is no separate shutdown field. A deprecated model may
still have `deprecated_at` as `not_verified`.

`replacement` is one catalog fact. A known value is a non-empty array of
identity objects, each with `provider_id`, `provider_model_id`, and
`samyak_id`. More than one object means the source documented more than one
successor; the array does not select a single one. `unknown` keeps `value`
null and provenance, and it is not an empty array. `conflict` keeps `value`
null and stores each competing array of identity objects on `claims`, with
that claim's provenance.

## How a fact value is selected

Export schema 1 is unchanged. The export embeds the catalog after selection.
It does not add observations the normalizer discarded.

Each of the five provider normalizers groups observations by model identity
and then selects each fact on its own. The steps below are the current
selection rule. Numeric ranks are not catalog fields and are not export
fields. Which document class has which rank is provider-specific.

1. An observation counts for a fact only when it states a value. A field left
   unset is omitted before ranking. That omission is not a disagreement, and
   it is not the fact status `unknown`, `not_verified`, or `not_applicable`.
2. Among observations that state the fact, only the highest source rank is
   eligible. A lower-ranked observation is ignored for the selected value and
   for conflict detection.
3. Equal values at that winning rank collapse to one value. Provenance is the
   first of those agreeing observations in normalizer input order. A later
   observation with the same value is not stored.
4. Distinct values at that winning rank become `status: conflict`. `value` is
   null. Each distinct value becomes one claim, with the provenance of the
   first observation that stated it. Another observation with a value already
   present does not add a claim.

`observed_sources` is derived from the provenance that remains on the model's
facts, in fact-field order. A source is listed when its id remains on a fact:
the provenance of a `known` or `unknown` fact, or the provenance of a
`conflict` claim. A fetched document can appear in
`freshness.source_retrieved_at` and still be absent from `observed_sources`.
The list does not record every parsed observation, and it does not record a
source that only repeated the selected value.

A `known` fact has one selected value and that value's provenance. It does
not mean every retrieved source stated the same value. Three observation
roles are distinct, and only the selected role is stored:

| Role | What it is | Stored on the fact |
| --- | --- | --- |
| Selected | The winning-rank observation that supplies a `known` value, or each claim when the fact is `conflict` | Yes |
| Supporting | Another winning-rank observation that stated the same value | No |
| Overridden | A lower-ranked observation that stated a value | No |

An overridden observation is absent from the model row and from the embedded
catalog in the export, whether or not its value matched the selected value.

Illustration of an override, using Together's current ranks. Deprecations
are rank 3 and state retirement date `2026-07-29`. The changelog is rank 1
and states `2025-08-28`. The fact is `known`, the value is `2026-07-29`, and
the provenance is the deprecations document. The changelog date is not a
claim. The changelog source is not added to `observed_sources` by this fact.
If both dates had come from rank 3, the fact would instead be `conflict`,
with one claim for each date. This illustration is the selection rule, not a
claim that a particular catalog row has those two dates.

Illustration of agreement at the winning rank. Two rank-2 documents both
state API access `true`. The fact is `known` and `true`. Provenance is the
earlier document. The later document is not stored on the fact and is not
added to `observed_sources` because of that agreement.

## History

History is a change log. Current lifecycle is the catalog fact.

`change_type: "removed"` means the provider page no longer lists that model.
It does not set lifecycle to `retired`. A model can be removed from a later
catalog without a retirement date, and a `retired` lifecycle value can exist
while the row is still in the catalog.

History snapshots omit provenance. They cannot be treated as a second copy of
the catalog fact. `previous_replacement` and `current_replacement` are
snapshots or null. A known snapshot `value` is a list of `samyak_id` strings,
not identity objects. A conflict snapshot keeps `value` null, and each entry
in `claims` is one of those lists. `unknown` keeps `value` null and `claims`
empty.

## Minimum validation

An importer can do all of this with a JSON parser:

1. Reject unless `product` is `samyak`, `capability` is `model_export`, and
   `export_schema_version` is `1`.
2. Reject unless `publication.status` is `complete` and `publication.reasons`
   is empty. Do not publish an incomplete artifact.
3. Reject unless `catalog.catalog_schema_version` is `1` and
   `catalog.capability` is `model`.
4. If `history` is null, treat history as absent. If it is an object, reject
   unless `lifecycle_history_schema_version` is `1` and `capability` is
   `model`. Do not treat either case as the current lifecycle.
5. Read lifecycle, aliases, conflicts, and provenance from the catalog.
   Display a conflict's claims rather than a single value.
6. Do not infer a shutdown date that the catalog does not contain.
7. Do not treat `change_type` `removed` as lifecycle `retired`.
8. Compare `verified_at` with `catalog.freshness.generated_at` only to
   understand the publication gate. Do not replace `publication.status` with
   `freshness.status`.
9. Read a `known` fact as the selected value. Do not treat it as agreement
   from every source in `freshness.source_retrieved_at`.

## Not stable

Python class names, CLI help text, notice `message` strings, provider parser
internals, and the order of conflict claims beyond preserving the catalog's
own order are implementation details.

Output-path checks resolve `..`, symlinks, and `~` before refusing to
overwrite `catalog.json`, `catalog.json.bak`, or `lifecycle-history.json`.
Hard links to those files are not detected. An unresolvable output path is
refused with the same error as a protected path.
