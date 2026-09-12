# Deprecations

Synthetic OpenAI deprecation history for current-event reduction tests.

## Upcoming deprecations

### 2025-01-01: Listing-sensitive current

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2027-01-01 | `listing-sensitive` | `gpt-upcoming-repl` |

### 2026-01-01: Competing upcoming A

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2026-12-01 | `two-upcoming` | `repl-a` |

### 2026-06-01: Competing upcoming B

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2027-06-01 | `two-upcoming` | `repl-b` |

### 2026-03-01: Upcoming shutdown already elapsed

Still listed under Upcoming, but the shutdown date is on/before verified_at.
Listing selects this event; calendar then promotes it to retired.

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2026-08-01 | `upcoming-elapsed` | `gpt-elapsed-repl` |
| 2026-08-01 | `upcoming-elapsed-with-past` | `gpt-elapsed-repl` |

## Past deprecations

### 2024-01-01: Older past under an elapsed upcoming event

If elapsed upcoming rows fell through to Past, this replacement would win.

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2024-06-01 | `upcoming-elapsed-with-past` | `gpt-old-elapsed-repl` |

### 2026-08-01: Listing-sensitive historical

The past announcement is later than the upcoming announcement so a
latest-`deprecated_at` rule would pick this row. Current-event reduction must
use `deprecation_listing` instead.

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2026-08-15 | `listing-sensitive` | `gpt-past-repl` |

### 2024-08-29: First historical retirement

Later shutdown than the second past event. Independently taking the latest
`retirement_at` would mix this date with the later announcement's replacement.

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2025-12-01 | `two-past` | `gpt-old-repl` |

### 2025-09-26: Current historical retirement

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2025-10-28 | `two-past` | `gpt-current-repl` |

### 2025-01-01: Ambiguous past A

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2025-06-01 | `two-past-tied` | `tied-repl-a` |

### 2025-01-01: Ambiguous past B

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2025-08-01 | `two-past-tied` | `tied-repl-b` |
