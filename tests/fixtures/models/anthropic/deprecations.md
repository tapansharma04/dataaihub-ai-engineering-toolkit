# Model deprecations

This page lists all API deprecations, along with recommended replacements.

## Overview

Anthropic uses the following terms to describe the model lifecycle:

- Active: The model is fully supported and recommended for use.
- Legacy: The model will no longer receive updates and may be deprecated in the future.
- Deprecated: The model is still functional but no longer recommended. Anthropic provides a recommended replacement and assigns a retirement date.
- Retired: The model is no longer available for use. Requests to retired models will fail.

Claude Mythos Preview (`claude-mythos-preview`) is deprecated. To migrate, see the migration guide.

The dates on this page apply to Anthropic-operated platforms: the Claude API, Claude Platform on AWS, and Microsoft Foundry. Partner-operated platforms (Amazon Bedrock and Google Cloud) set their own retirement schedules.

## Model status

Current and recently retired models are listed in the following table with their status:

| API model name | Current state | Deprecated | Tentative retirement date |
| --- | --- | --- | --- |
| claude-fable-5-1 | Active | N/A | Not sooner than September 1, 2027 |
| claude-opus-5 | Active | N/A | Not sooner than July 24, 2027 |
| claude-sonnet-5 | Active | N/A | Not sooner than June 30, 2027 |
| claude-haiku-4-5-20251001 | Active | N/A | Not sooner than October 15, 2026 |
| claude-index-only | Active | N/A | Not sooner than January 1, 2028 |
| claude-legacy-test | Legacy | N/A | Not sooner than March 1, 2027 |
| claude-soon-deprecated | Deprecated | August 1, 2026 | December 1, 2026 |
| claude-elapsed-deprecated | Deprecated | January 1, 2026 | June 1, 2026 |
| claude-opus-4-1-20250805 | Retired | June 5, 2026 | August 5, 2026 |

## Deprecation history

All deprecations are listed in the following sections, with the most recent announcements first.

### 2026-06-05: Claude Opus 4.1 model

This model was retired August 5, 2026.

| Retirement date | Deprecated model | Recommended replacement |
| --- | --- | --- |
| August 5, 2026 | `claude-opus-4-1-20250805` | `claude-opus-4-8` |

### 2026-08-01: Claude Soon Deprecated model

| Retirement date | Deprecated model | Recommended replacement |
| --- | --- | --- |
| December 1, 2026 | `claude-soon-deprecated` | `claude-opus-5` |

### 2026-01-01: Claude Elapsed Deprecated model

| Retirement date | Deprecated model | Recommended replacement |
| --- | --- | --- |
| June 1, 2026 | `claude-elapsed-deprecated` | `claude-sonnet-5` |

### 2025-01-21: Claude 2 models

These models were retired July 21, 2025.

| Retirement date | Deprecated model | Recommended replacement |
| --- | --- | --- |
| July 21, 2025 | `claude-2.0` | `claude-opus-4-8` |

## API parameter deprecations

| Parameter | Status | Behavior | Recommended replacement |
| --- | --- | --- | --- |
| `temperature` | Deprecated (Claude Opus 4.7 and later) | Returns a 400 error | Omit |
