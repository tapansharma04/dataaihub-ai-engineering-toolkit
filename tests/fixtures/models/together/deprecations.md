<!-- official source URL: https://docs.together.ai/docs/deprecations.md -->
<!-- source_id: together-docs-deprecations -->
# Deprecations

Together AI's model lifecycle policy, including upgrades, redirects, and
deprecation schedules.

## Model upgrades (redirects)

An upgrade is a model release that is materially the same model lineage.
Outcome: The current endpoint redirects to the upgraded version. The old version
remains available via dedicated endpoints.

## Active model redirects

The following models are redirected to newer versions. Requests to the original
model ID are automatically routed to the upgraded version:

| Original model | Redirects to | Notes |
| --- | --- | --- |
| `deepseek-ai/DeepSeek-V3` | `deepseek-ai/DeepSeek-V3.1` | Same architecture, targeted improvements |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistralai/Ministral-3-14B-Instruct-2512` | Same lineage, upgraded version |
| `DeepSeek-V3` | `DeepSeek-V3.1` | Display names are not API model strings |

If you need to use the original model version, you can always deploy it as a
dedicated endpoint.

## Scheduled deprecations

The following models are deprecated and will be removed from serverless
inference on the date listed.

| Removal date | Model | Recommended replacement | Supported by on-demand dedicated endpoints |
| --- | --- | --- | --- |
| 2026-09-14 | `openai/gpt-oss-20b` | `Qwen/Qwen3.5-9B` | Yes |
| 2026-01-01 | `togethercomputer/aging-test` | `openai/gpt-oss-120b` | No |
| 2027-01-01 | `togethercomputer/future-test` | `openai/gpt-oss-120b` | No |
| 2026-10-01 | `togethercomputer/event-test` | `Qwen/Qwen3.5-9B` | Yes |
| 2026-11-01 | `togethercomputer/ambiguous-test` | `openai/gpt-oss-120b` | Yes |
| 2026-11-01 | `togethercomputer/ambiguous-test` | `Qwen/Qwen3.5-9B` | Yes |

## Deprecation history

### Inference

The table below lists all models removed from serverless inference, most recent first.

| Removal date | Model | Supported by on-demand dedicated endpoints |
| --- | --- | --- |
| 2026-08-27 | `nvidia/Nemotron-3-ultra-550b-a55b` | No |
| 2025-01-01 | `togethercomputer/event-test` | No |
| 2024-06-01 | `togethercomputer/historical-only` | No |
