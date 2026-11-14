<!-- official source URL: https://docs.together.ai/docs/deprecations.md -->
<!-- source_id: together-docs-deprecations -->
<!-- Excerpt from the 30 October 2026 archived deprecations page. Removal-date cells are unchanged. -->
# Deprecations

## Scheduled deprecations

The following models are deprecated and will be removed from serverless inference on the date listed. Migrate to the recommended replacement or a [dedicated endpoint](https://docs.together.ai/docs/dedicated-endpoints.md) before that date.

| Removal date | Model | Recommended replacement | Supported by on-demand dedicated endpoints |
| :- | :- | :- | :- |
| 2026-09-14 | `openai/gpt-oss-20b` | `Qwen/Qwen3.5-9B` | Yes |
| 2026-09-14 | `google/gemma-4-31B-it` | `zai-org/GLM-5.3-Flash` | Yes |
| 2026-09-14 | `thinkingmachines/Inkling-Small` | `zai-org/GLM-5.3-Flash` | Yes |
| 2026-09-14 | `intfloat/multilingual-e5-large-instruct` | | No |

## Deprecation history

### Inference

The table below lists all models removed from serverless inference, most recent first.

| Removal date | Model | Supported by on-demand dedicated endpoints |
| :- | :- | :- |
| 2026-09-15 | `openai/gpt-oss-20b` | Yes |
| 2026-09-15 | `google/gemma-4-31B-it` | Yes |
| 2026-09-15 | `thinkingmachines/Inkling-Small` | Yes |
| 2026-09-15 | `intfloat/multilingual-e5-large-instruct` | No |
| 2026-08-19 | `black-forest-labs/FLUX.1-schnell` | No |
| 2026-08-19 | `Qwen/Qwen2.5-7B` | No |
| 2026-08-19 | `moonshotai/Kimi-K2.6` | Yes |
| 2025-08-28 | `Qwen/Qwen2-72B-Instruct` | No |

### Fine-tuning

The table below lists all models removed from the fine-tuning service, most recent first. These models can no longer be used as a base model for a fine-tuning job. Where a close equivalent exists, the suggested replacement is listed.

| Removal date | Model | Suggested replacement |
| :- | :- | :- |
| 2026-07-29 | `Qwen/Qwen2-72B-Instruct` | `meta-llama/Llama-3.3-70B-Instruct-Reference` |
