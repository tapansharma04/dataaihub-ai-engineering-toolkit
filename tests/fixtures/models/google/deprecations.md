<!-- official source URL: https://ai.google.dev/gemini-api/docs/deprecations.md.txt -->
<!-- source_id: google-docs-deprecations -->
<br />

This page lists the known deprecation schedules for stable (GA) and preview
models in the Gemini API. A "**deprecation**" is the announcement that we
no longer provide support for a model, and that it will be "**shut down**" in
the near future. Once a model is "**shutdown**", it is completely
turned off, and the endpoint is no longer available.

> [!NOTE]
> **Note:** The **shutdown dates listed in the table indicate the *earliest possible
> dates* on which a model might be retired**. We will communicate the exact shutdown date to users with advance notice to ensure a smooth transition to a replacement model.

## Gemini 3 models

| **Model** | **Release date** | **Shutdown date** | **Recommended replacement** |
|---|---|---|---|
| `gemini-3.8-flash` | September 2, 2026 | No shutdown date announced |   |
| `gemini-stable-dated` | May 7, 2026 | May 7, 2027 | `gemini-3.8-flash` |
| Preview models ||||
| `gemini-3.1-pro-preview` | February 19, 2026 | No shutdown date announced |   |
| `gemini-old-preview` | November 18, 2025 | March 9, 2026 | `gemini-3.1-pro-preview` |

## Gemini 2.0 models

| **Model** | **Release date** | **Shutdown date** | **Recommended replacement** |
|---|---|---|---|
| `gemini-2.0-flash` | February 5, 2025 | June 1, 2026 | `gemini-3.8-flash` |
| `gemini-2.0-flash-001` | February 5, 2025 | June 1, 2026 | `gemini-3.8-flash` |

## Imagen models

| **Model** | **Release date** | **Shutdown date** | **Recommended replacement** |
|---|---|---|---|
| `imagen-4.0-generate-001` | June 24, 2025 | August 17, 2026 | `gemini-3.1-flash-image` |

## Veo models

| **Model** | **Release date** | **Shutdown date** | **Recommended replacement** |
|---|---|---|---|
| `veo-3.0-generate-001` | September 9, 2025 | June 30, 2026 | `veo-3.1-generate-preview` or the GA models on the [Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/veo) |

## Gemini Omni Flash models

| **Model** | **Release date** | **Shutdown date** | **Recommended replacement** |
|---|---|---|---|
| Deprecated models ||||
| `gemini-omni-flash-preview` | June 30, 2026 | September 30, 2026 | `gemini-omni-1.1-flash` |
