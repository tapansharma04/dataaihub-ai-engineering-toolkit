<!-- official source URL: https://ai.google.dev/gemini-api/docs/models.md.txt -->
<!-- source_id: google-docs-models-index -->
This guide introduces all the models available through the Gemini API.

## Gemini 3

### All Gemini 3 models

| Model | Endpoint |
|---|---|
| [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) | `gemini-3.8-flash` |
| [Nano Banana 2](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image) | `gemini-3.1-flash-image` |
| [Gemini 3.1 Pro](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview) | `gemini-3.1-pro-preview` |
| [Gemini 3.5 Transcribe](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-transcribe) | `gemini-3.5-transcribe` `gemini-3.5-transcribe-live` |
| [Index Only](https://ai.google.dev/gemini-api/docs/models/index-only) | `gemini-index-only` |
| [Sparse Test](https://ai.google.dev/gemini-api/docs/models/sparse-test) | `gemini-sparse-test` |
| [Dated Stable](https://ai.google.dev/gemini-api/docs/models/gemini-stable-dated) | `gemini-stable-dated` |
| [Alias Target](https://ai.google.dev/gemini-api/docs/models/gemini-alias-target) | `gemini-alias-target` |

## Generative media models

| Model | Description | Endpoint |
|---|---|---|
| [Imagen 4 (Deprecated)](https://ai.google.dev/gemini-api/docs/models/imagen) | Text-to-image model. | `imagen-4.0-generate-001` |

## Previous models

> [!WARNING]
> These models are [deprecated](https://ai.google.dev/gemini-api/docs/deprecations) and will be shut down soon; migrate to newer models to prevent service interruptions.

| Model | Description | Endpoint |
|---|---|---|
| [Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-2.0-flash) (Shut down) | Previous workhorse model. | `gemini-2.0-flash` |
| Gemini Previous Unlabeled | Historical listing without a lifecycle label. | `gemini-previous-unlabeled` |
| Gemini Previous Deprecated (Deprecated) | Previous model with an explicit deprecation label. | `gemini-previous-deprecated` |

## Model version name patterns

Gemini models are available in stable, preview, latest, or experimental versions.

### Latest

Points to the latest release for a specific model variation. This alias will get hot-swapped with every new release.

For example: `gemini-flash-latest`.

## Model deprecations

For information about model deprecations, visit the [Gemini deprecations](https://ai.google.dev/gemini-api/docs/deprecations) page.

A Vertex resource is not a Gemini API model: [Vertex](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models).
