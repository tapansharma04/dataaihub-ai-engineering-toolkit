<!-- official source URL: https://docs.together.ai/docs/serverless/models.md -->
<!-- source_id: together-docs-serverless-models -->
# Available models

Serverless models are the fastest way to run inference on Together. This fixture
covers the public Serverless Models catalogue only.

Dedicated endpoints use `togethercomputer/dedicated-only-model`. Private uploads
such as `customer/private-finetune` are not serverless.

## Chat models

| Organization | Model name | API model string | Context length | Input pricing (per 1M tokens) | Quantization | Function calling | Structured outputs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI | GPT-OSS 120B | `openai/gpt-oss-120b` | 131072 | $0.15 | MXFP4 | Yes | Yes |
| Qwen | Qwen3.5 9B | `Qwen/Qwen3.5-9B` | 262144 | $0.17 | FP8 | Yes | Yes |
| Meta | Llama 3.3 70B Instruct Turbo | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | 131072 | $1.04 | FP8 | Yes | Yes |
| Google | Gemma 4 31B | `google/gemma-4-31B-it` | 131072 | $0.20 | - | - | - |
| Qwen | Qwen3.7 Max | `Qwen/Qwen3.7-Max` | - | $1.25 | - | - | - |
| Qwen | Qwen comma context | `Qwen/Qwen-comma` | 131,072 | $0.10 | - | Yes | Yes |
| Qwen | Qwen3.8 Flash | `Qwen/Qwen3.8-Flash` | 1000000 | $0.20 | FP8 | Yes | Yes |
| OpenAI | GPT-OSS 20B | `openai/gpt-oss-20b` | 131072 | $0.05 | MXFP4 | Yes | Yes |
| DeepSeek | DeepSeek V3.1 | `deepseek-ai/DeepSeek-V3.1` | 163840 | $0.60 | FP8 | Yes | Yes |
| Sibling | Sibling A | `sibling-org/sibling-a` | 8192 | $0.10 | - | Yes | Yes |
| Sibling | Sibling B | `sibling-org/sibling-b` | 8192 | $0.10 | - | - | - |
| Z.ai | GLM-5.2 | `zai-org/GLM-5.2` | 512000 | $1.40 | FP4 | Yes | Yes |
| Qwen | Qwen K suffix | `Qwen/Qwen-ksuffix` | 128K | $0.10 | - | Yes | Yes |

## Image models

| Organization | Model name | Model string for API | Price per MP |
| --- | --- | --- | --- |
| Black Forest Labs | Flux1.1 [pro] | `black-forest-labs/FLUX.1.1-pro` | $0.04 |

## Vision models

| Organization | Model name | API model string | Context length | Input pricing (per 1M tokens) | Output pricing (per 1M tokens) |
| --- | --- | --- | --- | --- | --- |
| Qwen | Qwen3.5 9B | `Qwen/Qwen3.5-9B` | 262144 | $0.17 | $0.25 |

## Video models

| Organization | Model name | Model string for API | Price per video |
| --- | --- | --- | --- |
| OpenAI | Sora 2 | `openai/sora-2` | $0.80 |

## Audio models

| Organization | Modality | Model name | Model string for API | Pricing |
| --- | --- | --- | --- | --- |
| Canopy Labs | Text-to-Speech | Orpheus 3B | `canopylabs/orpheus-3b-0.1-ft` | $15.00 per 1M chars |
| OpenAI | Speech-to-Text | Whisper Large v3 | `openai/whisper-large-v3` | $0.0015 per audio min |

## Embedding models

| Organization | Model name | API model string | Embedding dimension | Context length |
| --- | --- | --- | --- | --- |
| BAAI | bge-base-en-v1.5 | `BAAI/bge-base-en-v1.5` | 768 | 512 |

## Rerank models

There are currently no rerank models offered via serverless. Rerank models like
`mixedbread-ai/mxbai-rerank-large-v2` are only available with dedicated model
inference.

## Moderation models

| Organization | Model name | API model string |
| --- | --- | --- |
| Together | Shield | `togethercomputer/together-safety-policy` |
