<!-- official source URL: https://docs.fireworks.ai/guides/querying-embeddings-models.md -->
<!-- source_id: fireworks-docs-embeddings -->
# Embeddings & Reranking

Fireworks hosts embedding and reranking models.

## Qwen3

The Qwen3 Embedding family is hosted by Fireworks.

| Model | Model ID | Context | Availability |
| --- | --- | --- | --- |
| Qwen3 Embedding 8B | `fireworks/qwen3-embedding-8b` | 40k | Serverless and dedicated |
| Qwen3 Embedding 4B | `fireworks/qwen3-embedding-4b` | 40k | Dedicated |

## Voyage AI

Voyage models are not available on serverless, so they always run on a dedicated deployment.

| Model | Model ID | Context | Type |
| --- | --- | --- | --- |
| Voyage 4 | `fireworks/voyage-4` | 40k | Embeddings |
| Voyage ReRank 2.5 | `fireworks/voyage-rerank-2-5` | 32k | Reranking |

## Qwen3 rerankers

| Model | Model ID | Context | Availability |
| --- | --- | --- | --- |
| Qwen3 Reranker 8B | `fireworks/qwen3-reranker-8b` | 40k | Serverless and dedicated |

## Use any LLM as an embeddings model

```python
resp = client.embeddings.create(
    model="fireworks/glm-5p2",
    input=["First chunk to embed"],
)
```

## BERT-based models (legacy)

These models are serverless only. The "legacy" label is a catalogue category.

| Model id | Dimensions | Parameters |
| --- | --- | --- |
| `BAAI/bge-small-en-v1.5` | 384 | 33M |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 137M |

Retired. `sentence-transformers/all-MiniLM-L6-v2` is no longer enabled on the shared serverless pool and return a `not available` error. Use `BAAI/bge-small-en-v1.5` instead for a similarly small model.

```python
MODEL = "BAAI/bge-small-en-v1.5"
resp = client.embeddings.create(
    model=MODEL,
    input=["first sentence"],
)
```

Dedicated deployments use `accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID>` and are not public serving IDs.
