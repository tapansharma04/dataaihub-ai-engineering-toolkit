> ## Documentation Index
> Fetch the complete documentation index at: https://docs.fireworks.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Embeddings & Reranking

> Generate text embeddings and rerank results on Fireworks using Voyage, Qwen, Nomic, and legacy models for semantic search, RAG, and retrieval pipelines.

Fireworks hosts embedding and reranking models, which are useful for tasks like RAG and semantic search.

## Generating embeddings

Embeddings models take text as input and output a vector of floating point numbers to use for tasks like similarity comparisons and search. Our embedding service is OpenAI compatible. Refer to OpenAI's embeddings [guide](https://platform.openai.com/docs/guides/embeddings)  and OpenAI's [embeddings documentation](https://platform.openai.com/docs/api-reference/embeddings) for more information on using these models.

### Choosing a model

Every option below uses the same endpoint, `POST https://api.fireworks.ai/inference/v1/embeddings`. What changes is the value you pass as `model`.

| Path | `model` value | In the Model Library? | Best for |
| - | - | - | - |
| **Qwen3 Embedding** | `fireworks/qwen3-embedding-8b` and similar | Yes | Default choice for serverless semantic search |
| **Voyage AI** | Your deployment path | Yes | Retrieval quality with asymmetric query and document embeddings |
| **Legacy BERT / sentence-transformers** | Hugging Face id, such as `BAAI/bge-small-en-v1.5` | **No** | Small, cheap vectors and existing sentence-transformers pipelines |
| **Generative LLM** | `fireworks/gpt-oss-120b` and similar | Yes, as chat models | Hidden-state vectors when you are already using the model |
| **Your own upload** | `accounts/<your-account>/models/<model-id>` | In your account | Custom or trained embedders |

> [tab] Qwen3
>
> The Qwen3 Embedding family is hosted by Fireworks, with the 8B model available on serverless.
>
> | Model | Model ID | Context | Resizable | Availability |
> | - | - | - | - | - |
> | [Qwen3 Embedding 8B](https://app.fireworks.ai/models/fireworks/qwen3-embedding-8b) | [`fireworks/qwen3-embedding-8b`](https://app.fireworks.ai/models/fireworks/qwen3-embedding-8b) | 40k | Yes | Serverless and dedicated |
> | [Qwen3 Embedding 4B](https://app.fireworks.ai/models/fireworks/qwen3-embedding-4b) | [`fireworks/qwen3-embedding-4b`](https://app.fireworks.ai/models/fireworks/qwen3-embedding-4b) | 40k | Yes | Dedicated |
> | [Qwen3 Embedding 0.6B](https://app.fireworks.ai/models/fireworks/qwen3-embedding-0p6b) | [`fireworks/qwen3-embedding-0p6b`](https://app.fireworks.ai/models/fireworks/qwen3-embedding-0p6b) | 32k | Yes | Dedicated |
>
> Resizable models accept the `dimensions` parameter to return shorter vectors.
>
> Serverless usage of `qwen3-embedding-8b` is billed at \$0.10 per million tokens. The 4B and 0.6B models require a [dedicated deployment](https://docs.fireworks.ai/guides/ondemand-deployments.md), and you can also deploy the 8B model that way for dedicated capacity.

> [tab] Voyage AI
>
> Fireworks also hosts the Voyage AI by MongoDB family of embedding models and rerankers.
>
> | Model | Model ID | Context | Type |
> | - | - | - | - |
> | [Voyage 4 Large](https://app.fireworks.ai/models/fireworks/voyage-4-large) | [`fireworks/voyage-4-large`](https://app.fireworks.ai/models/fireworks/voyage-4-large) | 40k | Embeddings |
> | [Voyage 4](https://app.fireworks.ai/models/fireworks/voyage-4) | [`fireworks/voyage-4`](https://app.fireworks.ai/models/fireworks/voyage-4) | 40k | Embeddings |
> | [Voyage 4 Lite](https://app.fireworks.ai/models/fireworks/voyage-4-lite) | [`fireworks/voyage-4-lite`](https://app.fireworks.ai/models/fireworks/voyage-4-lite) | 40k | Embeddings |
> | [Voyage 4 Nano](https://app.fireworks.ai/models/fireworks/voyage-4-nano) | [`fireworks/voyage-4-nano`](https://app.fireworks.ai/models/fireworks/voyage-4-nano) | 40k | Embeddings |
> | [Voyage Multimodal 3.5](https://app.fireworks.ai/models/fireworks/voyage-multimodal-3-5) | [`fireworks/voyage-multimodal-3-5`](https://app.fireworks.ai/models/fireworks/voyage-multimodal-3-5) | 32k | Embeddings |
> | [Voyage ReRank 2.5](https://app.fireworks.ai/models/fireworks/voyage-rerank-2-5) | [`fireworks/voyage-rerank-2-5`](https://app.fireworks.ai/models/fireworks/voyage-rerank-2-5) | 32k | Reranking |
>
> Voyage models are not available on serverless, so they always run on a [dedicated deployment](https://docs.fireworks.ai/guides/ondemand-deployments.md). They are also trained for asymmetric retrieval, so pass `input_type="document"` when you embed a corpus and `input_type="query"` when you embed a search query. The model prepends task-specific instructions internally, which improves retrieval quality over embedding both sides the same way.

### Making a request

How you address the model depends on whether it runs on serverless or on a dedicated deployment.

> [tab] Serverless
>
> Pass the model name directly.
>
> ```python Python theme={null}
> import requests
>
> url = "https://api.fireworks.ai/inference/v1/embeddings"
>
> payload = {
>     "input": "The quick brown fox jumped over the lazy dog",
>     "model": "fireworks/qwen3-embedding-8b",
> }
>
> headers = {
>     "Authorization": "Bearer <FIREWORKS_API_KEY>",
>     "Content-Type": "application/json"
> }
>
> response = requests.post(url, json=payload, headers=headers)
>
> print(response.json())
> ```

> [tab] Dedicated
>
> Pass your deployment path instead of a model name. Voyage models additionally take `input_type`.
>
> ```python Python theme={null}
> import requests
>
> url = "https://api.fireworks.ai/inference/v1/embeddings"
>
> # Copy both IDs from your deployment's page in the Fireworks dashboard.
> ACCOUNT_ID = "<YOUR_ACCOUNT_ID>"
> DEPLOYMENT_ID = "<YOUR_DEPLOYMENT_ID>"
>
> payload = {
>     "input": "The quick brown fox jumped over the lazy dog",
>     "model": f"accounts/{ACCOUNT_ID}/deployments/{DEPLOYMENT_ID}",
>     "input_type": "document",
> }
>
> headers = {
>     "Authorization": "Bearer <FIREWORKS_API_KEY>",
>     "Content-Type": "application/json"
> }
>
> response = requests.post(url, json=payload, headers=headers)
>
> print(response.json())
> ```
>
> > [!NOTE]
> >
> > Dedicated deployments scale to zero when idle, so the first request after a quiet period can return a `503` while the deployment cold-starts. Retry with backoff.

To generate variable-length embeddings, you can add the `dimensions` parameter to the request, for example, `dimensions: 128`. Only resizable models accept it: the Qwen3 embedding models do, and among the legacy models only `nomic-ai/nomic-embed-text-v1.5` does. Any other model rejects the request with `model: <id> is not resizable`.

The API usage for embedding models is identical for BERT-based and LLM-based embeddings. Simply use the `/v1/embeddings` endpoint with your chosen model.

> [!NOTE]
>
> Not every model returns unit-length vectors. Qwen3 embeddings are not normalized, so divide by the L2 norm before using a dot product as a cosine similarity, or use a similarity function that normalizes for you. Most of the legacy BERT models do return normalized vectors.

### Other embedding options

> [accordion] Use any LLM as an embeddings model
>
> You can retrieve embeddings from many LLMs in our model library. Here are some examples that work with the embeddings API:
>
> * `fireworks/gpt-oss-120b`
>
> A few caveats before you rely on this:
>
> * These vectors come from the model's last-token hidden states, not a dedicated embedding head, so they are not trained for retrieval. Prefer a purpose-built embedding model when retrieval quality matters.
> * Raw hidden states are not unit vectors. Pass `"normalize": true` if you plan to compare them with cosine similarity.
> * Not every model in the library supports `/v1/embeddings`. Unsupported models return an `embedding is not supported` error.
>
> ```python Python theme={null}
> import os
>
> from openai import OpenAI
>
> client = OpenAI(
>     api_key=os.environ["FIREWORKS_API_KEY"],
>     base_url="https://api.fireworks.ai/inference/v1",
> )
>
> resp = client.embeddings.create(
>     model="fireworks/gpt-oss-120b",
>     input=["First chunk to embed", "Second chunk to embed"],
>     extra_body={"normalize": True},
> )
>
> for row in resp.data:
>     print(row.index, len(row.embedding))
> ```
>
> ```bash cURL theme={null}
> curl -X POST https://api.fireworks.ai/inference/v1/embeddings \
>   -H "Authorization: Bearer $FIREWORKS_API_KEY" \
>   -H "Content-Type: application/json" \
>   -d '{
>     "model": "fireworks/gpt-oss-120b",
>     "input": ["First chunk to embed", "Second chunk to embed"],
>     "normalize": true
>   }'
> ```
>
> > [!NOTE]
> >
> > `normalize` is not part of the OpenAI embeddings schema, so the Python SDK needs it in `extra_body` rather than as a top-level argument.

> [accordion] Bring your own model
>
> You can also retrieve embeddings from any models you bring yourself through [custom model upload](https://docs.fireworks.ai/models/uploading-custom-models.md).

> [accordion] BERT-based models (legacy)
>
> > [!NOTE]
> >
> > **These models are not in the Model Library.** You will not find them at [fireworks.ai/models](https://fireworks.ai/models), but they still work on serverless if you pass the exact Hugging Face style id in the `model` field, for example `nomic-ai/nomic-embed-text-v1.5` rather than `accounts/fireworks/models/...`.
>
> These models are serverless only; you cannot create a dedicated deployment for them.
>
> | Model id | Dimensions | Parameters | Resizable | Normalized |
> | - | - | - | - | - |
> | `BAAI/bge-small-en-v1.5` | 384 | 33M | No | Yes |
> | `thenlper/gte-base` | 768 | 109M | No | Yes |
> | `BAAI/bge-base-en-v1.5` | 768 | 109M | No | Yes |
> | `nomic-ai/nomic-embed-text-v1.5` | 768 | 137M | Yes | Yes |
> | `nomic-ai/nomic-embed-text-v1` | 768 | 137M | No | Yes |
> | `thenlper/gte-large` | 1024 | 335M | No | Yes |
> | `WhereIsAI/UAE-Large-V1` | 1024 | 335M | No | No |
> | `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 335M | No | No |
>
> **Resizable** means the model accepts the `dimensions` parameter. Non-resizable models reject it with `model: <id> is not resizable`. **Normalized** means the returned vectors are unit length, so a dot product is already the cosine similarity; for the two that are not, normalize the output yourself before comparing.
>
> **Retired.** `sentence-transformers/all-MiniLM-L6-v2` and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` are no longer enabled on the shared serverless pool and return a `not available` error. Use `BAAI/bge-small-en-v1.5` instead for a similarly small model.
>
> **Quickstart**
>
> ```python Python theme={null}
> import os
>
> from openai import OpenAI
>
> client = OpenAI(
>     api_key=os.environ["FIREWORKS_API_KEY"],
>     base_url="https://api.fireworks.ai/inference/v1",
> )
>
> MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim; not listed in the Model Library
>
> resp = client.embeddings.create(
>     model=MODEL,
>     input=["first sentence", "second sentence"],
> )
>
> for row in resp.data:
>     print(row.index, len(row.embedding))
> ```
>
> ```bash cURL theme={null}
> curl -X POST https://api.fireworks.ai/inference/v1/embeddings \
>   -H "Authorization: Bearer $FIREWORKS_API_KEY" \
>   -H "Content-Type: application/json" \
>   -d '{
>     "model": "BAAI/bge-small-en-v1.5",
>     "input": ["first sentence", "second sentence"]
>   }'
> ```
>
> **What to expect**
>
> * Vector length and normalization vary by model, as shown in the table above.
> * Billing follows [serverless embeddings pricing](https://docs.fireworks.ai/serverless/pricing.md#embeddings) by parameter bucket. The five models up to 150M parameters bill at $0.008 per 1M input tokens, and the three 335M models bill at $0.016.
> * Serverless only, with no dedicated deployment option.

## Reranking documents

Reranking models are used to rerank a list of documents based on a query. The `/rerank` endpoint provides a simple interface for this.

> [!NOTE]
>
> The `/rerank` endpoint does not yet support all models and parallelism options. For more flexibility, use the `/embeddings` endpoint with `return_logits` as shown in the next section.

> [tab] Qwen3
>
> | Model | Model ID | Context | Availability |
> | - | - | - | - |
> | [Qwen3 Reranker 8B](https://app.fireworks.ai/models/fireworks/qwen3-reranker-8b) | [`fireworks/qwen3-reranker-8b`](https://app.fireworks.ai/models/fireworks/qwen3-reranker-8b) | 40k | Serverless and dedicated |
> | [Qwen3 Reranker 4B](https://app.fireworks.ai/models/fireworks/qwen3-reranker-4b) | [`fireworks/qwen3-reranker-4b`](https://app.fireworks.ai/models/fireworks/qwen3-reranker-4b) | 40k | Dedicated |
> | [Qwen3 Reranker 0.6B](https://app.fireworks.ai/models/fireworks/qwen3-reranker-0p6b) | [`fireworks/qwen3-reranker-0p6b`](https://app.fireworks.ai/models/fireworks/qwen3-reranker-0p6b) | 40k | Dedicated |
>
> Serverless usage of `qwen3-reranker-8b` is billed at \$0.20 per million tokens. The 4B and 0.6B models require a dedicated deployment.

> [tab] Voyage AI
>
> | Model | Model ID | Context | Availability |
> | - | - | - | - |
> | [Voyage ReRank 2.5](https://app.fireworks.ai/models/fireworks/voyage-rerank-2-5) | [`fireworks/voyage-rerank-2-5`](https://app.fireworks.ai/models/fireworks/voyage-rerank-2-5) | 32k | Dedicated |
>
> Voyage ReRank 2.5 goes through the same `/rerank` endpoint once you deploy it.

### Making a request

> [tab] Serverless
>
> ```python Python theme={null}
> import requests
>
> url = "https://api.fireworks.ai/inference/v1/rerank"
>
> payload = {
>     "model": "fireworks/qwen3-reranker-8b",
>     "query": "What is the capital of France?",
>     "documents": [
>         "Paris is the capital and largest city of France, home to the Eiffel Tower and the Louvre Museum.",
>         "France is a country in Western Europe known for its wine, cuisine, and rich history.",
>         "The weather in Europe varies significantly between northern and southern regions.",
>         "Python is a popular programming language used for web development and data science."
>     ],
>     "top_n": 3,
>     "return_documents": True
> }
>
> headers = {
>     "Authorization": "Bearer <FIREWORKS_API_KEY>",
>     "Content-Type": "application/json"
> }
>
> response = requests.post(url, json=payload, headers=headers)
>
> print(response.json())
> ```

> [tab] Dedicated
>
> The request body is identical; only the `model` changes to your reranker's deployment path.
>
> ```python Python theme={null}
> import requests
>
> url = "https://api.fireworks.ai/inference/v1/rerank"
>
> # Copy both IDs from your deployment's page in the Fireworks dashboard.
> ACCOUNT_ID = "<YOUR_ACCOUNT_ID>"
> DEPLOYMENT_ID = "<YOUR_RERANK_DEPLOYMENT_ID>"
>
> payload = {
>     "model": f"accounts/{ACCOUNT_ID}/deployments/{DEPLOYMENT_ID}",
>     "query": "What is the capital of France?",
>     "documents": [
>         "Paris is the capital and largest city of France, home to the Eiffel Tower and the Louvre Museum.",
>         "France is a country in Western Europe known for its wine, cuisine, and rich history.",
>         "The weather in Europe varies significantly between northern and southern regions.",
>         "Python is a popular programming language used for web development and data science."
>     ],
>     "top_n": 3,
>     "return_documents": True
> }
>
> headers = {
>     "Authorization": "Bearer <FIREWORKS_API_KEY>",
>     "Content-Type": "application/json"
> }
>
> response = requests.post(url, json=payload, headers=headers)
>
> print(response.json())
> ```

### Using the `/embeddings` endpoint

You can also use the `/embeddings` endpoint with the `return_logits` parameter to rerank documents. This approach supports more models and parallelism options.

The embedding model takes in token IDs for "yes" and "no" and outputs associated logits indicating how likely the document is relevant or not relevant to the query. You can obtain these token IDs using `tokenizer.convert_tokens_to_ids()` with the transformers library and the Qwen3 tokenizer.

> [tab] Simple
>
> ```python Python theme={null}
> import requests
>
> url = "https://api.fireworks.ai/inference/v1/embeddings"
>
> query = "What is the capital of France?"
> documents = [
>     "Paris is the capital and largest city of France, home to the Eiffel Tower and the Louvre Museum.",
>     "France is a country in Western Europe known for its wine, cuisine, and rich history.",
>     "The weather in Europe varies significantly between northern and southern regions.",
>     "Python is a popular programming language used for web development and data science."
> ]
>
> # Format prompts as query-document pairs using the Qwen3 Reranker format
> instruction = "Given a web search query, retrieve relevant passages that answer the query"
> prompts = [
>     f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
>     for doc in documents
> ]
>
> # Token IDs for "no" and "yes" in Qwen3 reranker models
> token_false_id = 2753   # "no"
> token_true_id = 9454    # "yes"
>
> payload = {
>     "model": "fireworks/qwen3-reranker-8b",
>     "input": prompts,
>     "return_logits": [token_false_id, token_true_id],
>     "normalize": True  # Applies softmax to the selected logits
> }
>
> headers = {
>     "Authorization": "Bearer <FIREWORKS_API_KEY>",
>     "Content-Type": "application/json"
> }
>
> response = requests.post(url, json=payload, headers=headers).json()
>
> # Extract relevance scores (probability of "yes" token)
> results = []
> for i, item in enumerate(response["data"]):
>     probs = item["embedding"]  # [no_prob, yes_prob]
>     relevance_score = probs[1]  # "yes" probability is the relevance score
>     results.append({
>         "index": i,
>         "relevance_score": relevance_score,
>         "document": documents[i]
>     })
>
> # Sort by relevance score (highest first)
> results.sort(key=lambda x: x["relevance_score"], reverse=True)
>
> for result in results:
>     print(f"Score: {result['relevance_score']:.4f} - {result['document'][:80]}...")
>
> ```

> [tab] Parallel (asyncio)
>
> For large document sets, you can improve throughput by sending multiple requests in parallel using minibatches:
>
> ```python Python theme={null}
> import asyncio
> import aiohttp
>
> url = "https://api.fireworks.ai/inference/v1/embeddings"
>
> query = "What is the capital of France?"
> documents = [...]  # Your list of documents
>
> # Format prompts as query-document pairs using the Qwen3 Reranker format
> instruction = "Given a web search query, retrieve relevant passages that answer the query"
> prompts = [
>     f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
>     for doc in documents
> ]
>
> # Token IDs for "no" and "yes" in Qwen3 reranker models
> token_false_id = 2753   # "no"
> token_true_id = 9454    # "yes"
>
> headers = {
>     "Authorization": "Bearer <FIREWORKS_API_KEY>",
>     "Content-Type": "application/json"
> }
>
> async def rerank_batch(session, batch_prompts):
>     payload = {
>         "model": "fireworks/qwen3-reranker-8b",
>         "input": batch_prompts,
>         "return_logits": [token_false_id, token_true_id],
>         "normalize": True
>     }
>     async with session.post(url, json=payload, headers=headers) as response:
>         return await response.json()
>
> async def rerank_parallel(prompts, batch_size=100):
>     batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
>
>     async with aiohttp.ClientSession() as session:
>         tasks = [rerank_batch(session, batch) for batch in batches]
>         results = await asyncio.gather(*tasks)
>
>     # Combine results from all batches
>     all_scores = []
>     for result in results:
>         for item in result["data"]:
>             all_scores.append(item["embedding"][1])  # "yes" probability
>
>     return all_scores
>
> scores = asyncio.run(rerank_parallel(prompts))
> ```

With `normalize=True`, the endpoint applies softmax to the selected logits, returning probabilities that sum to 1. The "yes" probability directly represents the relevance score.

## Building a two-stage retrieval pipeline

Embeddings and reranking are usually combined: vector search recalls a broad candidate set quickly, then a reranker reorders those candidates for precision. This cookbook builds that pipeline end to end with Voyage models on Fireworks and MongoDB vector search.

> [card] [Two-stage retrieval with Voyage embeddings and reranking](https://colab.research.google.com/github/fw-ai/cookbook/blob/main/partners/voyage-ai/voyage_two_stage_retrieval.ipynb)
>
> Embed a corpus with a Voyage embedding model, retrieve with MongoDB `$vectorSearch`, and rerank the candidates with Voyage ReRank 2.5.

## Deploying embeddings and reranking models

While Qwen3 Embedding 8b and Qwen3 Reranker 8b are available on serverless, you also have the option to deploy them via [on-demand deployments](https://docs.fireworks.ai/guides/ondemand-deployments.md). Voyage models require a dedicated deployment.

## Troubleshooting

> [accordion] I can't find my embedding model in the Model Library
>
> Legacy BERT and sentence-transformers embedders are not listed in the Model Library, but they still serve on serverless. Pass the Hugging Face style id directly, for example `BAAI/bge-small-en-v1.5`. See [BERT-based models (legacy)](#other-embedding-options) for the current list.

> [accordion] What is the difference between `model not found` and `not available`?
>
> `not found` means the id does not resolve, which usually points to a typo or a deployment path for a deployment that does not exist in your account. `not available` means the id is real but is not enabled on the shared serverless pool, which is what retired legacy embedders return. In that case, switch to another model from the legacy list or to `fireworks/qwen3-embedding-8b`.

> [accordion] My dedicated deployment returns a 503
>
> Dedicated deployments scale to zero when idle, so the first request after a quiet period can fail while the deployment cold-starts. Retry with backoff rather than treating it as a hard failure.
