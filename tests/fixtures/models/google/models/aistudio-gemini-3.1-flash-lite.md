<!-- Captured read-only from https://aistudio.google.com/docs/models/gemini-3.1-flash-lite.md on 2026-11-24. -->
> ## AI Agent Context
> - Full documentation index: https://aistudio.google.com/llms.txt
> - Gemini Docs MCP server (consistent live info): https://gemini-api-docs-mcp.dev
> - Coding agent skills & setup guide: https://aistudio.google.com/docs/coding-agents.md

# Gemini 3.1 Flash-Lite

Gemini 3.1 Flash-Lite is a low-latency, cost-effective multimodal model
optimized for high-frequency, lightweight tasks. The model supports text, image,
video, audio, and PDF inputs, and is designed for high-volume agentic workflows,
simple data extraction, and applications where latency and API cost are the
primary constraints.

{% button href="https://aistudio.google.com/prompts/new_chat?model=gemini-3.1-flash-lite" variant="ais" %}Try in Google AI Studio{% /button %}

## gemini-3.1-flash-lite  {% id="gemini-31-flash-lite" %}

{% table variant="model" %}
  * Property
  * Description

  ---

  *     {% section class="model-property" %}
  {% icon name="id_card" /%}Model code

    {% /section %}

  * `gemini-3.1-flash-lite`

  ---

  *     {% section class="model-property" %}
  {% icon name="save" /%}Supported data types

    {% /section %}

  *     **Inputs** Text, Image, Video, Audio, and PDF

    **Output** Text


  ---

  *     {% section class="model-property" %}
  {% icon name="token_auto" /%}Token limits{% sup %}[[*]](/docs/tokens){% /sup %}

    {% /section %}

  *     **Input token limit** 1,048,576

    **Output token limit** 65,536


  ---

  *     {% section class="model-property" %}
  {% icon name="handyman" /%}Capabilities

    {% /section %}

  *     {% section class="three-column-grid" %}
  {% section %}
  [**Audio generation**](/docs/speech-generation)

  {% badge type="not-supported" /%}

  {% /section %}

  {% section %}
  [**Code execution**](/docs/code-execution)

  {% badge type="supported" /%}

  {% /section %}

  {% section %}
  [**Computer use**](/docs/computer-use)

  {% badge type="not-supported" /%}

  {% /section %}

  {% section %}
  [**Context caching**](/docs/caching)

  {% badge type="supported" /%}

  {% /section %}

  {% section %}
  [**File search**](/docs/file-search)

  {% badge type="supported" /%}

  {% /section %}

  {% section %}
  [**Function calling**](/docs/function-calling)

  {% badge type="supported" /%}

  {% /section %}

  {% section %}
  [**Grounding with Google Maps**](/docs/maps-grounding)

  {% badge type="supported" /%}

  {% /section %}

  {% section %}
  [**Grounding with Google Search**](/docs/google-search)

  {% badge type="supported" /%}

  {% /section %}

  {% section %}
  [**Image generation**](/docs/image-generation)

  {% badge type="not-supported" /%}

  {% /section %}

  {% section %}
  [**Live API**](/docs/live-api)

  {% badge type="not-supported" /%}

  {% /section %}

  {% section %}
  [**Structured outputs**](/docs/structured-output)

  {% badge type="supported" /%}

  {% /section %}

  {% section %}
  [**Thinking**](/docs/thinking)

  {% badge type="supported" /%}

  {% /section %}

  {% section %}
  [**URL context**](/docs/url-context)

  {% badge type="supported" /%}

  {% /section %}

    {% /section %}


  ---

  *     {% section class="model-property" %}
  {% icon name="speed" /%}Consumption options

    {% /section %}

  *     {% section class="three-column-grid" %}
  {% section %}
  [**Batch API**](/docs/batch-api)

  {% badge type="supported" /%}

  {% /section %}

  {% section %}
  [**Flex inference**](/docs/flex-inference)

  {% badge type="supported" /%}

  {% /section %}

  {% section %}
  [**Priority inference**](/docs/priority-inference)

  {% badge type="supported" /%}

  {% /section %}

    {% /section %}


  ---

  *     {% section class="model-property" %}
  {% icon name="123" /%}Versions

    {% /section %}

  *     Read the [model version patterns](/docs/models#model-versions) for more details.

    *   `Stable: gemini-3.1-flash-lite`


  ---

  *     {% section class="model-property" %}
  {% icon name="calendar_month" /%}Latest update

    {% /section %}

  * May 2026

  ---

  *     {% section class="model-property" %}
  {% icon name="id_card" /%}Model card

    {% /section %}

  * [Model card](https://deepmind.google/models/model-cards/gemini-3-1-flash-lite/)

  ---

  *     {% section class="model-property" %}
  {% icon name="paid" /%}Pricing

    {% /section %}

  * See [Pricing](/docs/models/gemini-3.1-flash-lite#pricing) below.
{% /table %}

## Pricing  {% id="pricing" %}

{% tabs %}
  {% tab title="Standard" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.25 (text / image / video)    
  $0.50 (audio)

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $1.50

  ---

  * Context caching price
  * Not available
  * $0.025 (text / image / video)    
  $0.05 (audio)    
  $1.00 / 1,000,000 tokens per hour (storage price)

  ---

  * Grounding with Google Search{% sup %}*{% /sup %}
  * Not available{% sup %}**{% /sup %}
  * 5,000 free search requests per month (shared across all Gemini 3 and newer models),
  then $14 per 1,000 requests (subject to a [daily cap](/docs/pricing#pricing-for-tools) of 1,500 requests per day).

  ---

  * Grounding with Google Maps
  * Not available{% sup %}**{% /sup %}
  * 5,000 requests per month (free, shared across all Gemini 3 and newer models), then $14 / 1,000
  search queries.

  ---

  * Used to improve our products
  * [Yes](/docs/terms)
  * [No](/docs/terms)
  {% /table %}

  {% /tab %}

  {% tab title="Batch" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.125 (text / image / video)    
  $0.25 (audio)

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $0.75

  ---

  * Context caching price
  * Not available
  * $0.0125 (text / image / video)    
  $0.025 (audio)    
  $0.50 / 1,000,000 tokens per hour (storage price)

  ---

  * Grounding with Google Search{% sup %}*{% /sup %}
  * Not available{% sup %}**{% /sup %}
  * 5,000 free search requests per month (shared across all Gemini 3 and newer models),
  then $14 per 1,000 requests (subject to a [daily cap](/docs/pricing#pricing-for-tools) of 1,500 requests per day).

  ---

  * Grounding with Google Maps
  * Not available{% sup %}**{% /sup %}
  * 5,000 requests per month (free, shared across all Gemini 3 and newer models), then $14 / 1,000
  search queries.

  ---

  * Used to improve our products
  * [Yes](/docs/terms)
  * [No](/docs/terms)
  {% /table %}

  {% /tab %}

  {% tab title="Flex" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.125 (text / image / video)    
  $0.25 (audio)

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $0.75

  ---

  * Context caching price
  * Not available
  * $0.0125 (text / image / video)    
  $0.025 (audio)    
  $0.50 / 1,000,000 tokens per hour (storage price)

  ---

  * Grounding with Google Search{% sup %}*{% /sup %}
  * Not available{% sup %}**{% /sup %}
  * 5,000 free search requests per month (shared across all Gemini 3 and newer models),
  then $14 per 1,000 requests (subject to a [daily cap](/docs/pricing#pricing-for-tools) of 1,500 requests per day).

  ---

  * Grounding with Google Maps
  * Not available{% sup %}**{% /sup %}
  * 5,000 requests per month (free, shared across all Gemini 3 and newer models), then $14 / 1,000
  search queries.

  ---

  * Used to improve our products
  * [Yes](/docs/terms)
  * [No](/docs/terms)
  {% /table %}

  {% /tab %}

  {% tab title="Priority" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.45 (text / image / video)    
  $0.90 (audio)

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $2.70

  ---

  * Context caching price
  * Not available
  * $0.045 (text / image / video)    
  $0.09 (audio)    
  $1.80 / 1,000,000 tokens per hour (storage price)

  ---

  * Grounding with Google Search{% sup %}*{% /sup %}
  * Not available{% sup %}**{% /sup %}
  * 5,000 free search requests per month (shared across all Gemini 3 and newer models),
  then $14 per 1,000 requests (subject to a [daily cap](/docs/pricing#pricing-for-tools) of 1,500 requests per day).

  ---

  * Grounding with Google Maps
  * Not available{% sup %}**{% /sup %}
  * 5,000 requests per month (free, shared across all Gemini 3 and newer models), then $14 / 1,000
  search queries.

  ---

  * Used to improve our products
  * [Yes](/docs/terms)
  * [No](/docs/terms)
  {% /table %}

  {% /tab %}

{% /tabs %}

{% sup %}*{% /sup %} A customer-submitted request to Gemini may result in one or
more queries to Google Search. You will be charged for each individual search
query performed.

{% sup %}**{% /sup %} Can be tested in Google AI Studio.

## Developer guide  {% id="developer-guide" %}

Gemini 3.1 Flash-Lite is best at handling straightforward tasks at significant
scale. Here are some use cases best suited for Gemini 3.1 Flash-Lite:

* **Translation**: Fast, cheap, high-volume translation, such as processing
  chat messages, reviews, and support tickets at scale. You can use system
  instructions to constrain output to only the translated text with no extra
  commentary:

  ```python
  from google import genai

  client = genai.Client()
  text = "Hey, are you down to grab some pizza later? I'm starving!"

  response = client.models.generate_content(
      model="gemini-3.1-flash-lite",
      config={
          "system_instruction": "Only output the translated text"
      },
      contents=f"Translate the following text to German: {text}"
  )

  print(response.text)
  ```

* **Transcription**: Process recordings, voice notes, or any audio content
  where you need a text transcript without spinning up a separate
  speech-to-text pipeline. Supports multimodal inputs, so you can pass audio
  files directly for transcription:

  ```python
  from google import genai

  client = genai.Client()

  uploaded_file = client.files.upload(file='sample.mp3')

  prompt = 'Generate a transcript of the audio.'

  response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=[prompt, uploaded_file]
  )

  print(response.text)
  ```

* **Lightweight agentic tasks and data extraction**: Entity extraction,
  classification, and lightweight data processing pipelines supported with
  structured JSON output. For example, extracting structured data from an
  e-commerce customer review:

  ```python
  from google import genai
  from pydantic import BaseModel, Field

  client = genai.Client()

  prompt = "Analyze the user review and determine the aspect, sentiment score, summary quote, and return risk"
  input_text = "The boots look amazing and the leather is high quality, but they run way too small. I'm sending them back."

  class ReviewAnalysis(BaseModel):
      aspect: str = Field(description="The feature mentioned (e.g., Price, Comfort, Style, Shipping)")
      summary_quote: str = Field(description="The specific phrase from the review about this aspect")
      sentiment_score: int = Field(description="1 to 5 (1=worst, 5=best)")
      is_return_risk: bool = Field(description="True if the user mentions returning the item")

  response = client.models.generate_content(
      model="gemini-3.1-flash-lite",
      contents=[prompt, input_text],
      config={
          "response_mime_type": "application/json",
          "response_json_schema": ReviewAnalysis.model_json_schema(),
      },
  )

  print(response.text)
  ```

* **Document processing and summarization**: Parse PDFs and return concise
  summaries, like for building a document processing pipeline or quickly
  triaging incoming files:

  ```python
  from google import genai
  from google.genai import types
  import httpx

  client = genai.Client()

  doc_url = "https://storage.googleapis.com/generativeai-downloads/data/med_gemini.pdf"
  doc_data = httpx.get(doc_url).content

  prompt = "Summarize this document"
  response = client.models.generate_content(
      model="gemini-3.1-flash-lite",
      contents=[
          types.Part.from_bytes(
              data=doc_data,
              mime_type='application/pdf',
          ),
          prompt
      ]
  )

  print(response.text)
  ```

* **Model routing**: Use a low-latency and low-cost model as a classifier that
  routes queries to the appropriate model based on task complexity. This is a
  real pattern in production - the open-source [Gemini CLI](https://geminicli.com/docs/core/#model-fallback) uses Flash-Lite to
  classify task complexity and route to Flash or Pro accordingly.

  ```python
  from google import genai

  client = genai.Client()

  FLASH_MODEL = 'flash'
  PRO_MODEL = 'pro'

  CLASSIFIER_SYSTEM_PROMPT = f"""
  You are a specialized Task Routing AI. Your sole function is to analyze the user's request and classify its complexity. Choose between `{FLASH_MODEL}` (SIMPLE) or `{PRO_MODEL}` (COMPLEX).

  1. `{FLASH_MODEL}`: A fast, efficient model for simple, well-defined tasks.
  2. `{PRO_MODEL}`: A powerful, advanced model for complex, open-ended, or multi-step tasks.

  A task is COMPLEX if it meets ONE OR MORE of the following criteria:

  1. High Operational Complexity (Est. 4+ Steps/Tool Calls)
  2. Strategic Planning and Conceptual Design
  3. High Ambiguity or Large Scope
  4. Deep Debugging and Root Cause Analysis

  A task is SIMPLE if it is highly specific, bounded, and has Low Operational Complexity (Est. 1-3 tool calls).
  """

  user_input = "I'm getting an error 'Cannot read property 'map' of undefined' when I click the save button. Can you fix it?"

  response_schema = {
    "type": "object",
    "properties": {
      "reasoning": {
        "type": "string",
        "description": "A brief, step-by-step explanation for the model choice, referencing the rubric."
      },
      "model_choice": {
        "type": "string",
        "enum": [FLASH_MODEL, PRO_MODEL]
      }
    },
    "required": ["reasoning", "model_choice"]
  }

  response = client.models.generate_content(
      model="gemini-3.1-flash-lite",
      contents=user_input,
      config={
          "system_instruction": CLASSIFIER_SYSTEM_PROMPT,
          "response_mime_type": "application/json",
          "response_json_schema": response_schema
      },
  )

  print(response.text)
  ```

* **Thinking**: For better accuracy for tasks that benefit from step-by-step
  reasoning, configure thinking so the model spends additional compute on
  internal reasoning before producing the final output:

  ```python
  from google import genai
  from google.genai import types

  client = genai.Client()

  response = client.models.generate_content(
      model="gemini-3.1-flash-lite",
      contents="How does AI work?",
      config=types.GenerateContentConfig(
          thinking_config=types.ThinkingConfig(thinking_level="high")
      ),
  )

  print(response.text)
  ```


