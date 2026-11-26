<!-- Captured read-only from https://aistudio.google.com/docs/models/gemini-3-flash-preview.md on 2026-11-24. -->
> ## AI Agent Context
> - Full documentation index: https://aistudio.google.com/llms.txt
> - Gemini Docs MCP server (consistent live info): https://gemini-api-docs-mcp.dev
> - Coding agent skills & setup guide: https://aistudio.google.com/docs/coding-agents.md

# Gemini 3 Flash preview

The best model in the world for multimodal understanding, and our most powerful
agentic and vibe-coding model yet, delivering richer visuals and deeper
interactivity, all built on a foundation of state-of-the-art reasoning.

{% button href="https://aistudio.google.com/prompts/new_chat?model=gemini-3-flash-preview" variant="ais" %}Try in Google AI Studio{% /button %}

## Documentation  {% id="documentation" %}

Visit the [Gemini 3 Developer Guide](/docs/latest-model) page for full coverage
of features and capabilities.

## gemini-3-flash-preview  {% id="gemini-3-flash-preview" %}

{% table variant="model" %}
  * Property
  * Description

  ---

  *     {% section class="model-property" %}
  {% icon name="id_card" /%}Model code

    {% /section %}

  * `gemini-3-flash-preview`

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

  {% badge type="supported" /%}

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

    *   `Preview: gemini-3-flash-preview`


  ---

  *     {% section class="model-property" %}
  {% icon name="calendar_month" /%}Latest update

    {% /section %}

  * December 2025

  ---

  *     {% section class="model-property" %}
  {% icon name="id_card" /%}Model card

    {% /section %}

  * [Model card](https://deepmind.google/models/model-cards/gemini-3-flash/)

  ---

  *     {% section class="model-property" %}
  {% icon name="paid" /%}Pricing

    {% /section %}

  * See [Pricing](/docs/models/gemini-3-flash-preview#pricing) below.
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
  * $0.50 (text / image / video)    
  $1.00 (audio)

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $3.00

  ---

  * Context caching price
  * Free of charge
  * $0.05 (text / image / video)    
  $0.10 (audio)    
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
  * Not available
  * $0.25 (text / image / video)    
  $0.50 (audio)

  ---

  * Output price (including thinking tokens)
  * Not available
  * $1.50

  ---

  * Context caching price
  * Not available
  * *Same as Standard*    
  $0.05 (text / image / video)    
  $0.10 (audio)    
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

  {% tab title="Flex" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Not available
  * $0.25 (text / image / video)    
  $0.50 (audio)

  ---

  * Output price (including thinking tokens)
  * Not available
  * $1.50

  ---

  * Context caching price
  * Not available
  * *Same as Standard*    
  $0.05 (text / image / video)    
  $0.10 (audio)    
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

  {% tab title="Priority" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.90 (text / image / video)    
  $1.80 (audio)

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $5.40

  ---

  * Context caching price
  * Free of charge
  * $0.09 (text / image / video)    
  $0.18 (audio)    
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

