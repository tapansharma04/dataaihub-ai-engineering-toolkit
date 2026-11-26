<!-- Captured read-only from https://aistudio.google.com/docs/models/gemini-3.1-pro-preview.md on 2026-11-24. -->
> ## AI Agent Context
> - Full documentation index: https://aistudio.google.com/llms.txt
> - Gemini Docs MCP server (consistent live info): https://gemini-api-docs-mcp.dev
> - Coding agent skills & setup guide: https://aistudio.google.com/docs/coding-agents.md

# Gemini 3.1 Pro preview

Built to refine the performance and reliability of the Gemini 3 Pro series,
Gemini 3.1 Pro Preview provides better thinking, improved token
efficiency, and a more grounded, factually consistent experience. It's optimized
for software engineering behavior and usability, as well as agentic workflows
requiring precise tool usage and reliable multi-step execution across real-world
domains.

{% button href="https://aistudio.google.com/prompts/new_chat?model=gemini-3.1-pro-preview" variant="ais" %}
  Try in Google AI Studio

{% /button %}

## Documentation  {% id="documentation" %}

Visit the [Gemini 3 Developer Guide](/docs/latest-model) page for full coverage
of features and capabilities.

## gemini-3.1-pro-preview  {% id="gemini-31-pro-preview" %}

{% table variant="model" %}
  * Property
  * Description

  ---

  *     {% section class="model-property" %}
  {% icon name="id_card" /%}Model code

    {% /section %}

  * `gemini-3.1-pro-preview`

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

    *   Preview: `gemini-3.1-pro-preview`
    *   Preview: [`gemini-3.1-pro-preview-customtools`](/docs/models/gemini-3.1-pro-preview#gemini-31-pro-preview-customtools)


  ---

  *     {% section class="model-property" %}
  {% icon name="calendar_month" /%}Latest update

    {% /section %}

  * February 2026

  ---

  *     {% section class="model-property" %}
  {% icon name="id_card" /%}Model card

    {% /section %}

  * [Model card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)

  ---

  *     {% section class="model-property" %}
  {% icon name="paid" /%}Pricing

    {% /section %}

  * See [Pricing](/docs/models/gemini-3.1-pro-preview#pricing) below.
{% /table %}

### gemini-3.1-pro-preview-customtools  {% id="gemini-31-pro-preview-customtools" %}

For those building with a mix of bash and custom tools, Gemini 3.1 Pro
Preview comes with a separate endpoint available via the API called
`gemini-3.1-pro-preview-customtools`. This endpoint is better at prioritizing
your custom tools (for example `view_file` or `search_code`).

Note that while `gemini-3.1-pro-preview-customtools` is optimized for agentic
workflows that use custom tools and bash, you may see quality fluctuations in
some use cases which don't benefit from such tools.

## Pricing  {% id="pricing" %}

{% tabs %}
  {% tab title="Standard" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Not available
  * $2.00, prompts <= 200k tokens    
  $4.00, prompts > 200k tokens

  ---

  * Output price (including thinking tokens)
  * Not available
  * $12.00, prompts <= 200k tokens    
  $18.00, prompts > 200k

  ---

  * Context caching price
  * Not available
  * $0.20, prompts <= 200k tokens    
  $0.40, prompts > 200k    
  $4.50 / 1,000,000 tokens per hour (storage price)

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
  * Not available
  * [No](/docs/terms)
  {% /table %}

  {% sup %}*{% /sup %} A customer-submitted request to Gemini may result in one or
    more queries to Google Search. You will be charged for each individual search
    query performed.

  {% sup %}**{% /sup %} Can be tested in Google AI Studio.
    

  {% /tab %}

  {% tab title="Batch" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Not available
  * $1.00, prompts <= 200k tokens    
  $2.00, prompts > 200k tokens

  ---

  * Output price (including thinking tokens)
  * Not available
  * $6.00, prompts <= 200k tokens    
  $9.00, prompts > 200k

  ---

  * Context caching price
  * Not available
  * *Same as Standard*    
  $0.20, prompts <= 200k tokens    
  $0.40, prompts > 200k    
  $4.50 / 1,000,000 tokens per hour (storage price)

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
  * Not available
  * [No](/docs/terms)
  {% /table %}

  {% sup %}*{% /sup %} A customer-submitted request to Gemini may result in one or
    more queries to Google Search. You will be charged for each individual search
    query performed.

  {% sup %}**{% /sup %} Can be tested in Google AI Studio.
    

  {% /tab %}

  {% tab title="Flex" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Not available
  * $1.00, prompts <= 200k tokens    
  $2.00, prompts > 200k tokens

  ---

  * Output price (including thinking tokens)
  * Not available
  * $6.00, prompts <= 200k tokens    
  $9.00, prompts > 200k

  ---

  * Context caching price
  * Not available
  * *Same as Standard*    
  $0.20, prompts <= 200k tokens    
  $0.40, prompts > 200k    
  $4.50 / 1,000,000 tokens per hour (storage price)

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
  * Not available
  * [No](/docs/terms)
  {% /table %}

  {% sup %}*{% /sup %} A customer-submitted request to Gemini may result in one or
    more queries to Google Search. You will be charged for each individual search
    query performed.

  {% sup %}**{% /sup %} Can be tested in Google AI Studio.
    

  {% /tab %}

  {% tab title="Priority" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Not available
  * $3.60, prompts <= 200k tokens    
  $7.20, prompts > 200k tokens

  ---

  * Output price (including thinking tokens)
  * Not available
  * $21.60, prompts <= 200k tokens    
  $32.40, prompts > 200k

  ---

  * Context caching price
  * Not available
  * $0.36, prompts <= 200k tokens    
  $0.72, prompts > 200k    
  $8.10 / 1,000,000 tokens per hour (storage price)

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
  * Not available
  * [No](/docs/terms)
  {% /table %}

  {% sup %}*{% /sup %} A customer-submitted request to Gemini may result in one or
    more queries to Google Search. You will be charged for each individual search
    query performed.

  {% sup %}**{% /sup %} Can be tested in Google AI Studio.
    

  {% /tab %}

{% /tabs %}

