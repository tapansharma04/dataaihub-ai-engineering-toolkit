<!-- Faithful excerpt captured from https://aistudio.google.com/docs/models/gemini-decision-1.md on 2026-11-24. -->
# Gemini Decision

## gemini-decision-1

{% table variant="model" %}
  * Property
  * Description

  ---

  * Model code
  * `gemini-decision-1`

  ---

  * Supported data types
  * **Inputs** Text, Image, Video, Audio, and PDF

    **Output** Calibrated probabilities (yes/no, choice, and score, with 0 output tokens)

  ---

  * Token limits
  * **Input token limit** 262,144

    **Output token limit** 0 (up to 1,000 questions per request)

  ---

  * Capabilities
  * {% section class="three-column-grid" %}
  {% section %}
  [**Calibrated decisions**](/docs/decisions-api)
  {% badge type="supported" /%}
  {% /section %}
  {% section %}
  **Audio generation**
  {% badge type="not-supported" /%}
  {% /section %}
  {% section %}
  **Function calling**
  {% badge type="not-supported" /%}
  {% /section %}
  {% /section %}
{% /table %}
