> ## Documentation Index
> Fetch the complete documentation index at: https://docs.together.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# GLM-5.3 quickstart

> Call GLM-5.3 and GLM-5.3 Flash on Together for long-horizon coding and agentic work.

GLM-5.3 is Z.ai's flagship mixture-of-experts (MoE) model, built for long-horizon coding and agentic work. It holds project-scale context across many steps and completes multi-stage tasks end to end. Thinking is on by default, so responses carry both a reasoning trace and a final answer.

Two variants are available on serverless inference, both with a 1M-token context window, streaming, function calling, and structured outputs:

| Model | Model ID | Input / 1M tokens | Cached input / 1M tokens | Output / 1M tokens |
| - | - | - | - | - |
| GLM-5.3 | `zai-org/GLM-5.3` | \$1.40 | \$0.26 | \$4.40 |
| GLM-5.3 Flash | `zai-org/GLM-5.3-Flash` | \$0.15 | \$0.03 | \$0.50 |

GLM-5.3 is Together's [recommended pick](https://docs.together.ai/docs/inference/recommended-models.md) for coding agents. GLM-5.3 Flash is the recommended pick for function calling and other high-volume calls. The examples below use `zai-org/GLM-5.3`. Swap in `zai-org/GLM-5.3-Flash` when cost and latency matter more than maximum depth.

## Call GLM-5.3

The reasoning trace arrives on `reasoning_content` and the final answer arrives on `content`. Stream the response and handle both channels, and skip chunks where `choices` is empty, since Together emits a final usage-only chunk.

GLM-5.3 emits the trace under `reasoning_content`, while some other Together models use the newer `reasoning` alias. Reading both keys makes one handler work across models.

```python Python theme={null}
from together import Together

client = Together()

stream = client.chat.completions.create(
    model="zai-org/GLM-5.3",
    messages=[
        {
            "role": "user",
            "content": "What are some fun things to do in New York?",
        }
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=8192,
    stream=True,
)

for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    thinking = getattr(delta, "reasoning_content", None) or getattr(
        delta, "reasoning", None
    )
    # Stream reasoning and content tokens as they arrive
    print(thinking or delta.content or "", end="", flush=True)
```

```typescript TypeScript theme={null}
import Together from "together-ai";
import type { ChatCompletionChunk } from "together-ai/resources/chat/completions";

const together = new Together();

// GLM-5.3 emits the trace on reasoning_content; other models use reasoning
type ReasoningDelta = ChatCompletionChunk.Choice.Delta & {
  reasoning_content?: string;
  reasoning?: string;
};

const stream = await together.chat.completions.create({
  model: "zai-org/GLM-5.3",
  messages: [
    {
      role: "user",
      content: "What are some fun things to do in New York?",
    },
  ],
  temperature: 1.0,
  top_p: 0.95,
  max_tokens: 8192,
  stream: true,
});

for await (const chunk of stream) {
  if (!chunk.choices?.length) continue;
  const delta = chunk.choices[0]?.delta as ReasoningDelta;
  const thinking = delta?.reasoning_content || delta?.reasoning;
  // Stream reasoning and content tokens as they arrive
  process.stdout.write(thinking || delta?.content || "");
}
```

```bash cURL theme={null}
curl -X POST "https://api.together.ai/v1/chat/completions" \
     -H "Authorization: Bearer $TOGETHER_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
        "model": "zai-org/GLM-5.3",
        "messages": [
          {"role": "user", "content": "What are some fun things to do in New York?"}
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "stream": true
     }'
```

## Set the reasoning effort

`reasoning_effort` accepts `"low"`, `"medium"`, `"high"`, and `"max"`. Use `"low"` for short, high-volume calls and `"max"` for the hardest planning, architecture, and multi-step agentic problems. At `"max"`, set `max_tokens` generously, since the trace and the answer share the same completion budget.

Thinking cannot be disabled entirely on GLM-5.3. `reasoning_effort="low"` cuts the trace to a few tokens, which is the closest equivalent for trivial turns. The levels are coarse dials rather than a strictly monotonic scale, so measure token counts on your own prompts before you tune. Invalid effort strings are accepted silently instead of returning an error, so validate the value in your own code.

```python Python theme={null}
from together import Together

client = Together()

completion = client.chat.completions.create(
    model="zai-org/GLM-5.3",
    messages=[
        {
            "role": "user",
            "content": "Design a three-tier architecture for a ticketing system.",
        }
    ],
    reasoning_effort="max",
    max_tokens=65536,
)

print(completion.choices[0].message.content)
```

```typescript TypeScript theme={null}
import Together from "together-ai";

const together = new Together();

const completion = await together.chat.completions.create({
  model: "zai-org/GLM-5.3",
  messages: [
    {
      role: "user",
      content: "Design a three-tier architecture for a ticketing system.",
    },
  ],
  reasoning_effort: "max",
  max_tokens: 65536,
});

console.log(completion.choices[0].message.content);
```

For broader guidance on reasoning controls and prompting, see [Reasoning](https://docs.together.ai/docs/inference/chat/reasoning.md).

## Preserve thinking across turns

For coding agents and other agent workflows, enable preserved thinking so the model retains its reasoning from previous turns. Set `"clear_thinking": false` in `chat_template_kwargs` to keep reasoning content in context across turns, which improves reasoning continuity and cache hit rates.

```python Python theme={null}
from together import Together

client = Together()

response = client.chat.completions.create(
    model="zai-org/GLM-5.3",
    messages=[
        {
            "role": "user",
            "content": "Refactor this module without changing its public API.",
        }
    ],
    chat_template_kwargs={
        "clear_thinking": False,  # Preserved thinking
    },
)

print(response.choices[0].message.content)
```

> [!NOTE]
>
> When using preserved thinking, return the model's `reasoning_content` blocks exactly as generated. Reordering or editing them degrades performance and hurts cache hit rates.

## Call tools

GLM-5.3 supports tool calling with reasoning interleaved between each step. Define tools in the standard OpenAI-compatible schema and pass them via `tools`.

To stream tool calls, set `stream=True`. The model emits tool call parameters incrementally, so concatenate the `arguments` fragments from each delta to rebuild the full call. Together does not use a separate `tool_stream` parameter.

```python Python theme={null}
from together import Together

client = Together()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. Beijing, Shanghai.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]

stream = client.chat.completions.create(
    model="zai-org/GLM-5.3",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
    stream=True,
)

final_tool_calls = {}

for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta

    # Reassemble streamed tool calls by index. The first delta for a call
    # carries the name, and later deltas carry argument fragments.
    if delta.tool_calls:
        for tool_call in delta.tool_calls:
            idx = tool_call.index
            if idx not in final_tool_calls:
                final_tool_calls[idx] = {
                    "name": tool_call.function.name,
                    "arguments": "",
                }
            final_tool_calls[idx]["arguments"] += (
                tool_call.function.arguments or ""
            )

for call in final_tool_calls.values():
    print(f"{call['name']}: {call['arguments']}")
```

```typescript TypeScript theme={null}
import Together from "together-ai";

const together = new Together();

const tools = [
  {
    type: "function" as const,
    function: {
      name: "get_weather",
      description: "Get current weather conditions for a city.",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "City name, e.g. Beijing, Shanghai.",
          },
          unit: { type: "string", enum: ["celsius", "fahrenheit"] },
        },
        required: ["location"],
      },
    },
  },
];

const stream = await together.chat.completions.create({
  model: "zai-org/GLM-5.3",
  messages: [{ role: "user", content: "What's the weather in Beijing?" }],
  tools,
  stream: true,
});

const finalToolCalls: Record<number, { name: string; arguments: string }> = {};

for await (const chunk of stream) {
  const delta = chunk.choices[0]?.delta;
  if (!delta?.tool_calls) continue;

  for (const toolCall of delta.tool_calls) {
    const idx = toolCall.index;
    if (!(idx in finalToolCalls)) {
      finalToolCalls[idx] = {
        name: toolCall.function?.name ?? "",
        arguments: toolCall.function?.arguments ?? "",
      };
    } else {
      finalToolCalls[idx].arguments += toolCall.function?.arguments ?? "";
    }
  }
}

for (const idx in finalToolCalls) {
  const call = finalToolCalls[idx];
  console.log(`${call.name}: ${call.arguments}`);
}
```

For the full tool-calling loop, including how to return tool results to the model, see [Function calling](https://docs.together.ai/docs/inference/function-calling/overview.md).

## Constrain the output to a schema

GLM-5.3 supports [structured outputs](https://docs.together.ai/docs/inference/chat/structured-outputs.md). Pass a JSON schema through `response_format` to constrain the response to a fixed shape. Parse `content` only, never `reasoning_content`.

```python Python theme={null}
from together import Together

client = Together()

response = client.chat.completions.create(
    model="zai-org/GLM-5.3",
    messages=[
        {
            "role": "user",
            "content": "Extract the person: John is 30 years old.",
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        },
    },
)

print(response.choices[0].message.content)
```

```typescript TypeScript theme={null}
import Together from "together-ai";

const together = new Together();

const response = await together.chat.completions.create({
  model: "zai-org/GLM-5.3",
  messages: [
    { role: "user", content: "Extract the person: John is 30 years old." },
  ],
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "person",
      schema: {
        type: "object",
        properties: {
          name: { type: "string" },
          age: { type: "integer" },
        },
        required: ["name", "age"],
      },
    },
  },
});

console.log(response.choices[0].message.content);
```

## Usage tips

| Tip | Rationale |
| - | - |
| **Read both `reasoning_content` and `reasoning`** | GLM-5.3 emits the trace on `reasoning_content`. Other Together models use the `reasoning` alias. One handler then covers both. |
| **Parse `content` only** | Never run JSON parsing over the thinking trace. |
| **Temperature = 1.0, top\_p = 0.95** | Recommended defaults. Adjust only one of the two, not both at once. |
| **Match `reasoning_effort` to the task** | Use `"low"` for trivial turns and `"max"` with generous `max_tokens` for hard planning and architecture work. |
| **Use GLM-5.3 Flash for high-volume calls** | Same 1M context and tool support at a fraction of the price, for function calling and latency-sensitive work. |
| **Use preserved thinking for agents** | Set `"clear_thinking": false` in `chat_template_kwargs` so coding agents keep reasoning continuity across turns. |
| **Think in goals, not steps** | GLM-5.3 is agentic. Give high-level objectives and let it orchestrate sub-tasks and tool calls. |
| **State constraints explicitly** | For engineering tasks, spell out hard constraints (no new dependencies, no API changes, run the tests) so the model holds the line. |

## Next steps

> [card] [Reasoning](https://docs.together.ai/docs/inference/chat/reasoning.md)
>
> Control reasoning depth and handle reasoning output across models.

> [card] [Function calling](https://docs.together.ai/docs/inference/function-calling/overview.md)
>
> Build tool-calling loops against any function-calling model.

> [card] [Recommended models](https://docs.together.ai/docs/inference/recommended-models.md)
>
> See Together's current picks for every use case.

> [card] [Serverless models](https://docs.together.ai/docs/serverless/models.md)
>
> Browse every model, context length, and price on serverless inference.


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.