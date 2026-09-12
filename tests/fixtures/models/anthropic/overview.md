# Models overview

Claude is a family of state-of-the-art large language models developed by Anthropic.

## Compare models

If you're unsure which model to use, start with Claude Opus 5. All current models support text and image input, text output, multilingual capabilities, vision, and tool use. Each model's page lists the platforms it's available on.

| Feature | Claude Fable 5.1 | Claude Opus 5 | Claude Sonnet 5 | Claude Haiku 4.5 | Claude Index Only |
| --- | --- | --- | --- | --- | --- |
| Description | For demanding reasoning | For complex work | Speed and intelligence | Fastest | Listed without a captured detail page |
| Model page | [Claude Fable 5.1](/docs/en/models/fable-5-1/overview) | [Claude Opus 5](/docs/en/models/opus-5/overview) | [Claude Sonnet 5](/docs/en/models/sonnet-5/overview) | [Claude Haiku 4.5](/docs/en/models/haiku-4-5/overview) | [Claude Index Only](/docs/en/models/index-only/overview) |
| Claude API ID | `claude-fable-5-1` | `claude-opus-5` | `claude-sonnet-5` | `claude-haiku-4-5-20251001` | `claude-index-only` |
| Context window | 1M tokens | 1M tokens | 1M tokens | 200K tokens | 100K tokens |
| Max output | 128K tokens | 128K tokens | 128K tokens | 64K tokens | 8K tokens |
| Retirement | Not sooner than September 1, 2027 | Not sooner than July 24, 2027 | Not sooner than June 30, 2027 | Not sooner than October 15, 2026 | Not sooner than January 1, 2028 |
| Claude API alias | `claude-fable-5-1` | `claude-opus-5` | `claude-sonnet-5` | `claude-haiku-4-5` | `claude-index-only` |
| Amazon Bedrock ID | `anthropic.claude-fable-5-1` | `anthropic.claude-opus-5` | `anthropic.claude-sonnet-5` | `anthropic.claude-haiku-4-5` | `anthropic.claude-index-only` |
| Google Cloud ID | `claude-fable-5-1` | `claude-opus-5` | `claude-sonnet-5` | `claude-haiku-4-5@20251001` | `claude-index-only` |

See also [Sparse Test](/docs/en/models/sparse-test/overview) and [No API](/docs/en/models/no-api-access/overview).

Legacy models (still available): Claude Fable 5, Claude Opus 4.8.

## Using the Models API

You can query model capabilities programmatically. JSON responses may include a capabilities object.
