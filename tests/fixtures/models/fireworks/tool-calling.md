<!-- official source URL: https://docs.fireworks.ai/guides/function-calling.md -->
<!-- source_id: fireworks-docs-tool-calling -->
# Tool Calling

```python
response = client.chat.completions.create(
    model="accounts/fireworks/models/kimi-k2-instruct-0905",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=tools,
)
```
