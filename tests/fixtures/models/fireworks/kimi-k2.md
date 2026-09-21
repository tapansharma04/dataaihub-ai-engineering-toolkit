<!-- official source URL: https://docs.fireworks.ai/models/kimi-k2.md -->
<!-- source_id: fireworks-docs-model-page:kimi-k2 -->
# Kimi K2 family

`accounts/fireworks/models/kimi-k2-instruct-latest` is an alias for `accounts/fireworks/models/kimi-k2-instruct`.

```python
response = client.chat.completions.create(
    model="accounts/fireworks/models/kimi-k2-instruct",
    messages=messages,
    max_tokens=512,
    tools=tools,
)
```

Family guidance is not copied onto sibling serving IDs.
