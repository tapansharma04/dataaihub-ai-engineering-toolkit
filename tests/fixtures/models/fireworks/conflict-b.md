# Vision Models

```python
response = client.chat.completions.create(
    model="accounts/fireworks/models/conflict-test",
    messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]
)
```
