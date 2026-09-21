<!-- official source URL: https://docs.fireworks.ai/guides/querying-vision-language-models.md -->
<!-- source_id: fireworks-docs-vision-models -->
# Vision Models

Vision-language models process text and images.

```python
response = client.chat.completions.create(
    model="accounts/fireworks/models/kimi-k2p5",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ]
        }
    ]
)
```
