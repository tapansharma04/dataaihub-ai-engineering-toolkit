<!-- official source URL: https://docs.fireworks.ai/guides/querying-text-models.md -->
<!-- source_id: fireworks-docs-text-models -->
# Text Models

Query text models through the Fireworks chat completions API.

```python
response = client.chat.completions.create(
  model="accounts/fireworks/models/deepseek-v3p1",
  messages=[{"role": "user", "content": "Hello"}]
)
```

Dedicated deployments use `accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID>`.
