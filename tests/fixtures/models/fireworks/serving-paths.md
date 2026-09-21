<!-- official source URL: https://docs.fireworks.ai/serverless/serverless-modes.md -->
<!-- source_id: fireworks-docs-serving-paths -->
# Serverless Modes

Fireworks Serverless offers three modes:

* **Standard** is the default mode. No `service_tier` parameter is needed.
* **Priority tier** is for workloads that require higher reliability during peak traffic.
* **Fast** is for workloads that require higher speeds.

## Priority tier

To use priority tier, set `service_tier` to `"priority"`.

```bash
curl https://api.fireworks.ai/inference/v1/chat/completions \
  -d '{
    "model": "accounts/fireworks/models/glm-5p2",
    "service_tier": "priority"
  }'
```

## Fast

Fast is not a different model. To use Fast, change the `model` ID as listed below.

| Model | `model` ID |
| --- | --- |
| Kimi K3 Fast | `accounts/fireworks/routers/kimi-k3-fast` |
| GLM 5.3 Fast | `accounts/fireworks/routers/glm-5p3-fast` |
| GLM 5.2 Fast | `accounts/fireworks/routers/glm-5p2-fast` |
| GLM 5.2 Fast (US) | `accounts/fireworks/routers/glm-5p2-fast-us` |

```bash
curl https://api.fireworks.ai/inference/v1/chat/completions \
  -d '{
    "model": "accounts/fireworks/routers/kimi-k3-fast"
  }'
```
