# Inference limits

Inference can take time even when an event needs immediate attention.
The model connection limits concurrent requests and bounds the prompt and response.
The inference queue admits work according to priority.

| YAML field | Purpose |
| --- | --- |
| `max_concurrent_requests` | Controls concurrent requests to one endpoint. |
| `request_timeout_seconds` | Includes time queued at the model server. |
| `context_window` and `override_context` | Supply and cap the serving context. |
| `context_input_fraction` | Reserves part of the window for response and framing. |
| `completion_token_limit` | Caps ordinary completions. |
| `retry_completion_token_limit` | Permits a larger completion after output truncation. |

A workflow can select reasoning settings appropriate to its task. The client
translates those settings for the provider. During a live run, use the
[profiler](timeline.md) to compare generation time with waiting time.

- [Model configuration](../models.md)
- [Connection contract][pm.llm.config.Connection]
- [Model client][pm.llm.activation.ChatClient]
