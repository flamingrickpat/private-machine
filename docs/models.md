# Configure models

The chat model drives dialogue and cognition. The embedding model finds related
memories. Keep the embedding model and dimensions consistent for an existing home.

1. Copy `config.example.yaml` to `config.yaml`.
2. Set `base_url`, `model`, and `api_key` for both chat presets and embeddings.
3. Set `embedding.dimensions` to the vector size returned by the endpoint.
4. Use `--config` to select another file.

Keep keys and literal MCP headers in the ignored local YAML. Use an empty key
for a local endpoint without authentication.

| Setting | Effect |
| --- | --- |
| `context_window` | Supplies a serving limit when the endpoint cannot report it. |
| `override_context` | Caps the serving limit. |
| `context_input_fraction` | Leaves room for framing and output. |
| `inference_io.completion_cache` | Allows completion replay. Simulations request fresh completions. |
| `inference_io.cache_directory` | Stores completion replay files. |
| `inference_io.prompt_log` | Records chat and embedding evidence. |
| `inference_io.prompt_log_directory` | Overrides the persona prompt directory. |
| `minecraft` | Selects the body endpoint, timeouts, and lease directory. |
| `tool_call.mcp_servers` | Lists the MCP servers the companion can discover. |

Explicit `--config` takes priority. The terminal and console otherwise try
`config.local.yaml`, then `config.yaml`, then `config.example.yaml`.
Relative cache, log, lease, and startup script paths start at the YAML file.

## Move an existing setup

Move the credential named by `api_key_env` into `api_key`. Remove
`api_key_env` and replace each `${NAME}` MCP header with its literal value.
Move embedding overrides into `embedding` and the body URL into `minecraft.mcp_url`.
Set cache and log options under `inference_io`.

Use the [overnight example](workflows/spamuel-baseline.md) to match the baseline.

- [Configuration loader][pm.llm.config.load_config]
- [HTTP requests and replay][pm.llm.http_client.JsonClient]
- [Embedding service][pm.memory.embeddings.EmbeddingService]
- [Prompt logs](subsystems/prompt-log.md)
