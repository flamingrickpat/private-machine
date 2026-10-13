# Model connections and embedding cache

## Current scope

The model runtime provides connections, explicit activation, direct chat calls, and one embedding service.
The terminal uses these connections for real dialogue and cognitive workflows.
The [PM classification provider](subsystems/workflow-inputs/pm-connections.md) reuses chat presets with lazy activation and an explicit serving bound.

The separate [typed decision client](subsystems/decisions/index.md) is available through pm-coder.
A02 verifies its API without changing existing workflow model choices.
Its [deployment guide](subsystems/decisions/operation.md) describes the independent Laya endpoint.
The alpha's [model policy](work_items/alpha-plan/model-policy.md) uses the normal LLM by default.

Parent: [README](index.md).

## Config

The credential-free example is `config.example.yaml`.
It contains `Default`, `UnitTestEvaluator`, and one `embedding` entry.
Both chat presets initially use the same Qwen endpoint.
This is a shared-backend evaluation, not an independent neutral reviewer.

The embedding request uses `perplexity/pplx-embed-v1-0.6b`.
OpenRouter currently returns `pplx-embed-v1-0.6b` in its response.
The explicit `response_model` field permits that alias. Other model names fail validation.
The request model remains the cache identity, so existing vectors stay compatible.

`config.alpha.example.yaml` also uses OpenRouter embeddings with local Qwen chat on port 8080.
The credential comes from `OPENROUTER_API_KEY`.
The unused local embedding service on port 8081 has a recorded invalid-vector defect.
Read [A08 verification](work_items/verification/a08-experienced-memory.md) for the real failure and recovery evidence.

### Local setup

1. Copy `config.example.yaml` to `config.local.yaml` for local changes.
2. Set `OPENROUTER_API_KEY` in the local process environment.
3. For a literal key instead, configure `embedding.api_key` in that ignored file.
4. For a literal key, remove `embedding.api_key_env`.
5. Start the configured Qwen endpoint before the live tests.
6. Set `PM_MODEL_CONFIG` to select another complete config file for those tests.

The client does not load `.env` files automatically.
The live tests use `config.local.yaml` when it exists. Otherwise, they use the example config.
Library code receives a config path explicitly.
Relative script and cache paths resolve from that file's directory.
Config parsing and connection resolution make no network requests.

### Hosted chat through OpenRouter

A chat preset can use `https://openrouter.ai/api/v1` with `api_key_env: OPENROUTER_API_KEY`.
OpenRouter has no `/props` route, so set `context_window` in the preset.
Context measurement then uses the labelled `tiktoken` estimate.

Workflows switch thinking off with `chat_template_kwargs: {enable_thinking: false}`.
A local llama.cpp server honors this field. OpenRouter ignores it and keeps reasoning on.
For the exact `openrouter.ai` host, [`provider_generation`][pm.llm.activation.provider_generation] adds the equivalent `reasoning: {enabled: false}`.
An explicit `reasoning` field from the caller always wins.
Other hosts receive the request unchanged.

A measured `qwen/qwen3.8-flash` request on 2026-10-13 returned 24 reasoning tokens with only `chat_template_kwargs`.
The same request with `reasoning.enabled: false` returned 0 reasoning tokens.

OpenRouter serves `qwen/qwen3.8-flash` only through Alibaba, at a precision that OpenRouter does not report.
A result from OpenRouter therefore does not establish the behavior of a Q3 build of the same model.

## Code ownership

| File | Responsibility |
| --- | --- |
| `pm/llm/config.py` | Config, presets, credential resolution, redaction |
| `pm/llm/http_client.py` | Shared HTTP requests, retries, cancellation |
| `pm/llm/activation.py` | Finite scripts, readiness, active calls, chat responses |
| `pm/llm/workflow_sessions.py` | Lazy classification sessions, serving context, and redacted service receipts |
| `pm/memory/embeddings.py` | Legacy reads, new cache writes, scoped preparation, cosine search |

These files contain no character policy or cognitive routing.

## Direct debugger path

This example makes real requests without a shell module thread.
Breakpoints inside `activate()` and `complete()` use the production path.

```python
import asyncio
from pm.llm.config import load_config, resolve_connection
from pm.llm.activation import ChatClient, ModelActivation

async def run_once():
    config = load_config("config.alpha.example.yaml")
    connection = resolve_connection(config, preset="Default")
    owner = ModelActivation()
    await owner.activate(connection, live_test=True)
    client = ChatClient(connection, owner, live_test=True)
    result = await client.complete([
        {"role": "user", "content": "Reply with one short sentence."},
    ])
    print(result.choices[0].message.content)

asyncio.run(run_once())
```

An explicit connection needs no config map:

```python
connection = resolve_connection(
    base_url="http://127.0.0.1:8080/v1",
    model="qwen",
    api_key="",
)
```

An omitted selection uses `Default` from the supplied config.
Mixed startup forms and unknown preset names fail.
Resolved connections retain their own credential value. They do not change the process environment.

## Activation and cancellation

An empty `startup_script` uses an existing endpoint.
A configured `.ps1`, `.sh`, or `.py` script must start its server in the background and then exit.
The activation owner checks the exit code and the advertised model alias.
Scripts must manage only their designated processes.

CAUTION: Do not configure `start38_uncensored_nvfp4_long.ps1` directly as an activation script.
Its restart loop never exits. It needs a finite wrapper for automatic activation.
The current setup uses the manually started server and leaves `startup_script` empty.
The runtime does not change that launcher or automatically switch local models.

All chat clients in one application share one `ModelActivation` owner.
The owner rejects activation during an active call.
Generation finishes before the evaluator activates.
The owner cannot control unrelated programs that use the same endpoint.

Calls accept a `threading.Event` named `cancel`.
Cancellation stops supported HTTP waits, readiness waits, or the directly owned startup process.
It does not claim to terminate detached server descendants.
Windows startup processes use hidden windows.

Generation has no response deadline or automatic output cap.
The HTTP client has a connection-establishment timeout, not an inference timeout.
Production transport errors retry until cancellation. `live_test=True` reports them immediately.
Permanent HTTP errors and invalid responses fail with credential-free messages.
The shared HTTP path uses [HTTPX timeout controls](https://www.python-httpx.org/advanced/timeouts/).

A reasoning backend emits hidden reasoning tokens before the visible answer.
The application sends no `max_tokens` cap, so the backend can finish its reasoning and its answer.
A cap sized for the visible answer alone can stop a JSON stage mid-object and report a stage output error.
There is no generation deadline.
The model stops when it is done.

## Context handling

Laya's decision-head budget does not apply to chat or pm-coder.
Direct Python chat sends one supplied message list, while pm-coder owns accumulated execution history and automatic compaction.
Read [Python chat and agent context](subsystems/llm-context.md) for the actual paths, live commands, and verified input boundaries.

## Embeddings and scoped retrieval

The old cache remains read-only at `~/.pm/embeddings` by default.
New entries use `.runtime/embeddings/v1/<model-hash>/`.
Each new JSON entry contains its format version, request model, exact input, dimensions, and vector.
Atomic replacement publishes each complete new entry.

Cache hits require the exact text, model, dimensions, and valid vector values.
The adapter preserves whitespace and reads the old little-endian float64 binary format.
It rejects zero and nonfinite vectors.
Cosine comparison divides by both vector norms.

```python
from pm.memory.embeddings import EmbeddingService

service = EmbeddingService(config.embedding, live_test=True)
visible_text = {record.id: record.content for record in scoped_records}
plan = service.inspect_index(visible_text)  # no network calls
print(plan.hits, plan.missing)
await service.prepare_index(visible_text)  # fetch only missing eligible text
results = await service.search("a shared memory", visible_text)
```

The caller supplies already-scoped records.
The service never enumerates Ghost or searches the whole cache for candidate memories.
Search refuses an incomplete index instead of silently embedding every missing memory.
Queries use the same embedding service.
The real client exposes `request_count` for cache-hit evidence.
The API uses OpenRouter's [embedding endpoint](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request).

## Tests

### Foundation checks

Run these direct tests without a model endpoint:

```powershell
uv run --locked python -m unittest tests.unit.pm.llm.test_config tests.unit.pm.memory.test_embeddings tests.unit.pm.llm.test_activation -v -b
```

These tests use real cache files and real finite subprocesses.
The cancellation tests do not substitute fake model responses.

### Required live checks

Run the model acceptance suite:

```powershell
uv run --locked python -m unittest tests.live.test_models -v -b
```

The suite requires the generation endpoint, evaluator endpoint, embedding credentials, and compatible Spamuel cache.
Missing requirements fail. No test silently skips them.
It performs generation, evaluation, three uncached embedding requests, and an exact legacy cache lookup.
The retrieval check compares a Minecraft memory with unrelated chemistry text.
Generation and evaluation run serially.
Evidence stays under `.runtime/w1/<run-id>/`, outside Git.

Run all current tests:

```powershell
uv run --locked python -m unittest discover -s tests -v -b
```

This command now includes the live tests.
It includes live model, cognition, conversation, and Minecraft checks. Each suite needs its configured endpoints.

## Related guides

Read the [system map](concepts/architecture.md) and [development loop](workflows/development.md).
