# Model connections and embedding cache

## Current scope

The model runtime provides connections, explicit activation, direct chat calls, and one embedding service.
The terminal uses these connections for real dialogue and cognitive workflows.
The [PM classification provider](subsystems/workflow-inputs/pm-connections.md) reuses chat presets with lazy activation and an explicit serving bound.

The separate [typed decision client](subsystems/decisions/index.md) is available through pm-coder.
A02 verifies its API without changing existing workflow model choices.
Its [deployment guide](subsystems/decisions/operation.md) describes the independent decision endpoint.
The alpha's [model policy](work_items/alpha-plan/model-policy.md) uses the normal LLM by default.
The endpoint can be local or remote.
The policy states the required capability class, not one server or one model.

Parent: [README](index.md).

## Config

The credential-free example is `config.example.yaml`.
It contains `Default`, `UnitTestEvaluator`, and one `embedding` entry.
Both chat presets initially use the same endpoint.
This is a shared-backend evaluation, not an independent neutral reviewer.

The endpoint can be local or remote.
A local server, vLLM, llama.cpp, or a hosted API is valid when it speaks the OpenAI API.
The YAML file records the base URL and the model names.
A process environment variable or a repository `.env` file supplies the credential.
The process environment wins. The `.env` file is the fallback.

A Windows environment variable and a repository `.env` file are both valid credential sources.
Use the environment variable on one machine and the `.env` file on another.

### Current models

Development selects `config.alpha.openrouter.yaml` and uses `qwen/qwen3.8-flash` for chat and
`perplexity/pplx-embed-v1-0.6b` for embeddings.
OpenRouter returns `pplx-embed-v1-0.6b` in its response.
The explicit `response_model` field permits that alias. Another model name fails validation.
The request model remains the cache identity, so existing vectors stay compatible.

The embedding service accepts any OpenAI-compatible embedding model.
A server that returns the requested model name needs no alias field.
The unused local embedding endpoint on port 8081 has a recorded invalid-vector defect.
Read [A08 verification](work_items/verification/a08-experienced-memory.md) for the real failure and recovery evidence.

### Local setup

1. Copy `config.example.yaml` to `config.local.yaml` for local changes.
2. Put the credential into the process environment, or into the git-ignored `.env` in the repository root.
3. For a literal key instead, configure `api_key` in an ignored file.
4. For a literal key, remove `api_key_env`.
5. Start the configured chat endpoint before the live tests.
6. Set `PM_MODEL_CONFIG` to select another complete config file for those tests.

Credential resolution reads the process environment first, then the repository `.env`.
[`pm.llm.env_file.credential`][] returns the value without writing it into `os.environ`.
`PM_ENV_FILE` selects another file.
The live tests use `config.local.yaml` when it exists. Otherwise, they use the example config.
Library code receives a config path explicitly.
Relative script and cache paths resolve from that file's directory.
Config parsing and connection resolution make no network requests.

### Endpoint behavior and context window

A hosted endpoint can have no `/props` route.
OpenRouter supplies its selected model's `context_length` through the model catalog.
For another endpoint without compatible discovery, set `context_window` in the preset.
Context measurement then uses the labelled `tiktoken` estimate.
The serving model must offer at least 128,000 tokens of context.
Read the [model policy](work_items/alpha-plan/model-policy.md) for the capability requirement.

Workflows switch thinking off with the neutral `enable_thinking: false` option.
[`provider_generation`][pm.llm.activation.provider_generation] translates that option inside the LLM wrapper.
Compatible local endpoints receive `chat_template_kwargs: {enable_thinking: false}`.
OpenRouter receives `reasoning: {enabled: false}` without the local template field.
An explicit `reasoning` field from the caller always wins.
Other generation fields retain their caller-supplied values.

A measured `qwen/qwen3.8-flash` request on 2026-10-13 returned 24 reasoning tokens with only `chat_template_kwargs`.
The same request with `reasoning.enabled: false` returned 0 reasoning tokens.

A hosted service may not report the served precision.
A result from a hosted service therefore does not establish the behavior of a local quantized build of the same model.

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
Production waits through connection failures, HTTP 429, and HTTP 5xx, with a 30-second pause before each retry.
There is no production retry quota. Temporary unavailability must not become a cognitive failure.
Cancellation interrupts request I/O, readiness, and retry pauses.
Diagnostics can set an explicit `transient_retries` bound. Exhaustion then raises `ModelError` and preserves the failed attempts.
`live_test=True` selects immediate failure for endpoint boundary tests.
The A18B evaluator and Jesse actor use the normal production wait policy.
A separate test deadline can stop a run as incomplete evidence; it does not establish a negative model judgment.
Retries repeat inference, so a lost response can cause duplicate provider work or charges.
They do not repeat tool execution or regenerate a valid low-scoring review.
Permanent HTTP errors and invalid responses fail with credential-free messages.
A provider can also return HTTP 200 with `finish_reason=error`, `content_filter`, or an embedded completion error.
[`pm.llm.http_client.ProviderCompletionError`][] rejects that partial content before a stage parser or public reply can use it.
Provider-filtered output is not a quota wait and is not retried to obtain another answer.
An ordinary completed refusal remains a valid model response.
The shared HTTP path uses [HTTPX timeout controls](https://www.python-httpx.org/advanced/timeouts/).

A reasoning backend emits hidden reasoning tokens before the visible answer.
The application sends no `max_tokens` cap, so the backend can finish its reasoning and its answer.
Direct private-thought calls now use separate [reasoning allowances](subsystems/inference-settings.md), with visible-answer reserves and explicit token ceilings.
A cap sized for the visible answer alone can stop a JSON stage mid-object and report a stage output error.
There is no generation deadline.
The model stops when it is done.

## Context handling

Laya's decision-head budget does not apply to chat or pm-coder.
Direct Python chat sends one supplied message list, while pm-coder owns accumulated execution history and automatic compaction.
Read [Python chat and agent context](subsystems/llm-context.md) for the actual paths, live commands, and verified input boundaries.
Quality takes priority over minimizing calls. Hard reasoning and review can use smaller, focused calls with explicit integration.
This [prompt-scope rule](work_items/alpha-plan/model-policy.md#prompt-scope-and-decomposition) applies to all prompts, not only reviewers.

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
