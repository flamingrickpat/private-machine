# Models and memory

## Summary

Chat workflows select a named preset or a complete explicit connection.
All semantic retrieval uses one embedding service and a compatible cache.
Ghost owns memory changes and persistence.

Parent: [implementation handoff](../../IMPLEMENTATION_PLAN.md).
Related: [runtime](runtime.md) and [test requirements](tests.md).

## Chat model config

Define `LlmPreset` as a string enum with `Default` and `UnitTestEvaluator`.
Future roles such as `Roleplay` require an enum value and a config entry, not a runtime branch.

Each connection contains `base_url`, `model`, credentials, and `startup_script`.
The URL includes the port.
Credentials come from either `api_key` or `api_key_env`, never both.
The resolver reads an environment reference once and passes the resolved secret directly to the client.
It never changes process environment variables to select a model for another thread.
Traces and config dumps redact credentials.

The two startup forms are exact:

- `preset=LlmPreset.Default`, or the equivalent config name.
- Explicit `base_url`, `model`, `api_key`, and optional `startup_script`.

Omitted selection means `Default`.
An explicit connection does not require a preset map.
Supplying both forms is an error. Unknown names never fall back silently.

Initially, `Default` uses `http://127.0.0.1:8080/v1` and model alias `qwen`.
Its existing launcher is `C:\source\ai\start38_uncensored_nvfp4_long.ps1`.
`UnitTestEvaluator` is a separate required entry.
It can explicitly point to the same backend until the user selects a separate reviewer.
Evaluation artifacts must record this shared-backend condition. Do not call it an independent neutral-model evaluation.

No latency threshold appears in acceptance or model selection.
Disable inherited generation wall-clock limits for these companion runs, including limits inside the selected agent driver.
Simulation time comes from explicit events or ticks, not time spent in inference.

## Model activation

An empty `startup_script` uses an existing endpoint.
A relative path resolves against the config file directory.
Activation is explicit before a run group, not a side effect of every request or object constructor.

Activation scripts start the intended server in the background and then exit.
They operate only on processes that the user designated for model switching.
They must not terminate every process named `llama-server`.
The existing Qwen launcher contains a restart loop, so activation needs a finite wrapper.
Stopping only its child server can cause the old launcher to restart the wrong model.

Nonzero script exit is an activation error.
After successful startup, readiness waits remain cancellable without a generation-speed deadline.
The endpoint must advertise the expected model identity.
Models that share a port need distinct served aliases when their identities differ.

The evaluator phase starts only after all generation calls finish.
The activation owner must not swap a backend while another active workflow uses it.
Embedding service lifetime is independent of chat-model switching.

## Embeddings: fixed selection

| Setting | Value |
| --- | --- |
| Base URL | `https://openrouter.ai/api/v1` |
| Model | `perplexity/pplx-embed-v1-0.6b` |
| Expected dimensions | `1024` |
| Credential source | `OPENROUTER_API_KEY` from local, ignored config or process environment |
| Startup script | Empty |
| Existing cache | `C:\Users\rick\.pm\embeddings` |

Use one `embedding` config entry, separate from the chat preset map.
Do not add per-workflow embedding overrides.
OpenRouter lists this model and provides the embeddings endpoint.
Sources: [model listing](https://openrouter.ai/perplexity/pplx-embed-v1-0.6b), [embedding API](https://openrouter.ai/docs/api_reference/embeddings).

The current old `.env` uses this endpoint and model.
The inspected cache contains 489 records naming this model and 1,024 dimensions.
Ninety-three cached inputs mention Spamuel or Jesse and match exact text in the seed.
This proves useful existing coverage, not complete coverage of every seed record.

Reuse the legacy cache format through a small adapter.
Its key is `md5(content)`, with model metadata in JSON and vector values in a binary sibling.
The binary starts with a little-endian unsigned 64-bit count, followed by float64 values.
The existing cache service and binary reader provide the source implementation.

Legacy input formatting is raw content. Do not add EmbeddingGemma prefixes or change whitespace during migration.
Require matching input, model, dimensions, and valid vector data for a cache hit.
Treat the existing cache as read-only during migration.
Write new entries to a versioned application cache, without replacing old records.
API key changes do not invalidate embeddings.

Some vectors are not unit-normalized. One inspected cache vector has an L2 norm near 4.604.
Use proper cosine comparison, or normalize both vectors locally before a dot product.
Do not mistake the existing dot-product helper for a general cosine implementation.
Local normalization does not require a new API request or a changed legacy cache.

Fetch only uncached eligible text.
An explicit index-preparation step reports hits and missing entries before semantic retrieval starts.
Do not trigger remote requests merely by loading Ghost in a debugger.
Queries and new memories also use the same service.
Changed content creates a new cache key.
Hidden records must remain excluded from a scoped search, even if their vectors exist in the shared cache.

The Spamuel seed contains 11,202 empty `embedding` fields.
It is byte-identical to the original unscoped archive inspected on the plan date.
Its large `mental_state_latent_vec` arrays are not semantic text embeddings.
Do not regenerate or resize those mental-state arrays.

## Ghost reads and proposals

Retain the existing knoxel classes, identities, scopes, provenance, and indexes.
Do not replace Ghost with a generic dictionary.
Only the Ghost service mutates canonical memory.
MCP and Bash expose the same scoped snapshot, plus proposals accepted within the current run.

Define one application-level `GhostWriteBatch` for a cognitive run.
It contains proposed new knoxels, proposed replacements, and expected revisions for replaced records.
The service reserves integer IDs under its lock, so later child stages can reference proposed records.
Unused reserved IDs can leave gaps. They are not visible memories.
The batch exposes a read view over the snapshot and its proposals.
This is a small staging object, not a database or general transaction engine.

Scoped MCP writes create proposals in this batch.
Python finalizers validate model output once at the external boundary, then add proposals through the same scoped operations.
Child workflows share the parent batch with narrower grants and separate artifacts.
Other runs cannot see these proposals before commit.

Permissions specify visible records and allowed operations on knoxel types or subtypes.
Creation, update, and deletion are distinct permissions.
Deletion is not part of the initial cognitive grants.
An explicit deny wins over an allow.
The default is read access within the supplied view, not access to all Ghost records.

## Commit and persistence

At successful parent completion, validate references and expected revisions, then commit the complete batch with its GWT record.
An unrelated append does not invalidate a batch. A conflicting update to the same record does.
A conflict rejects the batch without partial canonical changes and reports the affected records.
Do not merge conflicting model-generated updates silently.
Cancellation or invalid output discards proposals and retains diagnostic artifacts.

Use a versioned application snapshot around the existing serialized Ghost state.
Include GWT records, ID counters, record revisions, applied commit IDs, and managed-run receipts.
Keep semantic vectors in their cache, not in every snapshot.
Preserve original seed loading as a separate supported input.

Serialize a candidate checkpoint under the commit lock, write a temporary sibling, and atomically replace the prior checkpoint.
Only then install the canonical state and publish bus output.
A failed save preserves the previous committed checkpoint and produces an error.
No LLM or network call holds the Ghost lock.
Repeated submission of the same commit ID returns the prior receipt without duplicate memories.

Restart restores accepted memory, not live Python threads or external tool state.
Previously active external runs become `interrupted` until reconciled.
Never repeat uncertain external side effects automatically.
Transparent workflow resume and durable bus replay are outside this delivery.
