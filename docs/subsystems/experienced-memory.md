# Experienced memory

## Summary

Native agents can also use [direct scoped memory tools](agent-memory.md).
That path keeps search optional and gives the agent original records for its own judgment.

A08 retrieves original semantic and episodic evidence for live cognition.
Selection uses a stable permitted snapshot, a causal frontier, and reality/frame scopes.
The selected records reach action selection and dialogue with their original IDs.
Imported canon remains attributed background.

Parent: [cognitive pipeline](cognition.md).
Related: [accepted state](cognition-state.md), [emotional dynamics](emotional-dynamics.md), and [memory experiments](../workflows/memory.md).
Children: [consolidation and beliefs](memory-consolidation.md) and [A08 verification](../work_items/verification/a08-experienced-memory.md).
Direct agent access: [A14B memory search and graph tools](agent-memory.md).

## Entry and execution path

[`workflows.main_cognition.host.MainCogLoop.prepare_input`][] resolves stimuli through the invocation's permitted view.
[`pm.memory.bash_view.MemoryBashTarget.open`][] retains that detached view and its record revisions.
New host workflow events remain explicit stimuli.
They cannot expose memory appended after the invocation began.

[`pm.memory.store.MemoryStore.retrieve_experiences`][] calls [`pm.memory.retrieval.retrieve`][].
The newest stimulus supplies the `(tick_id, sub_tick_id)` frontier.
The stimulus scopes supply the allowed `(reality_id, frame)` pairs.
An empty permitted ID set exposes no memory.
The default trusted host view contains all accepted IDs.

[`pm.memory.retrieval.scoped_records`][] applies these boundaries before any embedding request.
Each record requires its typed references within the same boundaries.
This includes speaker Entity IDs in a restricted source grant.
A summary requires every linked source inside the same boundaries.
The rule also applies to a summary of another summary.
Missing legacy references therefore exclude an abstraction without discarding its visible originals.

## Retrieval and context limits

The eligible classes include dialogue Features, declarative and semantic records, beliefs, narratives, clusters, causal patterns, and body evidence.
The selector scans the whole eligible snapshot instead of only recent dialogue.
It combines cached cosine similarity with weighted distinctive terms.
Their weights are `0.65` and `0.35` respectively.
Recent context reserves four positions in the default 24-result limit.

Each invocation prepares at most 64 uncached chunks from the lexical, recent, and important pools.
Already cached vectors participate across the full eligible snapshot.
Missing coverage remains explicit in `memory_trace.coverage`.
Incomplete coverage limits purely semantic recall of an unindexed record without matching terms.
Full preparation removes that index limitation.
[`pm.memory.retrieval.index_chunks`][] splits long records into 2,000-character chunks with 200 characters of overlap.
Each chunk retains its original record ID.
The record's semantic score uses its strongest matching chunk.
Coverage counts distinguish complete records from indexed chunks.

Each text excerpt contains at most 1,200 source characters, plus attribution.
For long records, the excerpt starts near a rare matching term.
A semantic-only match keeps its strongest matching chunk.
The original record remains available by ID.
Embedding vectors stay in the shared cache, outside checkpoints.
Network ranking never holds the Ghost commit lock.

The active alpha configuration uses OpenRouter's `perplexity/pplx-embed-v1-0.6b` with 1,024 dimensions.
`OPENROUTER_API_KEY` supplies its credential.
The request model identifies its cache namespace.
Local CPU comparison vectors use another namespace and cannot replace remote vectors.
The gate uses OpenRouter's [embedding model catalogue](https://openrouter.ai/docs/api/api-reference/embeddings/list-embeddings-models) to verify the active model.

## Attribution and selection

[`workflows.main_cognition.schemas.MemoryCandidate`][] retains the ID, revision, scope, provenance, source IDs, timestamp, score, and selection reasons.
Its text identifies the source and evidence category.
The rendered `recorded_at` UTC timestamp preserves chronology in both dialogue paths.
Actual event metadata takes precedence over the legacy world timestamp.
Historical world observations require reobservation before a current-world claim.
Inactive or superseded beliefs remain historical evidence.

The Context LLM selects only supplied IDs.
It returns an empty `memories` list to avoid copying source passages through model generation.
[`workflows.main_cognition.steps.run`][] replaces its generated memory strings with the exact selected candidate text.
The route call receives that context explicitly.
The report, GWT, appraisal trace, action metadata, and dispatched payload retain the evidence chain.
The dialogue payload uses this selection instead of another recent-memory search.
[`workflows.dialogue_story.schemas.StoryRequest`][] also retains the accepted `selected_action` intent.
Story includes that intent beside the current user message in its fourth contract message.
Recall intents identify the supported episode and original evidence IDs.
Recent generated replies cannot prove that an older event happened.

The trace records index coverage, scope, frontier, competing scores, and rejected boundary counts.
It retains up to 16 competing records and 40 boundary rejection examples.
`context_selected_ids` distinguishes retrieval candidates from the LLM's final selection.

## Commit and concurrency

[`pm.memory.write_batch.WriteBatch.expect_revision`][] guards the input records, selected memories, and their linked sources.
A changed revision rejects the complete cognition batch with `GhostConflict`.
An unrelated concurrent append does not alter the frozen selection.
Appraisal, beliefs, experience summaries, goals, frame, and actions commit together.
The checkpoint save precedes publication.

Cancellation interrupts embedding requests through the workflow's stop event.
Completed vectors remain cached.
Invalid provider vectors never enter the cache.
A permanent provider error switches retrieval to scoped lexical and recent evidence.
`memory_trace.semantic` records the error and fallback.
It also identifies the configured endpoint and request model.
This fallback preserves dialogue while semantic ranking remains unavailable.
Local cache errors and cancellation still propagate.
Production retries transient transport errors five times, with a 30-second pause before each retry.
Cancellation interrupts requests and retry pauses.
Explicit live-test clients fail immediately for service diagnosis.

## Prepare and inspect the index

Run an inspection before preparation:

```powershell
uv run --locked python dev/tools/alpha_memory_index.py --config config.alpha.example.yaml --out .runtime/alpha/a08/index-inspection.json
```

For a specific accepted checkpoint, supply `--checkpoint PATH`.
For a restricted grant, supply `--ids PATH` with a JSON array of permitted IDs.
Supply `--reality ID --frame in_world` or `--frame meta` to restrict the reality/frame scope.
Then add `--prepare` to fetch missing vectors.
For remote preparation, `--workers 8` permits eight independent requests at a time.
The default uses one worker.
Every worker receives only a part of the same permitted snapshot.
Cancellation or a failed request stops the preparation group and retains completed cache entries.
Add `--query TEXT` to retain a retrieval trace in the receipt.

The console's existing **Embed missing records and search** operation remains available.
It preserves its type/subtype filters and cancellation control.
Console inspection search and live cognition have separate selection contracts.
The [memory experiment procedure](../workflows/memory.md) describes the console controls.

## Seed and restart

[`pm.memory.store.MemoryStore.from_yaml`][] marks imported records as `background` with `SourceKind.Migration` provenance.
Metadata preserves the source path, file SHA-256, and original provenance.
An imported utterance proves what the source says, rather than an action by this runtime.
It cannot establish past physical senses or an unsupported emotion.
The seed file stays unchanged.

An existing checkpoint takes precedence over the seed.
Restoration retains accepted attribution without importing those records again.
Old checkpoints retain their existing provenance and do not silently acquire the fresh-import policy.
A fresh instance establishes the explicit canon baseline.

## Diagnose recall

1. Find the original record by its ID in the accepted snapshot.
2. Inspect `memory_trace.coverage`, the frontier, and the allowed scopes.
3. Inspect competing scores and boundary rejection examples.
4. Inspect `context_selected_ids` and the action's `memory_evidence`.
5. Compare the public claim with the original source text.

Record-count equality establishes persistence only.
A correct public claim also needs relevant original evidence and appropriate attribution.
[A08 verification](../work_items/verification/a08-experienced-memory.md) separates these results from A09's public-voice work.
