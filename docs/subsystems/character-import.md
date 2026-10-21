# Character import and memory reconsolidation

## Summary

Many users already have a character: a character card and a chat log.
The character import moves such a character into pm in a few commands.
It creates a character home folder, imports the chat log, and runs memory reconsolidation.
Then it adds a few synthetic records, so the companion knows about the move.
After the import, start the app with `--home <folder> --config <models.yaml>` and chat.

Memory reconsolidation is runtime code in `pm.memory.reconsolidation`.
It is a port of the memory consolidation of the old generation (`DynamicMemoryConsolidator` in `private-machine-head/pm/memory_consolidation.py`).
The companion can also start it by itself through the managed workflow `memory_reconsolidation`.

Parent: [experienced memory](experienced-memory.md).
Related: [experience consolidation](memory-consolidation.md), [structures, attention, and the workspace](workspace.md), and [managed work](managed-work.md).

## Quick start

1. Write the character card into a Markdown file, for example `card.md`.
   The card can use the placeholders `{companion_name}` and `{user_name}`.
2. Optional: write the traits into `personality.json`.
3. Export the chat log as CSV (see [CSV format](#csv-format)).
4. Run the commands:

```powershell
uv run --locked python -m pm.app.character_cli init --home homes/ava --companion-name Ava --user-name Bob --card card.md --personality personality.json
uv run --locked python -m pm.app.character_cli import --home homes/ava --csv chat.csv --config config.alpha.openrouter.yaml
uv run --locked python main.py --home homes/ava --config config.alpha.openrouter.yaml
```

`init` refuses a folder that exists and is not empty.
`--force` deletes that folder first, with its memory and receipts.
Other `init` options: `--assistant-card FILE`, `--timestamp-format FORMAT`, and `--capabilities caps.yaml`.
The capabilities file has the keys `supported_capabilities`, `unsupported_capabilities`, and `capability_notes`.
Without it, the home gets the default lists of `pm.constants`.

`import` options: `--limit N` imports only the first N rows. `--no-reconsolidate` skips reconsolidation.
`reconsolidate --home DIR --config FILE` runs reconsolidation again on a saved home.

## Home folder

| Path | Content |
| --- | --- |
| `character.yaml` | The `PersonaConfig` fields without `personality_profile`: names, timestamp format, cards, and capabilities. Unknown keys fail. |
| `personality.json` | User-written traits: `{"<PersonalityProfile field>": 0..1}`. An unknown key fails and names the allowed keys. A missing key keeps 0.5. |
| `ghost.json` | The accepted memory store checkpoint. `init` creates it. The app loads it and saves to it. |
| `ghost_<tick>_<UTC datetime>_<suffix>.json` | A stable backup before each main cognition attempt. Backups have no retention limit. |
| `logs/` | Process logs with UTC time, process ID, thread ID, thread name, and crash traces. |
| `runtime/` | Kernel folders, managed tasks, and other runtime artifacts. |
| `import/` | A copy of each imported CSV and one JSON receipt for each import run. |

`pm.app.character_home.CharacterHome` holds the paths and the loaders.
The shared cache and inference transcripts live outside the home. Read [runtime files and recovery](runtime-files.md).
`load_persona()` merges `personality.json` into the persona.
`open_store()` loads `ghost.json` and applies the persona files, so an edited card takes effect at the next start.
A changed companion or user name fails, because the stored entities and dialogue name the old speakers.
`open_store()` fails with "Run `character init` first" when `ghost.json` is missing.

## CSV format

- The columns are `name,message,timestamp`. The `timestamp` column is optional.
- The file is UTF-8, with or without a byte order mark. Quoted messages can span lines.
- `name` is the companion name or the user name, in any case. Another name fails and names the row.
- A timestamp is ISO 8601 (for example `2026-05-01T20:10:00Z`) or `YYYY-MM-DD HH:MM[:SS]`.
  A time without an offset is UTC. A time with an offset becomes UTC.
- Give every row a timestamp, or give none. Timestamps must not decrease.
- Without timestamps, the import gives the messages times 2 minutes apart.
  The last message is one hour before the import time, so the first live message starts a new session.

`dev/tools/export_chat_csv.py` writes such a CSV from a seed or a checkpoint.
It keeps only the public dialogue of the user and the companion.
`--days D --per-day N` keeps a small sample over several days.

## What the import creates

The import reads only what the live loop reads (`MainCogLoop.prepare_input` in `workflows/main_cognition/host.py`).
So the first live cycle after an import needs no special case.

| Record | Content |
| --- | --- |
| `Entity` | The user (`Human`) and the companion (`Self`). `init` creates them. |
| User `Feature` | One `Dialogue` feature for each user row. Provenance `User`, like `UserInputProxy.submit`. Public and causal. The exchange is already resolved (`exchange_status: resolved`, `public_reply_id`). |
| Companion `Feature` | One `Dialogue` feature for each companion row. Provenance `Self`, addressed to the user (`addressee_entity_id`), already published. |
| `CycleState` | One for each exchange, with a new tick. The mental state is neutral: the personality of the home and the default readouts. The stimulus group holds the user messages. No decision, no dispatch, no codelet output. |
| `GwtTurnRecord` | One minimal v2 turn for each exchange: the stimuli and the exchange records. No summary, experience, or decision. |

An exchange is the user messages and then the companion replies that follow them.
Each exchange commits in its own write batch with a new tick.
The record times are the times of the log.
The metadata keys `executed_at` and `session` hold the log time and a session id.
A silence of two hours starts a new session id.

The import invents no thought, feeling, appraisal, or expectation.
Every imported record has an `ImportMark` in `metadata["import_mark"]` (`pm.records.import_marks`): the import id, the CSV hash, the row, and the origin.
The provenance `source_ref` also starts with `chat_import:`.
The store keeps one consolidation receipt `chat_import:<import id>` with the tick range.

The import works on an in-memory copy of the store.
Only a completed import saves `ghost.json`.
The import writes a `running` receipt into `import/` at the start.
On an error, it replaces it with a failed receipt and raises.
A killed process (for example by a timeout) leaves the `running` receipt.
`ghost.json` then keeps its old content.
A CSV file that this home already imported fails.
A log that starts before the newest dialogue of the home also fails.

## Synthetic transfer records

After reconsolidation, the import adds three records with `ImportMark(origin=transfer, synthetic=True)`:

1. A `WorldEvent` feature from `memory_transfer` at the import time: the companion moved from the earlier chat harness into pm, and the inner life of that time is reconstructed only as summaries.
2. A `SemanticRecord` about the companion. The self shard requires at least one self-model record, so this record usually reaches the self shard.
3. When the companion has a `SelfImage` narrative, a new version of the newest one with one more line about the move.

The texts are in `pm.constants` (`TRANSFER_*`). Their wording is neutral.

## Memory reconsolidation

`pm.memory.reconsolidation.reconsolidate(store, complete=..., embeddings=..., force=...)` runs one pass.
It reads the accepted store, calls the model outside the commit lock, and commits each group of results through `MemoryStore.commit`.
A long run keeps the groups that it finished.
The report is also stored as the consolidation receipt `reconsolidation:<run id>`.

### Stages

1. **Topical clusters.** The unclustered events (causal `Dialogue` and `WorldEvent` features) get contextual embeddings.
   The clustering finds the topics. Each consecutive topic block becomes a topical `MemoryClusterRecord` (level 100) with a summary and a state description.
2. **Temporal clusters.** Each closed range with events and without a summary becomes a temporal `MemoryClusterRecord`:
   time of day (level 6, key `2026-05-01-EVENING`), day (5), week (4, `2026-W17`), month (3), and year (1).
   A logical day runs from 05:00 to 05:00.
3. **Facts.** Each topical cluster without facts gives `DeclarativeFactRecord` records.
   One call extracts the facts with the numbers of the lines that state them. A second call files all facts of the scene.
   Each new claim gets a canonical `SemanticRecord`. An equal claim (same subject and essence) adds the fact to the existing record.
4. **Narratives.** Each kept narrative gets a new `Narrative` version when enough new evidence exists.
   The companion has all eight types. The user has `PsychologicalAnalysis`, as in the old code.

### Sources

| Record | Source fields |
| --- | --- |
| Topical cluster | `included_event_ids`, `min/max_event_id`, `min/max_tick_id`, `lineage.source_event_ids` |
| Temporal cluster | `included_event_ids` (raw range) or `included_cluster_ids` (summary of summaries), `lineage` |
| Fact | `source_cluster_id`, the cited events in `min/max_event_id` and `lineage.source_event_ids` |
| Semantic record | `source_knoxel_ids` (the facts), `lineage` (the first fact and its events) |
| Narrative | `lineage.source_event_ids` (the evidence), `lineage.parent_record_ids` (the previous version) |

The lineage operation is `memory_reconsolidation.<stage>`. The provenance is `System` with `reconsolidation:<run id>`.
Clusters and facts have `memory_kind: experience_summary`.

### Incremental runs

A second run processes only new work:

- An event is clustered when a consolidated topical cluster holds it. The small per-cycle clusters of main cognition (`experience_summary` without reconsolidation lineage) do not count.
- A temporal key that exists is skipped. Only closed ranges count.
- A cluster with `facts_extracted` is skipped. The flag changes in the same batch as the facts.
- A narrative reads only the evidence after the largest source id of its newest version.

A run without new events makes no model call.

### Trigger

`reconsolidation_due(store)` ports `consolidate_memory_if_needed`: a run is due when 60 unclustered events wait.
Without `force`, the topical stage waits for that count. With `force` (the CLI and the workflow), three events are enough.
The other stages always check for work.

### Prompts and calls

The prompts are in `pm/memory/reconsolidation/prompts.yaml`.
Each call is an unconscious, story-framed process of the character's memory, without an assistant frame.
The code fills the names from the persona.
Times in the prompts use `RECONSOLIDATION_TIME_FORMAT` (`Friday, 01 May 2026, 20:10`), not the persona format.
In the first real run, the numeric format `01.05.26` became "January 5" in a summary.
Each answer is one JSON object. Pydantic validates it.
Invalid answers receive repair turns until context space runs out, through the [shared runtime repair loop](../models.md#completion-replay-and-output-repair).
Narratives must contain 50 to 6,000 characters. Length errors receive the same repair turns.
A provider failure (`ModelError`) repeats the call (at most three times).
Transport errors and HTTP 429/5xx retry inside `JsonClient`.
At most four calls of one stage run in parallel.

Images in records reach reconsolidation only as alt text: a tag reads `[image: <alt text>]` in each story line.
Before the first stage, `reconsolidate` writes the missing alt texts with `complete_alt_text`.
See [images in records](multimodal-records.md).

### Mapping to the old code

| Old principle (`DynamicMemoryConsolidator`) | New implementation | Change |
| --- | --- | --- |
| `_compute_contextual_embeddings`: weights `log(1+tokens)` and `1/(1+d)`, window 3 | `clustering.contextual_embeddings` | Neighbors in another session do not count. |
| `_get_optimal_clusters`: agglomerative average linkage, thresholds 1.0 to 0.1, best silhouette for `max(2,n/20) < k < min(n,n/8)` | `clustering.average_linkage`, `cut`, `silhouette`, `optimal_labels` | Pure Python, one dendrogram (nearest-neighbor chain) for all 91 cuts. With too few events, the session numbers are the labels; the old code made no cluster. |
| Label smoothing, `cluster_blocks_with_min_size`, `split_block_keep_order` (8) | `smooth_labels`, `topic_blocks`, `split_block` | A block also ends at a session change. Windows of 400 events keep the quadratic cost small; the last block of a window moves to the next window. The newest block stays open for 60 minutes. |
| Topical summary of all causal features of the block's tick range (`SummarizeTopicConversation`) | `_Run.topical` | Reads only story features (dialogue, events, thoughts, experiences), not appraisal records. |
| `_update_mental_state_of_cluster` (`DescribeMentalStatePeriod`) | `_Run.describe_state` | Reads the cycle states of the tick range (the old code compared ticks with event ids). A neutral imported state counts as "not recorded". The description vector goes to the shared cache, not the checkpoint. |
| `_generate_missing_closed_temporal_ranges` (05:00 day, TOD/day/week/month/year) | `temporal.closed_ranges`, `_Run.temporal` | Ranges come from the event times, not from every day since 2020. One NIGHT segment 22:00 to 05:00; the old two segments could share a key. A long range reads the nearest finer level (day from times of day, week and month from days, year from months), not always the topical clusters. |
| `_extract_declarative_memory` + `_categorize_fact` (one call per fact) | `_Run.facts` | Facts come from the raw scene of topical clusters only (the canon facts also come only from topical clusters). Each fact cites its lines. One filing call per scene adds the semantic layer (subject, record type, epistemic status, essence) and the canonical `SemanticRecord`. |
| `refine_narratives`: rolling buffers of 3000 to 4500 tokens since `last_refined_with_tick` | `_Run._narrative_chain` | The first version needs only 300 tokens, so a small import gets narratives; the old code started from placeholders. The cursor is the largest source id of the newest version. The companion card is shown as background. |
| `consolidate_memory_if_needed` (60 unclustered events) | `reconsolidation_due`, `reconsolidate(force=False)` | Unchanged threshold. |
| Graph memory integration (`CognitiveMemoryManager`) | Not ported | Out of scope. |

## Companion-callable workflow

`workflows/memory_reconsolidation/` is a managed workflow.
`host.build_reconsolidation_tools` binds the live store, the stage model, and the embeddings.
`steps.run` calls the tool with the run's stop event and writes `outbox/report.yaml`.
`workflows/companion.py` registers it in `_managed_workflows` as a singleton.
The route prompt (`workflows/main_cognition/phases/prompts/route.yaml`) uses it with `ToolCall` when the character decides to sort through memories or to sleep on what happened.
The artifact workflow `memory_consolidation` is a different workflow and stays unchanged.

## Limits

- One commit rebuilds the whole store. A very long log takes long to import, also without model calls.
- A topical block can mix reality scopes. Scoped retrieval then skips the summary but keeps its sources.
- Events that arrive in a range after its summary exist do not change that summary.
- Reconsolidation does not build graph memory (`GraphNode`, `GraphEdge`).
