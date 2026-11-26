# Character import and memory reconsolidation

## Summary

Many users already have a character: a character card and a chat log.
The character import moves such a character into pm in a few commands.
It creates a character home folder, imports the chat log, and runs memory reconsolidation.
Then it adds a few synthetic records, so the companion knows about the move.
After the import, start the app with `--home <folder> --config <models.yaml>` and chat.

The importer and the live companion share the same memory reconsolidation pass.
During conversation the companion can choose when to run it and whether to sleep while it works.

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
uv run --locked python -m pm.app.character_cli import --home homes/ava --csv chat.csv --config config.yaml
uv run --locked python main.py --home homes/ava --config config.yaml
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
| `ghost.pickle` | The accepted memory store checkpoint. `init` creates it. The app loads it and saves to it. |
| `ghost_<tick>_<UTC datetime>_<suffix>.json` | A stable backup after each successful main cognition commit. Backups have no retention limit. |
| `logs/` | Process logs with UTC time, process ID, thread ID, thread name, and crash traces. |
| `runtime/` | Kernel folders, managed tasks, and other runtime artifacts. |
| `import/` | A copy of each imported CSV and one JSON receipt for each import run. |

`pm.app.character_home.CharacterHome` holds the paths and the loaders.
The shared cache and inference transcripts live outside the home. Read [runtime files and recovery](runtime-files.md).
`load_persona()` merges `personality.json` into the persona.
`open_store()` loads `ghost.pickle` and applies the persona files, so an edited card takes effect at the next start.
A changed companion or user name fails, because the stored entities and dialogue name the old speakers.
`open_store()` fails with "Run `character init` first" when `ghost.pickle` is missing.

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

The import saves the validated dialogue and a source receipt before it starts memory work. Each complete unit then gets its own checkpoint. On interruption, repeat the import with the same CSV, row limit, and reconsolidation option. The source receipt identifies the original import and prevents duplicate dialogue. The final checkpoint includes the transfer records and completed import receipt. A completed CSV cannot be imported twice. Import older logs before newer dialogue. [Import implementation][pm.app.character_cli.import_chat]

## Synthetic transfer records

After reconsolidation, the import adds three records with `ImportMark(origin=transfer, synthetic=True)`:

1. A `WorldEvent` feature from `memory_transfer` at the import time: the companion moved from the earlier chat harness into pm, and the inner life of that time is reconstructed only as summaries.
2. A `SemanticRecord` about the companion. The self shard requires at least one self-model record, so this record usually reaches the self shard.
3. When the companion has a `SelfImage` narrative, a new version of the newest one with one more line about the move.

The texts are in `pm.constants` (`TRANSFER_*`). Their wording is neutral.

## Memory reconsolidation

A shared episode records what happened with Jesse and how Spamuel interpreted it. PM keeps one episode with the participants, time, and links to the original evidence. The source records keep their speaker, addressee, scope, and wording. The calendar hierarchy lets Story reach farther back when the originals exceed its prompt budget. [Episode membership][pm.memory.cluster_membership.ClusterMembership]

### When the companion starts work

Main cognition measures the external dialogue and observations still available to file in each visible reality and frame. It estimates their text cost locally. A short recent exchange stays open. During a long continuous conversation, older material can already form episodes. An earlier rendered Story can also report which history actually fitted its prompt. The decision receives a dated cue when there is work. The companion chooses whether and when to act. [Assessment][pm.memory.reconsolidation.assessment.MemoryWorkReader] and [decision prompt][workflows.main_cognition.phases.decide.build_messages]

Both choices use the managed singleton `memory_reconsolidation`. The route supplies `reconsolidation_mode` on its tool call.

| Mode | While it runs | Publication |
| --- | --- | --- |
| `background` | Conversation and ordinary thought continue. Foreground requests have priority when a model slot opens. | A successful main cycle accepts finished units. |
| `sleep` | Cognitive control parks routine inference for the approved action. A waking stimulus or the matching worker's terminal result releases the wait. | Wake closes the worker's admission, applies ready units, and saves before the waking input enters memory. |

An active model request can still occupy its slot. The wake boundary can also spend time serializing the character's full checkpoint. Clean shutdown drains finished work. Recovery opens a sleeping character awake when the process that owned preparation is gone. [Managed worker][workflows.memory_reconsolidation.host.ReconsolidationRuntimeOwner], [publication queue][pm.memory.reconsolidation.coordination.ReconsolidationCoordinator], and [cognitive control][workflows.cognitive_control.host.CognitiveControl]

### How the pass forms memories

1. **Find an episode.** The worker freezes the accepted history and groups anchors by reality and frame. It uses contextual embeddings and agglomerative clustering in bounded windows. A session gap separates scenes. The last recent topic stays open. Dialogue and world observations usually anchor the scene. A retained private finding or a thought used in a main decision can also anchor one. [Source selection][pm.memory.reconsolidation.membership.EpisodeFrontier] and [clustering][pm.memory.reconsolidation.clustering.topic_blocks]
2. **Write the episode and file its facts.** The recollection gives named dialogue the most detail. It can describe an interpretation or a consequential train of thought in the third person. Its membership lists the covered events. Its lineage identifies which originals supplied the description. Facts cite individual external source lines. An entity-linked filing carries its observer's accepted claim, subject, scope, and receipt. A later correction or challenge changes its eligibility for current recall. The classifier can link a paraphrase to a prior semantic record for the same subject and source authority. Its new factual occurrence records the new citation. The original occurrences remain available through historical links. [Filing][pm.memory.reconsolidation.facts.EpisodeFacts] and [entity context](entity-context.md)
3. **Build the calendar hierarchy.** Closed dayparts, days, weeks, months, and years use the character's world timestamps. The logical day starts at 05:00 UTC. A period waits for its eligible episodes. Short periods use original evidence. A longer period can combine verified child summaries with original evidence for the uncovered events. A later source can produce a new version of a previously filed period. [Period selection][pm.memory.reconsolidation.pipeline.ReconsolidationPass.temporal_units] and [calendar ranges][pm.memory.reconsolidation.temporal.closed_ranges]
4. **Refine narratives.** A narrative draws on completed days and the evidence types for that narrative. It waits for enough new material. The new version records the consumed sources and links to its predecessor. [Narrative refinement][pm.memory.reconsolidation.pipeline.ReconsolidationPass.narrative_units]

The worker prepares one complete unit at a time in its own working view. An episode and its factual filings become ready together. Publication checks the source heads, filing dependencies, and accepted claim families. A change to the evidence defers the affected unit. Unfinished work stays eligible for a later pass. Completed units and their receipts share a store commit. The offline command checkpoints after each accepted unit. [Preparation and publication][pm.memory.reconsolidation.publication.MemoryPreparation] and [offline command][pm.app.character_cli.reconsolidate_home]

### What Story reads

Story can use a new episode before it has a daily parent. Its planner checks the recorded source IDs. An event between two members of a sparse episode still needs its own representation. The selected history, exact omissions, and estimated prompt cost are available in the renderer capture. Earlier companion speech retains the assistant role. Private material and third-person recollections enter the context as evidence. The recent exchange and current mental context retain their places at generation time. [Temporal history][pm.memory.temporal_context] and [Story rendering][workflows.dialogue_story.history_rendering]

The pass writes episodic and temporal clusters, factual occurrences, semantic records, and narrative versions. Graph nodes and edges have separate owners. The synchronous artifact review workflow also has its own purpose. [Experience consolidation](memory-consolidation.md)
