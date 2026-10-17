# Asynchronous conversation review

## Summary

The conversation runner connects an independent Jesse model to the full companion application.
Both sides can compose while new messages arrive.
Review uses the complete public conversation in delivery order.
Separate records retain the causes and the context visible during composition.

Parent: [public publication](public-publication.md).
Related: [private loop](private-loop.md), [Minecraft](minecraft-actor-controls.md), and [development review](../workflows/development.md).
For the larger scenario repertoire and automated blinded review, use [A18B cognitive evaluation](cognitive-evaluation.md).

## Run a conversation

The runner is `dev/work_items/conversation_review.py`.
Its default model configuration is `config.alpha.openrouter.yaml`.
It uses actual inference for Jesse and every companion workflow.
Jesse has a separate model client and no runtime tools.

```powershell
uv run --locked python dev/work_items/conversation_review.py `
  --out .runtime/alpha/NEW-long-gap --condition long-gap `
  --start-minecraft --reset-world

uv run --locked python dev/work_items/conversation_review.py `
  --out .runtime/alpha/NEW-recent --condition recent `
  --start-minecraft --reset-world
```

Run these commands sequentially. Each fixture owns the same Minecraft ports.
The server is `C:/source/minecraft-server-real`.
The MCP checkout is `../pm-minecraft`, with its installed interpreter and `.env_real` configuration.
The runner refuses occupied ports and validates every removal target before a reset.
It isolates the character files and stops its owned services after the conversation.
Application cleanup signals and joins managed workers before the services stop.
The world reset changes real world data and requires `--start-minecraft`.

The runtime keeps its normal main loop, private loop, managed work, and world polling.
Minecraft availability does not prove that the companion chose to observe or perform work.
Inspect actual observations, body journals, run receipts, and world effects before claiming execution.

## Human situation and timing

`long-gap` imports the unchanged full Spamuel canon.
Jesse knows that months passed during shutdown and acknowledges that absence.
`recent` imports the same records into a separate store and shifts all structured timestamps by one recorded offset.
The newest source timestamp becomes yesterday.
Record text, identity, order, and intervals remain unchanged.
`history-transform.json` records the original seed digest, offset, endpoints, and record count.
Historical date words inside prose remain preserved and can limit this comparison.

Jesse receives the last twenty public seed messages and the actual public chat available at composition time.
Its prompt includes its human situation, conversational style, and a topic.
It receives no private thoughts, memory records, tool receipts, trace IDs, or expected answers.
The topic comes from a recorded random seed or `--topic`.

The default target is ten Jesse messages.
Some messages form short bursts. Other messages allow a bounded pause for public activity.
This pause does not require a corresponding reply.
The final settling interval permits later companion messages.
The runtime decides how often it speaks; twenty total messages are an approximate review target.

## Evidence and errors

| File | Meaning |
| --- | --- |
| `chat.jsonl` | Actual delivery order, record IDs, causes, and generation metadata |
| `jesse-setup.json` | Human situation, topic, and initial public history |
| `jesse-inputs/` | Exact public snapshots and send times for each composition |
| `llm-logs/` | Actual model requests and responses |
| `origins.json`, `source/` | Repository revision, source bytes, installed package origins, and seed digest |
| `minecraft-fixture.json`, `services/` | Server seed, character isolation, service logs, and reset facts |
| `receipt.json` | Completed experiment, messages, failures, run receipts, and pending human review |
| `failure.txt`, `runtime-error.txt` | Tracebacks when execution fails |
| `review-bundle.json` | One whole-conversation review item |

The capture subscribes to actual `user.input` and `dialogue.reply` events.
It deduplicates repeated delivery by record ID.
It never reads an active checkpoint from another process.
New messages do not replace a draft's earlier causal context.

A runtime error stops the application and fails the experiment.
The runner retains partial chat and tracebacks.
The application records a clean stop only after all owned workers finish without an application error.
Missing companion speech cannot become a pass.
Successful completion establishes observed execution only. It supplies no automatic immersion rating.

## Register and review

```powershell
uv run --locked python dev/review/server.py register `
  .runtime/alpha/YOUR-RUN/review-bundle.json
```

The review item shows the whole conversation with UTC delivery timestamps.
Read messages in their displayed order.
Overlapping composition can produce `Jesse 1, Jesse 2, Spamuel 1, Spamuel 2` without a routing defect.
Use causal diagnostics to investigate attribution; do not infer causes from arrival position.
Keep the old failures and human feedback when a later run supplies repaired evidence.

## Read private activity beside the conversation

The A19 review continuation is `dev/work_items/alpha_a19_review.py`.
Its entry point is [`dev.work_items.alpha_a19_review.run`][].
It copies an accepted reflection checkpoint into a fresh evidence directory.
The full application resumes pending speech before the script supplies later story and scheduling messages.
The private revision comes from the accepted source run. The continuation captures subsequent public behavior.
Normal private timer activity waits for one hour during this focused conversation.
The application still owns recall, codelets, attention, action selection, and public speech.
This fixture supplies Jesse's messages. It does not use the independent Jesse model described above.
World polling stays disabled because this scenario concerns a shared story.

```powershell
uv run --locked python dev/work_items/alpha_a19_review.py `
  --source .runtime/alpha/YOUR-A19-EVALUATION/ghost.json `
  --out .runtime/alpha/YOUR-A19-REVIEW
uv run --locked python dev/review/cognitive_trace.py `
  .runtime/alpha/YOUR-A19-REVIEW --id YOUR-REVIEW-ID
uv run --locked python dev/review/server.py register `
  .runtime/alpha/YOUR-A19-REVIEW/review-bundle.json
uv run --locked python dev/review/server.py serve
```

The runner waits for each source exchange's recorded public outcome.
A reply to an older request cannot complete a newer request.
The deadline is 1,800 seconds. A runtime error or missing reply prevents review registration.
Shutdown stops and joins owned workers before the final receipt reaches disk.
Use `--drain-only` to finish pending exchanges without adding new messages after a restart.
The continuation preserves previously captured delivery rows when its source receipt contains them.

The offline exporter reads the completed checkpoint and copies selected records into an immutable bundle.
See [`dev.review.cognitive_trace.build_bundle`][] for the export contract.
It uses the generic [runtime record-package contract](record-packages.md) for chat, private activity, and each retained GWT turn.
The review server still reads no checkpoint and runs no inference.
The A19 selection starts at record 11203, after the complete 11,202-record canon.
It retains the supplied mistaken belief, actual revision, and all transitive recorded sources.
Vectors stay outside the review bundle.

The browser opens the chatlog first. Ratings stay beside the captured activity.
**Thoughts & routing** shows private thoughts and each later attention cycle.
Expand codelet output to compare selected and unselected candidates.
Read the attention frame's focus contents and the routing decision's actual explanation.
The selected action names the public workflow that receives it.
**Knoxel contents** searches exact record text and record numbers.
Opening a knoxel shows its sources and the later records that reference it.
The **Record packages** view also opens each named collection directly.
Record headings expose useful metadata on hover. **Record metadata** retains every captured field except embeddings.
These references establish recorded dependencies, not the truth of a thought or an isolated causal effect.
Structured explanations and reflection plans retain their original wording under **Recorded explanation and raw output**.
Experiment cards show the final captured ledger state. Their retained history preserves the earlier lifecycle actions.
The scenario marks the introduced mistaken belief as supplied setup.
Engineering failures remain in their evidence directories. They do not enter this completed conversation's review item.
