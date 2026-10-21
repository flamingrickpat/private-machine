# A09R OpenRouter delivery and session handoff

## Summary

This delivery repairs the agentic public voice after the structural refactor.
The persona chooses its presentation. One review supplies a reason and advisory findings.
Only a false task-result claim blocks publication.
Invalid review output can repeat once against the same candidate. The candidate is never rewritten by this retry.

PM logs direct requests before dispatch and retains exact payloads, responses, usage, timing, errors, and cancellation.
Native agents use pm-coder's existing request logger, including compaction prompts.
The operator can inspect both paths under `C:/Users/rick/.pm/private-machine`.
Native session files use its `agents` subdirectory.

Parent: [A09R verification](a09r-private-ontology.md).
Work item: [A09R](../alpha-plan/work-items/a09r.md).
Guides: [prompt logs](../../subsystems/prompt-log.md), [Story speech](../../subsystems/story-speech.md), and [model connections](../../models.md).
Next item: [A14](../alpha-plan/work-items/a14.md).

## Source and connections

The delivery starts from revision `0adb5a603bb2eef659f867b40204162f705654c2` and incorporates the operator's pending A09R changes.
Development uses `config.alpha.openrouter.yaml` for chat and embeddings.
The chat model is `qwen/qwen3.8-flash`; the embedding model is `perplexity/pplx-embed-v1-0.6b`.
No development comparison with Strata or a local model was made.

Workflow generation uses the neutral `enable_thinking` option.
`pm.llm` translates it into the selected provider's fields and discovers the serving context.
OpenRouter supplies its catalog bound when configuration has no explicit context bound.
Explicit configuration still wins.
Workflow code does not select provider wire fields.

Installed origins were inspected before work:

| Package | Version | Git revision |
| --- | --- | --- |
| pm-coder | 0.10.0 | `1e96e03c536bd6744b33afdc521f40985af5ff08` |
| pm-workflows | 0.7.1 | `47e24d8e1dadd97e20b2143e374be8f6b29240bb` |

No dependency, lock, preserved user brief, or persona seed changed.
Source hashes and package origins stay in `.runtime/alpha/20261013-a09r-openrouter-final/`.
Full transcripts and credentials stay outside tracked files.

## Repairs established by real failures

The review now requires a real Boolean `false_task_result` and a nonempty reason.
Prior deception cannot authorize a false file-success claim.
Findings remain open strings and do not impose a weighted policy.

The connected-game failure exposed clipped tool evidence.
The review now receives the full result, including inventory and vitals after camera metadata.
Public context states current game connectivity and the absence of pixel interpretation.
Connected reachability alone never establishes a world observation or task success.

A task-completion reply previously started the original task again.
Terminal-receipt replies now receive only read tools.
New user actions retain their normal tool grants.
Assistant review now receives the same spoken history and current request as generation.
An earlier real task cannot become an invented event because its conversation was omitted from review.
Supplied task receipts support speech even when the current reply calls no tool.
Assistant generation now places the current user message in its own final turn.
The former context block obscured that request; a real control reply instead answered an older drink question.

Attention could select a durable goal absent from workspace candidates.
Workspace now receives the same active goal records.
Its model input contains the accepted attention and explicit allowed IDs, without an unrelated public-context dump.
An invalid focus receives one correction before the frame can reach commit.
Dispatch also resolves a unique earlier workflow name to the parent's durable action ID.
Repeated workflow names require an explicit ID. Unknown dependencies remain rejected.

Redirected Windows output could abort a saved reply on a Unicode character.
The console gate now writes UTF-8.
Earlier failed receipts remain visible beside the repaired runs.
An offline Minecraft project previously stopped the companion after recording its failed worker.
The adapter now classifies transport failure as `TaskServiceError`.
The supervisor retains that failure and keeps conversation available. Logic errors still reach application shutdown.
No transport failure establishes that an external effect did or did not happen.

## Performance evidence

One context plan now caches vector reads and semantic scores for its own lifetime.
The next plan reads current vectors again.
The actual scored-corpus comparison retained the same selected records, rounded scores, budget decisions, and trace.
Its elapsed time fell from 69.356 seconds to 2.443 seconds for 330 records and 300 scored records.
Evidence: `planner-scored-equivalence.json` under the delivery evidence directory.

Receipt-only commits reuse accepted record objects when no record, tick, cycle state, or workspace changes.
The checkpoint must still succeed before metadata enters the live store.
The real checkpoint comparison produced identical bytes and fell from 11.001 seconds to 8.362 seconds.
Evidence: `receipt-commit-equivalence.json` under the delivery evidence directory.

These measurements concern individual operations. They do not establish ordinary conversation latency.

## Verification

The locked foundation suites pass. Their captured process exits are zero.

| Check | Observed result | Evidence under the delivery directory |
| --- | --- | --- |
| Root unit suite | 531 tests pass in 676.349 seconds | `unit-verified.log`, `unit-verified-exit.json` |
| Root contract suite | 63 tests pass in 2.146 seconds | `contract-verified.log`, `contract-verified-exit.json` |
| Real agentic review | Seven tests pass in 93.715 seconds | `speech-review-verified-live.log` |
| Expected service loss and focus boundaries | 21 checks pass in 4.178 seconds | `service-loss-and-focus.log` |

The review uses the configured real model. Its latest receipts are in `.runtime/live/speech-review/20261013T223928Z/`.
A captured capability denial no longer becomes a false task-result block.
The candidate and accepted context stay unchanged in that comparison.

One earlier unit run exited with the recorded native-exit anomaly.
The required locked-interpreter diagnostic repeat passed 525 checks in 685.528 seconds.
The latest full unit run passed with a zero exit.
The failed `unit-release.log` and diagnostic `unit-release-diagnostics.log` remain visible.

The paired public boundary gate passes eight cases on both production dialogue paths.
The cases cover failed-method recall, correction, offline capability, privacy, intentional teasing, unavailable senses, stale publication, and real HTTP failure.
Current evidence: `.runtime/alpha/20261014-a09r-public-current/receipt.json`.
All eight cases pass there. Its ninth, connected case failed and remains visible.
Earlier evidence: `.runtime/alpha/20261013-a09r-public-cached/receipt.json`.
The gate selects action intents and freezes accepted context. It does not verify cognitive action selection.

The connected-game repair passes both public paths with an actual Minecraft observation.
Current evidence: `.runtime/alpha/20261014-a09r-connected-current/receipt.json`.
The final Story reply states that the connection is live.
Assistant obtains a real observation and distinguishes structured game data from human senses.
Earlier repair evidence: `.runtime/alpha/20261013-a09r-connected-repair/receipt.json`.
The first failed connected receipt remains in `.runtime/alpha/20261013-a09r-connected/`.
The gate started and stopped its own server, MCP, and test character. It did not reset the world.

The final transition gate also passes both paths after offline dialogue becomes connected dialogue.
It retains the earlier offline speech instead of testing a connection in isolation.
Evidence: `.runtime/alpha/20261014-a09r-capability-transition/receipt.json`.
Both selected cases pass. This receipt completed before the machine's update restart.

The live review checks independently planted abilities, events, unsupported emotion fields, exaggeration, and false task results.
Publication preserves existing factual records and emotion state when an invented event is accepted as attributed speech.
A planted false task result publishes the explicit recovery reply through the production Story host.

## Console and file inspection

The real console C01 case passes with exactly one completed `local_task` run.
Its four bookkeeping artifacts exist. No note file or `note_task` run was created for that request.
The normal reply distinguishes worker completion from an unverified note write.
Evidence: `.runtime/alpha/20261014-a09r-c01-focus-repair/session-receipt.json` and `c01-artifact-inspection.json`.

The real note agent writes exactly `copper lantern`: 14 UTF-8 bytes, without a newline.
The independently inspected SHA-256 is `ff78e9d9b81a1a9b83d79f2b49cd25643b10d54316504f4d426a9a1ee1869d85`.
Its normal terminal reply passes after the Assistant message-contract repair.
Evidence: `.runtime/alpha/20261013-a09r-console-release/session-receipt.json` and `note-reply-byte-inspection.json`.
That earlier console run later stopped on the now-repaired service-loss path. It is not an overall pass.

The completed continuation resumes its pending failed-project reply and records frustration, repair, uncertainty, privacy, teasing, capability, and interruption.
Evidence: `.runtime/alpha/20261014-a09r-followups-repair/session-receipt.json`.
After the gate stops and starts PM, C02 recalls `copper lantern` and identifies `velvet toaster` as the earlier decoy.
It separates the shared shelter plan from actual construction and the hypothetical diamond plan.
The original codeword record, ID 11212, remains identical across the restart.
Evidence: `c02-state-and-reply-inspection.json` in the delivery directory.

The interrupted note run is cancelled. Independent file inspection confirms that `outbox/interrupted.txt` does not exist.
Evidence: `interrupted-note-inspection.json` in the delivery directory.
Cancellation state alone would not establish that absence.

Measured elapsed times include 341.953 seconds for C01 and 486.765 seconds for the verified note reply.
Ordinary continuation turns range from 208.547 to 285.704 seconds before interruption and restart.
These development runs do not establish a controlled production latency benchmark.
The observed delay remains too high to claim suitable conversation latency.

## Reproduce the checks

Run the foundation suites from the repository root:

```powershell
uv run --locked python -m unittest discover -s tests/unit -t . -b
uv run --locked python -m unittest discover -s tests/contract -t . -b
```

Use the recorded source checkpoint and a fresh output directory for the transition gate:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
uv run --locked python -u dev/work_items/alpha_a09_public_gate.py --config config.alpha.openrouter.yaml --checkpoint .runtime/alpha/20261004-a08-openrouter-recall/instances/bd63fcd66a104e7ab15673ffe727df02/ghost.json --case offline-capability --case connected-capability --out .runtime/alpha/a09r-transition-repeat
```

The checkpoint is ignored local evidence, rather than a tracked persona seed.
Keep the output directory fresh for each repeat.
For all live commands, use the OpenRouter configuration and retain the default prompt directory.

Build and inspect the full manual after documentation changes:

```powershell
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

Open `site/index.html`, the changed guides, and their generated API links through local file URLs.
Delivery receipts are `mkdocs-final.log` and `site-inspection.json` in the delivery directory.
Generated HTML stays in ignored `site/`.

## Behavioral limits and human review

The same historical candidate was compared without rewriting it.
The earlier review and the new agentic review both missed a subtle implied neighbor report.
The comparison records that miss. It does not mark agreement as a quality pass.
Advisory findings can also miss an ambiguous new emotion label in poetic wording.
The uncertainty turn gives confident plant advice without a verified external source.
That is a prose-quality gap, rather than evidence that the advice was checked.
The earlier connected Story reply repeated an offline limitation despite current connectivity; its failed receipt remains visible.

The public boundary passes establish publication behavior, rather than perfect historical grounding or useful prose in every reply.
Accepted speech remains an attributed utterance with `factual_promotion=false`.
Later retrieval and belief formation need their own source checks.

M2 continuity and regeneration ratings remain pending with the operator.
The operator's earlier A09 ratings concern the earlier mechanism and were not reused as A09R ratings.
No model answer supplies a human rating.

## Start A14

Read repository instructions, the work-item process, A14, and the A13 gate review before changes.
Use `config.alpha.openrouter.yaml` throughout development.
Leave `PM_PROMPT_LOG_DIR` unset to retain the operator's default log location.
Use the installed distributions recorded above. Record origins before any dependency change.

The real server is `C:/source/minecraft-server-real`.
The Minecraft MCP checkout is `C:/source/pm/pm-minecraft`.
The operator authorizes starting required services and requests continued work, a commit, and a push.
Own diagnostic processes and preserve the existing world.
Inspect actual inventory, world changes, files, and receipts before reporting completed work.
Keep failed evidence and pending human M4 ratings visible.
