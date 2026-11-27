# Gamma: work after the continuous-character milestone

Status: opened by the owner on 2026-11-14. The owner brought G02 forward for the beta trial on 2026-11-26.

Gamma is the milestone after [beta](beta.md). Beta is still in progress: B05 is in close-out, and B06 and B07 have not started.
This file records milestone decisions and verification.

| Item | Status | Subject |
| --- | --- | --- |
| G01 | Accepted as the first gamma work item | Reflective inquiry work deferred from B05 |
| G02 | Implemented and exercised on disposable beta homes | Memory reconsolidation and sleep |
| G03 | Owner requested a live experimental attempt during Gamma | [Unconscious valence, attraction, avoidance, and reflective inference](gamma/relationship-qualia.md#g03-unconscious-simulation-and-reflective-inference) |

## G01: reflective inquiry work deferred from B05

The owner moved this work out of B05 on 2026-11-14, so that B05 can close on behaviour a person can review.
The reasons and the audit are in [the B05 record](beta-b05.md#audit-and-close-out-decisions-2026-11-14).
The four parts concern how the companion examines and adjusts its own way of working.
They are not needed for the companion to notice and report its own recent thinking; B05 close-out covers that.

### 1. A graph step that commits a reflection decision

Today a `reflection_review` step in the inquiry graph reads the recorded experiments, the retained episodes, and their sources.
It cannot change the experiment ledger. The reflection child that produces a `ReflectionDecision` runs only in the retained linear path.

The blocker is a persistence boundary, observed in the source during B05:
every graph commit is an append-only recovery root, and `stage_decision` replaces an existing experiment record.
A recovery-root commit rejects replacements.

Owner decisions this part needs before any design:

- Where an experiment transition is accepted: a derived, non-recovery-root commit like the one `ensure_inner_turn` makes, or another boundary.
- What the operation record retains, so a repair can replay the decision without paying for the model calls again.
- Whether three extra model calls on each reflection step are acceptable.

### 2. Capability doubt in the graph

`triggers._capability_doubt` fires only in the linear path, because the graph evaluates its trigger policy without inspection evidence.
Giving the graph that evidence creates new private opportunities and new paid calls on private ticks.
That is a behaviour change, so it needs its own measured decision.

### 3. A step proposes the next recall query (decision 3B)

The owner accepted this direction during B05 as a separate pass.
An inquiry step would propose the query of the next `memory_search` step.
It changes the step contract of every inquiry, so every recording must be recorded again.

### 4. A sample that uses the extended reflection citation grant

A reflection step extends its citation grant with the reflection records.
No recorded sample has cited a record that only the extension permits, so the extension is present but unobserved.
This part is evidence work. It needs a scenario whose reflection material holds a record outside the origin grant.

### Starting G01

G01 needs an architecture proposal first, as `AGENTS.md` section 5 requires.
Inspect the current source at that time. The observations above describe `3c2640b` and can be stale.
No part of G01 has run, and no part is implemented.

## G02: memory reconsolidation and sleep

The owner accepted early implementation for the Friday beta trial. The design preserves the existing Story voice boundary and entity-claim ledger. A managed pass prepares complete episodes and their factual filings on a private view. Cognitive control publishes them at a main-cycle, wake, or clean shutdown boundary. The same pass serves offline commands. The live decision reads external work and measured Story omissions. New characters require no seed rewrite.

The acceptance run uses disposable character homes, real local chat and embedding endpoints, and an existing synthetic overnight checkpoint. Inspect original sources, linked claims and exclusions, independently durable units, cancellation during a model call, natural and input wake, clean restart, and actual prompt roles. Compare the foreground conversation with the pre-change revision. The manual describes the runtime under [character import and memory reconsolidation](../../docs/subsystems/character-import.md).


### Verification on 26 November 2026

The comparison revision was `c8b876f`. All character mutations used disposable copies or fresh homes. The results in the table used the single local Qwen slot at port 8080 and the embedding endpoint at port 8081. The evidence is under `.runtime/g02-memory/`.

| Question | Observed result | Evidence under that directory |
| --- | --- | --- |
| Can the character choose and end sleep? | After Jesse explicitly asked for sleep, the first decision was deferred by an entity update. Reconsideration selected sleep and dispatched the real managed worker. A new message woke the character in 47 ms. Natural completion and operator cancellation also released the wait. Spontaneous selection was not observed in this run. | `live-worker/main-choice-short1/receipt.json`, `live-worker/natural1/receipt.json` |
| What survives interruption? | Input arrived during a model call for the second episode. Reopening showed only the first complete episode. The next pass filed the second without duplicating the first. Background preparation kept two ready units invisible until a main-cycle commit. | `live-worker/transport1/receipt.json`, `live-worker/resumed2/receipt.json`, `live-worker/background1/receipt.json` |
| Can the offline paths resume? | An interrupted CLI run reduced eight eligible records to four. Resuming produced two episodes and five calendar periods. Repetition made zero model calls. An interrupted import retained its original source IDs, completed once, and rejected a duplicate. | `cli-recovery-check1/cli-verification.json`, `cli-recovery-check1/cli-idempotent.log`, `import-resume-20261126T084024Z-9342bdee/character/import/import-20261118T100000-6ded5bda.json` |
| Are speakers, corrections, and thoughts kept distinct? | Three speakers shared an episode. A correction removed the earlier linked preference from current recall. Two paraphrases of Mara's preference produced one current semantic head with two occurrences on different days. An independent bread-making fact on the same line survived. A private worry contributed to the episode without becoming a claim about Jesse. | `entity-release/before.json`, `entity-release/after.json`, `entity-release/paraphrase.json`, `entity-release/debugger-stacks.json` |
| Does a mature character benefit? | With a 24,000-token Story input, HEAD represented 105 originals. The updated planner represented 1,291 of 1,874 eligible originals, reported 583 omissions, and used 23,993 tokens. Local prompt fitting took 1.01 s. A normal public reply on a copy of the 14,799-record overnight character recalled the five-faced bush. | `coverage-probe-baseline.json`, `coverage-probe-final.json`, `qualified/confirmed-reply.json` |
| What does maintenance cost on that home? | Recalculating work took 0.47 s cold and 0.016 s with cached text costs. The pass filed one complete five-source episode and reopened it. Saving its full checkpoint took 8.44 s. The model-slot wait and full-checkpoint latency still matter on a large home. | `overnight-unit/verification.json`, `warm-cache.json` |
| Can a conversation continue after filing? | Three live questions spanned 3 h 10 min of virtual time. The first answer was only “Spamuel” despite the correct evidence in its prompt. Subsequent answers distinguished Spamuel's worry from Jesse's words and correctly named Mara's preference. This exercised revisiting the home across a simulated afternoon. | `entity-release/conversation.json` |

The focused run passed 86 deterministic checks across episode selection, factual filing, the entity ledger, Story, inference admission, and wake boundaries. The live import and managed sleep tests used the local services. The operator smoke test also wrote and reopened an idle snapshot and shut down cleanly. The strict manual build succeeded. Headless Chromium resolved 45 internal article links across the three edited pages. The debugger transcript records a real model-to-publication call stack. See `relevant-tests.err`, `live-worker/operator-real1/receipt.json`, `docs-verification.json`, and `entity-release/debugger-stacks.json`.

The first large-home trial loaded workflow code that changed during the test and failed at the Story observation schema. The successful repeat pinned the application source and used the original accepted checkpoint. It reused only derived vectors and image descriptions for matching original records. Model timings include local endpoint contention. A measured large-home reply took 507 seconds after input. Both revisions published an additional autonomous reply while draining work at shutdown. Both homes reopened successfully. See `continued-homes.json`.

One separate repeat of the synthetic two-line import used the checkout's OpenRouter configuration. The provider reported $0.002906404 for three chat and four embedding requests. The live import test now defaults to `config.example.yaml`. Both live tests passed with that default. The repeat's import request logs confirm local addresses for all seven requests. See `import-resume-20261126T091326Z-91550d4f/provider-use.json`, `import-resume-20261126T091506Z-80184008/prompts/`, and `local-live.err`.

The three earlier failures in the game-choice suite reproduce on HEAD (a missing capture, an obsolete implicit-checkpoint assumption, and an affect expectation). The dependency-rule suite also reports pre-existing violations. The individual dependency violations match HEAD. The strict type comparison includes 52 changed production and simulator modules and found no added diagnostics. The readability checker found no added violations on the changed paths. See `simulation-types-diff.json`, `readability-all.json`, `game-choices-baseline.log`, `architecture-diff.json`, `live-worker/wait-suite-diff.json`, and `live-worker/cli-restart-diff.json`.
