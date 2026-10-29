# K002 refactor audit

This audit examines revision `e90514e` against its parent, `8480b73`.
It also resumes K002 acceptance against the local model services.
K002 remains in progress until its live gates finish.

Parent: [K002](index.md).
Acceptance: [execution evidence](verification.md).
Refactor scope: [typing and docstrings](../refactor/typing-and-docstrings.md).

## Evidence and environment

The evidence root is `.runtime/kanban/k002-audit-20261029/`.
The previous K002 run folders are absent from this checkout.
The two source persona homes remain under `~/.pm/private-machine/`.
Fresh runs fork these homes without changing them.

The chat service is `http://127.0.0.1:8080/v1`, model `qwen`.
The embedding service is `http://127.0.0.1:8081/v1`, model `qwen3-embedding-0.6b`.
The embedding probe passed against the real service with 1,024 dimensions.
Its paraphrase similarity was 0.773, unrelated mean was 0.284, and margin was 0.489.

The original virtual environment referenced an inaccessible Python junction.
An attempted launcher check then caused `uv` to remove packages from the active environment.
The affected runs remain failed evidence: `20261029-133555-cb93e486` and `20261029-133629-1eb20de9`.
These failures do not establish a product regression.

The replacement environment is `.runtime/toolchain/audit-venv/`.
It uses Python 3.12.13 and dependencies installed with `uv sync --locked`.
Its launcher uses the standard Python virtual-environment executable because the local `uv` trampoline could not start.
No dependency manifest or lockfile changed.
The ordinary `.venv` was also restored. `uv run --locked --no-sync python --version` reports Python 3.12.13.

## Reproduced regression

The refactor changed `Envelope.from_json` from constructor unpacking to selected field copying.
That change introduced two differences:

- Omitted optional `target` and `causation_id` fields raised `KeyError`.
- Unknown fields disappeared instead of raising `TypeError`.

The baseline probe reproduced both differences in `envelope-before.json`.
The repair restores constructor unpacking and retains the named `EnvelopeJson` annotation.
Missing required fields and unknown fields again raise constructor errors.
Three regression checks cover these cases without substitute collaborators.

## Static audit

The executable comparison removes docstrings and annotations before comparing existing function bodies.
The retained receipt is `executable-diff.txt`.
The review covers changed constructor calls, serialization, record creation, indexing, runtime assembly, and simulation execution.
It does not establish equivalence for every possible runtime input.

The configured strict Pyright check covers 11 files and reports zero errors.
That result does not establish repository-wide type correctness.
A separate strict audit covers these nine files:

- `pm/bus/envelopes.py`
- `pm/app/control.py`
- `pm/llm/prompt_encoding.py`
- `pm/records/memory_records.py`
- `simulation/plans.py`
- `simulation/client.py`
- `simulation/engine.py`
- `simulation/server.py`
- `workflows/companion.py`

The baseline reports 958 diagnostics. The refactor reports 515 diagnostics in the same selection.
Both receipts remain in the evidence root.
The refactor improves that selection, but its typing migration remains incomplete.
Most remaining diagnostics concern simulation controller mappings and runtime control contracts.
An accumulating document can still have named field contracts. Growth alone does not require an open mapping.

New Pydantic class docstrings also enter generated JSON schemas.
The simulation sends those schemas to the model.
Thus, documentation edits can change model-facing instructions even when explicit prompt templates remain unchanged.
Live acceptance must cover this boundary.

## Focused verification

Commands use `.runtime/toolchain/audit-venv/Scripts/python.exe`.

| Receipt | Result |
| --- | --- |
| `locked-env-focused.log` | 37 causal recovery, real HTTP/WebSocket, coordinated checkpoint, review, and architecture checks passed. |
| `context-rendering.log` | 18 deterministic context-budget, record-rendering, and turn checks passed. |
| `envelope-tests.log` | 11 serialization and real bus-isolation checks passed, including the three regression checks. |
| `pyright.json` | The configured 11-file strict check passed. |
| `pyright-envelope.json` | The repaired envelope module passed strict checking. |
| `embedding-quality.json` | The real embedding geometry probe passed. |
| `planning-contract-tests.log` | 13 real-store planning and shard-rendering checks passed. Direct planner checks explicitly disable the encoder. |

The selections have distinct purposes. None establishes companion quality.
No full unit or contract suite was requested or run.

## Live acceptance

The clean cat run is `20261029-134055-5e5c177f`.
Its model-generated plan retains arrival, midnight sleep, and the mandatory 07:00 morning facts.
The clean peer run is `20261029-134101-b1c3ac14`.
It starts separate PM processes with reciprocal public connections.

Attempt 1 of both runs failed before public speech.
The complete Story prompts exceeded the conservative 78,400-token input ceiling.
The cat estimate was 79,860 tokens. The peer estimate was 80,261 tokens.
The runtime retained both tracebacks and the accepted cognition states.

The memory planner spent the input allowance before final Story instructions and framing entered the prompt.
The repair extends the shared selector with a workflow-owned renderer.
Story reserves instructions, current speech, prior dialogue, shard evidence, supplied sources, and repair feedback first.
Optional retrieved evidence then competes by its existing score against the complete rendered prompt.
The review applies selection to its own complete template and generated draft.
Task receipts remain required. Each attempt retains separate generation and review selection receipts.
Required overflow still raises before inference.

The captured cat replay contains 306 dialogue turns and 1,198 evidence records.
Its replay estimate decreased from 79,997 to 78,341 tokens after six optional memory groups were omitted.
The full request and selection receipt remain in `story-request.json` and `story-selection-replay.json`.
This is a rendering replay, not a passing live conversation.

Thirty-one focused budget, Story rendering, and speech-cleaning checks passed in `story-rendering-tests.log`.
The final selector took 30.21 seconds for this captured request.
The repair also avoids repeated transcript normalization inside each artifact loop.
The initial slower replay was stopped during local selection and retained as diagnostic evidence.

Attempt 2 reached actual speech generation, then exposed the separate review-layout overflow.
The captured review estimate decreased from 83,141 to 78,384 tokens through optional selection.
The model returned a valid review. `review-selection-replay.json` retains the selection.

The full Story kernel also completed against fresh inference with `PM_COMPLETION_CACHE=0`.
`kernel-uncached-receipt.json` confirms two chat calls with `cache_hit=false`.
The server counted 53,138 generation-input tokens and 59,612 review-input tokens.
The conservative estimates were 78,341 and 78,398 tokens.
The physical kernel folder retains `story-request.json` and `story-attempts/story-1-01.json`.
The initial kernel replay used ordinary runtime caching and remains separate evidence.

Both runs resumed as attempt 3 with generation and review selection.
The cat replay published reply 11303 once and advanced to the midnight sleep stage.
The peer run restored its coordinated initial barrier before restarting both processes.
Both runs then received pause requests to load durable selection capture at a coordinated boundary.
At that checkpoint, morning delivery, peer routing, recovery, and completed-report inspection remained required.

Both pause requests completed with coordinated checkpoints.
Cat progress retained two completed stages. Peer progress retained its first completed stage.
Attempt 4 resumes those checkpoints with `PM_COMPLETION_CACHE=0`.
The separate forced-interruption gate uses another fork and never terminates an existing participant or model service.

Peer attempt 4 completed both mandatory stages with uncached inference.
The transit ledger records accepted delivery from primary reply 11303 to other, then from other reply 11361 to primary.
The primary participant answered the other participant's message.
Its later reply 11358 remains explicitly pending at the final coordinated barrier.
This does not establish that every later message was delivered.
The report is `.runtime/simulation/runs/20261029-134101-b1c3ac14/attempts/4/report.json`.

The browser inspection exposed Windows separators in saved relative report paths.
The browser split those paths on `/`, so it constructed invalid evidence links.
New captures now emit `/` separators. The browser normalizes existing Windows captures when opening them.
The repaired browser loaded the completed report and 156 private-activity records without JavaScript errors.
`peer-browser-check.json` and screenshots retain that inspection.

The separate live recovery gate passed in `live-recovery/result.json`.
It forcibly interrupted its own PM process during an actual new cognition prompt.
Normal restart retained tick 808, approved action 11257, and reply 11303.
Newer input 11304 stayed outside the clean projection and remained in raw history.
The gate used a fork of the cat checkpoint; it did not interrupt the running scenario.

The cat run has completed arrival and midnight sleep. Its 07:00 stage remains pending.
Normal post-reply wakes continue every ten virtual minutes during Jesse's sleep.
Each complete cognition cycle takes several real minutes, so the overnight gate remains expensive.
No cognition phase was bypassed and no experimental state control was added.

The browser history, checkpoint selector, required facts, and review controls loaded without JavaScript errors.
Screenshots and `browser-check.json` retain that inspection.
The rating field remained empty.

No human quality rating was entered.
The Story guide also described an obsolete assistant prefill and three-message layout.
It now describes the implemented interleaved layout and final prompt selection.
The repairs preserve subsystem ownership and the existing publication boundary.
No conceptual diagram changed.

## Configured encoder failure boundary

The audit found two existing catches that hid configured-service failures.
`_embed_focus` replaced `ModelError` with lexical scoring.
`PublicDialogue._reply_context` caught every planning exception and returned only shard context.
Both catches contradicted the required failure policy.
Configured encoder and internal planning errors now propagate. Transport retries remain in the HTTP client.
An explicitly absent encoder still permits direct lexical planning.
The planner now names its encoder result and correctly annotates its actual `Snapshot` input.

The before/after probe calls the real encoder through an invalid route, which returns HTTP 404.
Before the repair, both boundaries returned partial context. After it, both raise the original `ModelError`.
The receipts are `encoder-boundaries-before.json` and `encoder-boundaries-after.json`.
`healthy-context-planning.json` confirms retrieval against the healthy encoder with a real memory store.
The earlier corrupt-vector probe in `planning-failures-before.json` was inconclusive and is not acceptance evidence.

These failure-boundary and path-format repairs followed the frozen source of attempt 4.
The completed conversation establishes its recorded source behavior.
The direct endpoint probes and current browser check exercise the later repairs.
The mandatory morning gate and human quality review remain open; K002 stays in progress.

## Evening continuation

At 21:12 UTC, neither the browser nor its participant process was running.
The saved controller still said `running`, at 01:30 local virtual time with 12 timeline entries.
The last process log ended during analogous-episode recall without a captured product exception.
This is an interrupted attempt, not a completed overnight test.
The cause of process termination is unknown.
`interruption-2112/` preserves the controller and process log before restart.
The model services remained available. Attempt 5 resumes the saved participant and clock with completion caching disabled.

Current capture code also recaptured the completed peer participant.
`peer-acceptance.json` confirms two cycle packages and 504 prompt files with existing forward-slash artifact paths.
The browser followed the request-to-public-reply link without an error.
The final local HTML scan checked 296,746 links with zero missing file targets.
Seven sampled manual pages loaded without JavaScript errors; changed pages and API references were visually inspected.
No diagram changed in this repair.

The Story prompt loader and review loader now validate named YAML field contracts.
Generation parameters retain the current temperature, top-p, and thinking settings.
Unknown Story or review configuration fields raise instead of disappearing during validation.
Rendering separates spoken lines and private thoughts with named fields.
The kernel tool adapter restores the concrete asynchronous completion type erased by its foreign registry.
No cancellation argument is overwritten.
Twenty-five current rendering and shard checks passed in `typed-rendering-final.log`.
The configured strict check remains limited to its 11 files.
A separate four-file Story check still reports 77 diagnostics in legacy rendering and parsing functions.
It does not establish complete strict typing for Story.
