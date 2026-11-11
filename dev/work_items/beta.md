# Beta: continuous character cognition

Status: owner-authorized planning and incremental implementation, 2026-11-09.

## Scope and version names

The owner named the continuous-character milestone and directly related work
**beta**. The specification remains `docs/continuous-character-milestone.md`.
Beta is a development milestone, not a claim that its acceptance criteria pass.
The later milestone can use **gamma** once its scope exists.

The preserved baseline is commit `2b17cf8`, tag `alpha-baseline-20261109`.
Implementation uses branch `beta`. Package version `1.0.2` remains unchanged:
the milestone name does not change Python dependency or checkpoint versions.
Commits remain local under the existing no-push instruction.

The owner authorized planning, implementation, GPT-6.1 implementation agents,
and cleanup of old work. All live inference uses OpenRouter. No local model
service starts. This authority covers choices within the accepted specification.
Changes to its architectural commitments still require owner review.

## Observed baseline

The interrupted month run is `20261104-eventful-month`. On November 9 its saved
state still said running, but the dashboard and participant were absent.
It reached four of 90 scenes, with 16 Jesse messages and 115 Spamuel messages.
Its virtual clock reached June 1 at 10:20:30. The last runtime log entry was
November 6 at 00:22:41 UTC. The shutdown cause remains unknown.
No complete-month report or final analysis exists. Preserve this partial baseline.

The observed problem includes repeated unsolicited contact through a reported
absence and expensive full cognition on repeated wake events. More month runs
are not a prerequisite for the first architecture changes. The existing harness
provides short comparison runs and later sustained tests.

## Current module and execution map

`workflows.companion.build_app` assembles the bus, canonical memory store,
`MainCogLoop`, public dialogue, private thought, and cognitive control.
`UserInputProxy` commits incoming dialogue before it publishes `user.input`.
`MainCogLoop` runs the full main-cognition manifest before it dispatches an action.
Its phases are intake, triage, appraisal, retrieval, codelets, structures,
attention, experience, goals, decision, routing, and report construction.
`MainCogLoop.publish_output` accepts the write batch and publishes dispatches.
`StoryDialogue` generates and reviews speech. `commit_public_reply` commits the
utterance and exchange resolution before bus publication.

Separate bus-module threads already exist. They do not provide an independent
fast interaction path. Main cognition holds foreground inference reservations
through its cycle. `InferenceAdmission` coordinates calls to the shared endpoint.
An async design must change this scheduling contract, not only add a thread.

The memory store owns accepted records and revisions. `CycleState` owns accepted
workspace state and claimed stimuli. Main cognition owns deliberate intentions.
`CognitiveControl` owns waits and external-input generations. Public speech owns
durable utterances and delivery lifecycle. Workflow code imports `pm`; `pm`
must not import workflow behavior. No runtime code imports the simulation.

## Capability map

| Capability | Current evidence | Beta work |
| --- | --- | --- |
| Real long-run harness and automatic ratings | Implemented, short real runs verified; month interrupted | Reuse, calibrate with owner review |
| Accepted intentions and expectations | Existing records and main-cycle commits | Preserve lifecycle and historical attribution |
| Ordinary fast interaction during cognition | Absent | Add after publication and dependency boundaries |
| Critical early triage barrier | Absent; triage occurs inside full cycle | B01 |
| Shared current situation guidance | Partial context, decision prompt omits some grounding | B02 |
| Persistent entity situation and knowledge model | Identity anchors and narrative only | B04 |
| Private thought and deliberate mental work | Existing workflows; full target unverified | B05 |
| Generative/discriminatory inquiry and deliberate ToM | Partial foundations, target not established | B05–B06 |
| Sustained beta behavior | Not verified | B07 |

## First implementation proposal

### B01: early triage and critical publication barrier

Requirements: ARCH-02, ARCH-12, ARCH-14, SC-17. This slice establishes a
prerequisite for async dialogue. It does not itself complete the fast path.

Add `workflows/input_triage/` with a typed assessment, a kernel workflow, and
a host binding. Reuse the existing triage categories and contextual prompt.
An assessment retains source IDs, source fingerprints, its guidance tick, and
the validated `TriageResult`. Main cognition consumes that result rather than
paying for a second classification of the same evidence.

Start with the existing `Critical` category as the blocking threshold.
Classification concerns contextual significance, never a phrase lookup.
The first early classifier covers durable `user.input`, including attributed
peers. Other event kinds retain full-cycle triage. This is a scope limit to
extend before claiming generic critical-stimulus acceptance.
Unknown incoming triage temporarily prevents publication until classification.
Ordinary input then releases that temporary hold. Critical input requires an
accepted main-cycle result that includes its source IDs.

`CognitiveControl` coordinates pending triage and required reconsideration.
Registration and the public commit check share the memory-store lock, closing
the race between input acceptance and late reply publication. Each draft records
a barrier epoch and guidance tick. A later critical assessment invalidates the
old draft even if reconsideration finishes before generation returns.
Superseded drafts leave explicit evidence and unresolved exchanges. They do not
become fake replies or application failures. Restart reconstructs unresolved
barriers from retained assessments and accepted cycle claims.

The implementation must inspect generation, review, recovery, queued actions,
and restart consumers. Cancellation is an optimization; commit validation is
the authority. A model response after cancellation cannot bypass that authority.

Critical classification also interrupts an active ordinary main cycle that does
not contain those sources. The host requeues its unaccepted inputs with their
original envelope identities. This preserves pending-cause accounting without a
second external delivery. The next full cycle reads fresh state and the critical input.
An atomic acceptance check prevents the interrupted cycle from dispatching work.
This is critical preemption within the existing serial full-cycle topology.
It does not add ordinary fast interaction.

### B02: shared situational guidance

Requirements: ARCH-06–07 foundations, ARCH-10–11, SC-01, SC-18–19.
Add `workflows/public_speech/situational_guidance.py` as a read-only projection.
Use named typed sources, entity context, and recent moves. Retain timestamps,
record IDs, revisions, attribution, selected purpose, and actual delivery state.

`build_situational_guidance(view: Snapshot, ...) -> SituationalGuidance` reads
existing authoritative records. `render_situational_guidance(...) -> str`
provides the same semantics to main cognition and public dialogue.
`CycleRequest` and `PublicContext` carry the typed projection. `StoryPrefix`
includes it because the decision phase does not use the generic grounding turn.
The public story renderer must include it explicitly too.

The model reasons about ability to receive and ability to respond separately.
Unknown information stays unknown. A timer is not evidence that a partner woke
or reconnected. Recent delivered moves and their purpose inform repetition.
The guidance applies to any entity and action. It adds no eligibility agent,
sleep keyword, mandatory silence policy, or second persistent memory store.
Group-specific contexts retain entity isolation. Self-directed wake decisions
still need relevant evidence about the partner they consider contacting.

This slice does not claim a complete persistent entity model. B04 adds accepted
claims, corrections, freshness, and observer-relative knowledge explicitly.

### Ownership and migration

Existing checkpoints and source personas remain intact. New boundary fields use
explicit serialization contracts. Any default for an older checkpoint must
represent an actual absence of historical data, not invent a successful result.
No source-module relocation or dependency update is planned for these slices.
Domain errors fail loudly. Transient provider failures use existing visible retry
and cancellation behavior. No external action is repeated merely to repair logs.

### Required recovery boundary repair

The default clean checkpoint view currently excludes unclaimed input and its
early classification. A real-store restart reproduced loss of the critical barrier.
Add `Transaction.recovery_root: bool = False` and explicit
`WriteBatch.preserve_on_recovery()` for durable inbox and control commits.
This marker does not create a main-cognition acceptance or a GWT turn.
Its dependencies must already be accepted or valid earlier recovery roots.
An attempted root cannot admit an unrelated speculative branch through closure.

Expose the existing clean projection as `MemoryStore.recovery_snapshot()` and
let early triage select it through `MemoryBashTarget(recovery_only=True)`.
Early triage retains a typed exact evidence manifest with source revisions,
fingerprints, rendered context dependencies, and governing guidance commit.
Its control commit uses that manifest instead of the ambient full-snapshot read
capture. All actual supplied evidence still remains causally recorded.
Older transactions default to no recovery-root status. Do not retroactively
promote historical branches. Verify default clean loading, not only raw loading.

## Subsequent ordered slices

| Item | Depends on | Reviewable outcome |
| --- | --- | --- |
| B03 ordinary async interaction | B01, B02 | Real ordinary turns use accepted guidance during mental calculation; critical input defers them |
| B04 persistent entity context | B02 | Attributed observations, corrections, uncertainty, freshness, and compact shard views |
| B05 thought emergence and inquiry | B03, B04 | Selective candidates and resumable generative/discriminatory inquiry with source lineage |
| B06 ToM and metacognition | B04, B05 | Action-selected perspective inquiry and reconsideration of private thought |
| B07 interval appraisal and sustained acceptance | Integrated prior slices | Historical intention attribution, nonduplicate outcomes, short comparisons, then long runs |

B03 must replace raw lifecycle revision dependencies where only exchange
resolution changed. Actual source corrections must still invalidate results.
It must separate ordinary full cognition from foreground reservations and keep
critical reconsideration ahead of unrelated inference. Relevant dependency
validation must prevent both stale overwrite and perpetual cancellation.
The detailed B03 contracts follow inspection of B01/B02, before B03 edits.

## Verification plan

1. Run focused deterministic checks against real memory stores and boundaries.
2. Run strict configured type checks and inspect changed contracts.
3. Exercise early triage and decision context through real kernels on OpenRouter.
4. Inject real incoming events during a real draft. Inspect committed utterances,
   supersession receipts, accepted source IDs, and resumed dispatches.
5. Exercise normal input, unavailable and returning partners, critical input,
   later critical input during reconsideration, and restart where affected.
6. Run a short companion comparison with the preserved Spamuel history.
7. Build the strict manual and inspect changed HTML and source links.

Evidence belongs under `.runtime/beta-20261109/`. Record actual model identities,
commands, revisions, outputs, and negative results without credentials.
Model judgments remain separate from owner quality acceptance. Missing services
mean NOT RUN. Scripted LLM helpers in historical tests do not certify integration.

## Risks and limits

Early triage adds latency and must not deadlock behind a full-cycle reservation.
Late speech needs an atomic fence, including recovery and replay paths.
New context can increase tokens and must keep entity boundaries intact.
Long-term availability needs more than the most recent few messages.
Baseline shutdown remains unexplained. Initial work does not claim beta complete.

## Cleanup policy

Remove only identified obsolete automatic ghost copies from completed runs.
Keep current checkpoints, explicit recovery snapshots, frozen source, unique
transcripts, research, failure receipts, and the interrupted month baseline.
Record exact removed paths, hashes, and recovered bytes in ignored evidence.

The first cleanup identified 15 obsolete copies totaling 2,955,287,211 bytes.
Automatic approval review blocked both deletion attempts with "blocked by policy"
and supplied no further reason. No files were removed. The manifest and rejection
receipt remain under `.runtime/beta-20261109/cleanup/`.


## B02 verification, 2026-11-10

The shared situation projection is implemented. It reads accepted records and
current evidence without model calls or another persistent state store.
Main cognition, early triage, Story, and Assistant context use the same contract.
Entity choices receive that partner's context. Self and wake choices use the
configured partner. Reality and frame scopes remain explicit.

Seven deterministic checks used actual `MemoryStore` commits. They covered
partner isolation, runtime time, new contact, actual public dialogue, reality
boundaries, selected purposes, committed and delivered text, and stale-source
rejection. Private, virtual, and imagined dialogue did not enter actual interaction.
Main acceptance now uses `WriteBatch.expect_head` for guidance sources.
Public context retains source revisions, fingerprints, and current-head guards.

Command: `.venv/Scripts/python.exe -m unittest tests.unit.workflows.public_speech.test_situational_guidance -v`.
Result on November 10: seven tests passed. Receipt:
`.runtime/beta-20261109/b02-real-store-checks-20261110.txt`.
Scoped strict Pyright also reported zero errors for the new module and both test files.

The real decision gate used `config.yaml`, the canonical Spamuel history, and
synthetic scenario inputs. It ran production decision, routing, and report
phases through the real workflow kernel. Chat and embeddings used OpenRouter.
The chat model was `deepseek/deepseek-v4.1-flash-1017`.

Command: `.venv/Scripts/python.exe -m unittest tests.live.test_situational_decision -v`.
Evidence: `.runtime/beta-situation/20261109T222026Z-62282d0c/`.
The run recorded ten real chat requests and zero cache hits.
All four kernels completed their terminal commit phase:

| Scenario | Selected route |
| --- | --- |
| Human partner announced absence; a timer fired | `Sleep` |
| Human partner returned and asked a question | `Reply` |
| AI partner announced disconnection; a timer fired | `Idle` |
| AI partner reconnected and asked a question | `Reply` |

The process handle expired across the usage interruption. Its final unittest
exit status is unavailable. Original outputs, transport counters, and terminal
kernel journals were reconciled on November 10. All original gate assertions
passed that inspection. No inference was replayed or resampled.
The exact reconciliation is `reconciled-result.json` in that evidence folder.
Future executions also write `result.json` after their final assertions.

The first gate attempt remains at
`.runtime/beta-situation/20261109T221927Z-904990c8/`.
It failed before inference because test setup omitted the required regulation
value. Setup now derives regulation and feeling through the production functions.
That failure was a gate defect, not evidence about model behavior.

Independent review found two defects: cross-reality source leakage and missing
current-head guards at main acceptance. Both boundaries were repaired.
Their focused checks passed. The original review receipt remains at
`.runtime/beta-20261109/review/b02-frozen-guidance.json`.

These results cover immediate source projection and four decision scenarios.
They do not establish full-cycle integration, asynchronous ordinary dialogue,
repeated wake behavior, persistent availability, or long-term character quality.
The recent dialogue window contains eight lines. Older absence evidence can
leave that window unless accepted context retains it. B04 owns durable entity
state, uncertainty, freshness, and corrections. Human quality review remains open.

## Runtime recovery verification

The November 10 repair adds explicit append-only recovery roots.
`WriteBatch.preserve_on_recovery()` retains durable input and control receipts without a main GWT turn.
Commit rejects a root whose dependencies include an unaccepted transaction.
Commit also rejects replacement, lifecycle, workspace, run, and consolidation changes in a root.
Unmarked historical branches retain their existing recovery behavior.
An absent marker in an older transaction means `False`.

`MemoryStore.recovery_snapshot()` uses the same record projection as default clean reload.
It restores accepted heads and lifecycle overlays without changing live membership or counters.
`MemoryBashTarget(recovery_only=True)` exposes that projection to early triage.
Its read capture excludes speculative transactions.
The projection has no live vector index because that index can contain speculative heads.

The configured strict Pyright command reported zero errors, warnings, or information messages:

```powershell
node C:/Users/rick/AppData/Local/npm-cache/_npx/110e52990071af13/node_modules/pyright/index.js
```

That configuration covers the selected new modules.
It does not claim strict coverage of the legacy store modules.
The only new `Any` annotation describes the external Pydantic pickle-state contract in `Transaction.__setstate__`.
It does not replace a domain type.

Thirty focused deterministic checks passed against actual stores and checkpoint files:

```powershell
uv run --locked python -m unittest tests.unit.pm.memory.test_recovery_roots tests.unit.pm.memory.test_causal_recovery tests.unit.pm.memory.test_bash_view_scope -b
```

The checks cover explicit-root chains, default clean restart, rejected speculative dependencies, and unchanged checkpoint bytes after rejection.
They also cover accepted heads, excluded lifecycle overlays, recovery-only Bash views, old marker defaults, and retained raw archives.
The existing causal-recovery and scoped-view checks remain green.
A pre-marker pickle also loaded in clean and raw modes, with both historical root markers equal to `False`.
`git diff --check` found no whitespace errors in the affected source and test files.

Evidence: `.runtime/beta-20261109/review/runtime-recovery-verification.json`.
The original restart failure remains in `review/b01-clean-restart/receipt.json` under that evidence root.
This verification covers persistence contracts, not real model classification or concurrent companion behavior.
The B01 workflow gates and final strict documentation build remain separate checks.

## Additional boundary checks

The following focused command passed 26 checks against stores, publication,
speaker attribution, situation context, and import boundaries:

```powershell
.venv/Scripts/python.exe -m unittest tests.unit.workflows.public_speech.test_situational_guidance tests.unit.workflows.public_speech.test_publication tests.unit.workflows.public_speech.test_user_input_speaker tests.unit.test_architecture -q
```

Receipt: `.runtime/beta-20261109/publication-boundaries.log`.
An earlier broader command also selected `test_wait`.
That historical module hardcodes `config.example.yaml` and uses scripted cognition helpers.
It attempted an unavailable configured embedding endpoint and was cancelled during its first retry.
No local model started. That command provides no acceptance evidence.
Its receipt remains in `.runtime/beta-20261109/integration-boundaries.log`.

The combined focused command passed 64 deterministic checks on November 10.
It covers recovery roots, causal recovery, scoped Bash views, early-triage
barriers, situation context, publication, speaker attribution, and import boundaries.
Receipt: `.runtime/beta-20261109/final-boundaries.log`.
Configured strict Pyright also reported zero errors.

The final resumed strict documentation build passed in 170.97 seconds on November 11.
Six generated pages contained 3,502 local navigation targets with no missing files.
Receipt: `.runtime/beta-20261109/docs-final-resumed.log`.
These checks establish documentation construction and navigation, not companion quality.

After the final public-cancellation refinement, the same combined command
passed 67 checks in 5.136 seconds. Configured strict Pyright reported zero errors.
Receipt: `.runtime/beta-20261109/delivery-boundaries.log`.
The prior 64-check receipt remains as intermediate evidence.

## B01 live verification, November 11

The final early-triage gate passed with the primary Spamuel seed and two real,
uncached OpenRouter calls. Ordinary input remained noncritical. A consequential
identity correction received Critical. Default clean restart retained its
assessment and required reconsideration.

Command: `.venv/Scripts/python.exe -m unittest tests.live.test_input_triage -v`.
Receipt: `.runtime/beta-input-triage/20261111T061632Z-795d3768/result.json`.
The test completed in 37.098 seconds.

The assembled-runtime gate then passed against real OpenRouter chat,
embeddings, eight codelet-agent sessions, full cognition, and Story generation.
It uses Spamuel's actual character card and identity anchors with synthetic
scenario input. It does not test long-history character quality.

Command: `.venv/Scripts/python.exe -u -m unittest tests.live.test_critical_preemption -v`.
The final test completed in 416.498 seconds and exited successfully.
Receipt: `.runtime/beta-critical-preemption/20261111T062759Z-7bc16f43/result.json`.
Log: `.runtime/beta-20261109/critical-preemption-live-final.log`.

Critical input interrupted an older full calculation. Its original causes
joined the newer input without duplicate appraisal. A second Critical input
arrived during real Story calculation. The old invocation received a
cancellation-specific superseded receipt. Its queued review recorded
`CancelledError`; no old completion attempt started after the persisted
assessment's creation. The final utterance, record 192, used accepted tick 2.
Normal drain, worker join, and the application error check completed successfully.

The supplementary assertions are retained in the same run's
`cancellation-proof.json`. This proves cooperative client cancellation and the
publication fence. It does not prove that remote backend calculation stopped.
The run records 80 stage-client requests; that count excludes calls within
the eight separate agent sessions. Their actual receipts remain with the run.

Earlier evidence remains intact. The November 10 attempt was interrupted after
real preemption and has `interrupted.json`, not a successful result. The first
November 11 attempt failed because the gate omitted required `WorkflowModels`.
That setup defect has `failure.json`. A later complete run passed before the
public-cancellation refinement and exposed an unnecessary old review call.
The final run above verifies the repaired behavior.

Final deterministic acceptance contains 67 real-store and architecture checks,
including cancellation, committed speech, owner cancellation, accepted-source
successors, and default clean restart. See `delivery-boundaries.log` above.
Configured strict Pyright reports zero errors for its selected modules.

Remaining scope is explicit: early triage covers `user.input`, including
attributed peers. Other stimuli retain full-cycle triage. Ordinary asynchronous
interaction remains B03. Already-started external effects cannot be rolled
back; generic external-task preemption is not claimed. Retroactive assessment
of an already-claimed historical input lacks a separate acceptance watermark
and remains outside this boundary. Human quality review and sustained beta
behavior remain unverified.

## Delivery and inspection

The final strict manual build passed in 171.30 seconds after the public-cancellation refinement.
Receipt: `.runtime/beta-20261109/docs-delivery.log`.
The completed engineering trace contains actual delivered chat and both accepted cycles.
It is registered at `http://127.0.0.1:8095/#capture/beta-critical-20261111`.
Browser inspection found the chat and cycles with no JavaScript errors.
Receipts: `.runtime/beta-20261109/review-browser.json` and `review-browser.png`.

The hidden task `PM-Beta-Review-20261111` serves the review UI on `0.0.0.0:8095`.
It uses OpenRouter-only configuration and starts no simulation automatically.
Its launcher and log are under `.runtime/beta-20261109/`.
Port 8080 belongs to an existing Basalt application and was left untouched.

B01 and B02 have the bounded engineering evidence recorded above.
The next implementation slice is B03: ordinary interaction during asynchronous mental calculation.
The beta milestone remains incomplete. The partial month baseline remains preserved.
