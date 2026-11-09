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
