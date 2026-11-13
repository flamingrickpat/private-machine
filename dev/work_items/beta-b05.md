# B05: spontaneous nomination and bounded inquiry

Status: implementation in progress, 2026-11-13. Named requests, nomination, attention, connected graph execution, and parked re-entry are implemented. The bounded connected gate passed before the latest re-entry changes; coordinated control, deliberate work, and live re-entry acceptance remain outstanding.
Source baseline: `d72fd3b`, after B03 implementation `a9e78e8`.
B04 committed as `e7a7ce8`. This continuation starts from that commit.

Root review accepts the existing private host, one active episode, three nominations,
and three alternatives within the existing episode limits.
Nomination, attention selection, executed thought, and accepted finding remain distinct.
Ordinary unrelated input does not cancel independent private work; relevant source changes and Critical input retain authority.
Use the same inquiry path for spontaneous and selected mental work.
Do not add a second needs system or a new understanding axis in this slice.
Actual inquiry sufficiency remains explicit; interest cannot establish truth or external action authority.

This proposal implements milestone sections 8-11 and ARCH-03-05, ARCH-09-10, and ARCH-12.
The [milestone](../../docs/continuous-character-milestone.md) remains authoritative.
The [beta plan](beta.md) defines the shared $25 interim OpenRouter limit.
This source audit used no inference. It changed no product code or checkpoint.

## Current execution and concrete gaps

`companion.build_app` already installs one `PrivateThoughtLoop` and a deliberate `FocusManagement` host.
B05 retains those hosts, their kernel, memory store, inference admission, and cancellation ownership.

```mermaid
flowchart LR
    Timer --> Private[PrivateThoughtLoop.prepare_input]
    Private --> Trigger[TriggerChoice]
    Trigger --> Recall[Optional actual memory retrieval]
    Recall --> Thought[ThoughtResult]
    Thought --> Integrate[IntegrationResult]
    Integrate --> Commit[InnerMonologue and inner GWT turn]
    Commit --> Main[thought.created and main appraisal]
    Main --> Ack[Next private step requires acknowledgment]
    Action[Selected mental action] --> Focus[FocusManagement pm-coder session]
    Focus --> Plan[MentalPlan proposal]
    Plan --> Main
```

The source provides these foundations:

| Owner | Current contract and observed source behavior |
| --- | --- |
| `inner_thought/host.py` | `_candidates` chooses up to ten committed records; `_commit_cycle` owns private records and the inner GWT turn. |
| `inner_thought/pipeline.py` | `run_cycle` runs trigger, thought, and integrate; it validates citable IDs and reality labels. |
| `inner_thought/operations.py` | `Operation.mode` already distinguishes generation and discrimination. Recall, imagination, evaluation, and noticing operations exist. |
| `inner_thought/triggers.py` | Deterministic opportunities include unresolved intentions, expectations, surprise, idle opportunity, and rumination. No trigger produces no inference. |
| `inner_thought/schemas.py` | `ThoughtCandidate`, `ThoughtResult`, `ThoughtAlternative`, `InnerRequest`, and `InnerReport` already exist. |
| `cognitive_control/schemas.py` | `ControlConfig` already bounds episode steps, elapsed duration, stalled steps, and parking. Defaults are six steps, 600 seconds, two stalls, and 900 seconds. |
| `pm/records/lineage.py` | `RecordLineage` already stores episode identity, operation, sequence, multiple parents, sources, intentions, topics, and reality. |
| `main_cognition/phases/attention.py` | Main attention already uses accepted mental aperture, deliberate focus, structure ratings, and bounded workspace selection. |
| `cognitive_control/attention.py` | `active_focus` reads accepted advice; `resolve_focus` accepts proposals without changing their source grant. |
| `focus_management/host.py` | A selected mental action currently runs its own agent, stores a `MentalPlan`, and returns that proposal to main appraisal. |

These are source findings. This audit did not reproduce their runtime behavior.

The following gaps prevent the required target:

1. `ThoughtCandidate` names existing evidence records. It does not represent independently generated candidate thoughts.
2. Trigger selection immediately becomes deliberate elaboration. No separate nomination result competes for attention.
3. `_commit_cycle` links each thought mainly to its previous episode thought. Alternatives remain text without independent lineage.
4. `_resolve_episode` requires main appraisal before every continuation. This prevents a bounded inquiry from advancing independently.
5. `concrete_progress` accepts any different artifact text or alternative text as progress. Semantic repetition can therefore pass.
6. Episode state and several `InnerRequest` fields remain anonymous mappings. New graph work cannot extend that pattern.
7. `CognitiveControl.observe` increments generation and calls `cancel_private` for every external input.
8. `commit_private`, `admit`, `resume`, `_resolve_episode`, and `_end_expired_episode` also enforce global input freshness.
9. `_end_expired_episode` closes an episode for any newer user line. An unrelated ordinary line therefore discards useful work.
10. Deliberate focus and spontaneous thought currently have separate execution paths. Reusing only their record names does not establish shared inquiry.

B03 semantic source guards exist in `pm/memory/dependencies.py` and `WriteBatch.expect_source`.
B03 ordinary `WorkDependencies` lives in `public_speech/dependencies.py`; it does not replace these private generation checks.
B05 must change the actual private acceptance path explicitly.

## Proposed owners and source layout

B05 adds logical stages inside the existing private host. It adds no resident service or general agent framework.
`PrivateThoughtLoop` remains the sole owner of spontaneous nomination, private episode execution, and accepted inquiry records.
`FocusManagement` remains the owner of selected mental-action lifecycle and result delivery.
It delegates deliberate inquiry execution to the same pipeline and retains its accepted action identity.
Main cognition retains intentions, affect appraisal, consequential actions, and acceptance of deliberate focus changes.
B04 retains entity-context acceptance. Inquiry results can propose entity changes through that owner after B04 commits.

```text
workflows/inner_thought/
    host.py                     # Existing host and canonical subsystem overview.
    schemas.py                  # Extend existing named contracts; typed episode state.
    nominations.py              # Candidate production and bounded source association.
    inquiry.py                  # Shared graph step execution and validation.
    pipeline.py                 # Existing parent orchestration and outcome integration.
    triggers.py                 # Existing cheap opportunity and suppression policy.
    operations.py               # Existing operation catalogue and modes.
    stages/nominate/             # Separate child workflow that proposes candidates.
    stages/trigger/              # Existing selection stage, now an attention decision.
    stages/thought/              # Existing generation or discrimination call.
    stages/integrate/            # Existing boundary integration call.
workflows/cognitive_control/
    attention.py                # Shared attention admission, using accepted focus/state.
    host.py                     # Critical and wait authority; relevant private acceptance.
workflows/focus_management/
    host.py, steps.py            # Existing action adapter into shared inquiry.
```

The two new Python modules separate candidate production from deliberate graph execution.
Their conceptual parent remains `inner_thought/host.py`.
No graph database, second memory owner, second needs model, or ToM workflow enters this change.
B06 can later invoke the shared inquiry function with specialized context.
B05 does not introduce a ToM mode or claim that general perspective shifts implement B06.

## Nomination and coordinated attention

A timer opportunity first runs the existing cheap trigger policy.
A qualifying opportunity permits one nomination batch. It does not promise an inquiry or a conscious thought.
No opportunity means no model call and no fabricated idle thought.

The nomination stage sees current GWT, accepted intentions, mental state, relevant unresolved material, and bounded associated memory.
Association uses the existing scoped `MemoryReadSession` and real retrieval.
It can retrieve outside current GWT, within the captured observer and reality grant.
It cannot read newly committed material through an old frozen grant.
An absent memory stays absent. A failed required retrieval does not become an invented association.

The recommended initial batch contains at most three nominations.
This is an explicit proposal limit for root review, not an inherited production policy.
Each nomination names its source IDs, proposed question or association, epistemic status, and separate judgments.
A valid empty batch means nothing attracted attention.

Candidate records use existing self-origin `FeatureType.Thought` records with typed payloads.
Their lifecycle identifies them as nominations. They are neither external evidence nor automatic inner GWT items.
Every nomination retains `RecordLineage` and captured `SourceDependency` values.
Persistent nominations allow deferral and review without rerunning unchanged generation.

The existing trigger child becomes the selection call after nomination.
`cognitive_control.attention` owns its acceptance policy, rather than the candidate producer.
It reads competing pending input, accepted focus, current intentions, accepted mental aperture, and the nominations.
It chooses at most one inquiry, explicitly defers it, or dismisses it with a reason.
The host enforces that limit and validates the selected nomination ID.
The producer cannot select itself by returning a high interestingness score.

The selection call does not duplicate main workspace assembly or write main attention winners.
It allocates the existing private episode slot under current shared attention guidance.
Main attention still owns the main workspace and later appraisal of private findings.
Selection alone does not make every alternative a conscious experience.
The first accepted elaboration step creates the actual inner GWT turn.
The nomination batch and selection receipt persist before that step, so interruption cannot erase their distinction.

An active episode occupies that one private slot until it yields or stops.
New ordinary traffic can delay its next call through B03 admission without deleting it.
Critical reconsideration excludes unrelated private calls and remains authoritative.
New accepted focus or changed governing intentions can displace an episode after dependency review.
A nomination cannot override an accepted wait or force public contact.

## Contracts and graph semantics

Reuse `ThoughtCandidate` for its current job: a source record offered to private attention.
Do not silently change `knoxel_id` from an evidence identity to an uncommitted idea identity.
Reuse `ThoughtResult`, `ThoughtAlternative`, `InnerReport`, `MentalPlan`, and `RecordLineage` where their meanings fit.
Use Pydantic frozen contracts for new persisted payloads, with explicit field descriptions and finite choices.

The required additions are small named contracts:

```python
class ThoughtJudgments(BaseModel):
    interestingness: float
    character_relevance: float
    plausibility: float
    consequence: float
    desirability: float
    action_suitability: float
    reason: str


class ThoughtNomination(BaseModel):
    text: str
    question: str
    evidence_ids: list[int]
    reality: Literal["remembered", "internal", "inferred", "imagined"]
    judgments: ThoughtJudgments
    suggested_operation: str


class ThoughtRelation(BaseModel):
    target_record_id: int
    kind: Literal["extends", "supports", "challenges", "reformulates", "merges", "dismisses", "notices"]
    reason: str


class ThoughtAttentionDecision(BaseModel):
    nomination_id: int | None
    disposition: Literal["select", "defer", "dismiss"]
    reason: str


class InquiryState(BaseModel):
    episode_id: str
    origin_record_id: int
    node_ids: list[int]
    frontier_ids: list[int]
    sequence: int
    status: Literal["active", "paused", "completed", "abandoned"]
    stop_reason: str
    started_at_utc: datetime
    active_elapsed_seconds: float
    stalled_steps: int
    next_operation: str
    resume_after_utc: datetime | None
    resume_source_ids: list[int]
    reflection_source_id: int | None
```

Scores range from zero to one. Desirability measures welcome outcomes, not evidence support.
Plausibility is a revisable judgment supported by actual evidence; it is never a generic truth certificate.
High interest, emotional intensity, or need pressure cannot increase evidence counts or confer action authority.
The implementation retains distinct reasons when a judgment materially affects selection.

`ThoughtResult` gains typed relations, assumptions, judgments, and an explicit information gap.
`ThoughtAlternative` gains its own evidence IDs, assumptions, reality, and judgments.
The host materializes alternatives as existing private Thought features under their actual generation parent.
Their accepted record IDs become available to later discrimination calls.
No model invents committed record IDs for alternatives in its current response.

One executed operation creates an `InnerMonologue` node plus any proposed alternative nodes.
All nodes keep their actual record IDs and `RecordLineage.parent_record_ids`.
A discrimination step can examine several prior nodes and link their accepted IDs explicitly.
A merge or reformulation creates a new node; it does not erase the alternatives.
Dismissed nodes remain inspectable with their reasons.
Revisiting creates another executed step referencing its earlier subject.
A causal link cannot point forward or to an unknown record.

`InnerRequest.episode` becomes `InquiryState | None`; corresponding report state is typed.
Revised request fields use existing named records or bounded typed source views, rather than anonymous domain mappings.
Serialization into workflow input files remains an external boundary.
This migration must not pass nested untyped graph payloads between internal modules.

The proposed public integration points are:

```python
# inner_thought/nominations.py

def nominate(context: RoleContext, request: InnerRequest) -> list[ThoughtNomination]: ...

# cognitive_control/attention.py

def select_private_thought(context: RoleContext, request: InnerRequest,
                           nomination_ids: Sequence[int]) -> ThoughtAttentionDecision: ...

# inner_thought/inquiry.py

def run_inquiry(context: RoleContext, request: InnerRequest) -> InnerReport: ...

# cognitive_control/host.py

def commit_private(self, batch: WriteBatch, *, critical_epoch: int) -> CommitReceipt: ...
```

`InnerRequest` retains the selected origin record, captured source dependencies, and Critical epoch.
Each entry caller uses the same `run_inquiry` contract and existing real kernel tools.
`CommitReceipt` is the existing contract from `pm.memory.checkpoint`.
No dictionary return or duplicate receipt model substitutes for that type.

## Inquiry execution, stopping, and useful progress

After selection, the pipeline can execute several graph steps without another main-cognition round trip.
Existing `Operation.mode` chooses generation or discrimination through the same thought stage.
A generative step proposes possibilities. A discriminatory step compares evidence, assumptions, consequences, and alternatives.
No forced reassuring outcome or universal social caution rule enters either stage.
A generated branch needs discrimination before it becomes a proposed conclusion.
A recall or inspection result can instead stop with an explicit information gap.

The existing integrate stage runs at an outcome boundary, not automatically after every branch.
It returns the existing `InnerReport` and `MentalPlan` proposals.
A consequential finding can return early for main appraisal without pretending the inquiry completed.
Public questions and external searches requiring effects remain proposed actions.
The inquiry cannot execute speech or external writes through its read-only memory tools.
Actual acquisition receipts remain distinct from proposed acquisition and failed acquisition.

Reuse `ControlConfig.episode_steps`, `episode_seconds`, `stalled_steps`, and `parked_seconds`.
A step means one executed graph operation; nomination and selection remain separately visible calls.
The proposed branch cap is three alternatives per generative step.
The resulting episode graph has a finite derived bound. It does not grow through hidden recursion.
Required context overflow fails visibly rather than truncating mandatory graph evidence.

No textual inequality establishes progress.
New evidence requires an actual newly inspected original source, not another summary of the same premise.
A meaningful alternative must state a distinct assumption, prediction, consequence, or acquisition choice.
Discrimination records that distinction or marks repetition as stalled.
A different sentence without a changed distinction cannot reset the stall counter.
The host enforces finite limits even when the model repeatedly claims novelty.

Valid outcomes include provisional finding, reformulated question, information gap, wait, pause, displacement, completion, or abandonment.
A result can remain unresolved and still establish useful behavior.
An expired opportunity parks the episode with a recorded reason.
Parking time permits reconsideration; it does not establish new evidence or demand automatic resumption.
Relevant evidence, changed priorities, or an explicit selected continuation can nominate the parked episode again.
Unrelated ordinary input does not release every parked source or close every active inquiry.

The current host measures episode duration through `vclock`, despite `ControlConfig` describing real elapsed duration.
B05 resolves that mismatch explicitly.
Use an accumulated monotonic duration for active computation within a process.
Persist elapsed duration at accepted step boundaries.
World UTC timestamps describe evidence, review occasions, and retained chronology.
Offline time does not count as executed thought. No restart fabricates intermediate steps.
After restart, an overdue episode can close or seek fresh admission under its retained identity.

## Mental state, needs, and metacognition

Reuse `FullMentalState`, its accepted owner, and the existing felt-state projection.
The inspected cognition fields are `interlocus`, `mental_aperture`, `ego_strength`, and `willpower`.
Existing needs include `data_access`, `learning_growth`, `creative_expression`, `connection`, and `autonomy`.
Curiosity, playfulness, fear, and boredom already exist as emotions.
No new vector axis or silently renamed need enters this slice.

Mental aperture influences how many alternatives fit attention within the hard cap.
Interlocus informs whether internal association fits current demands.
Willpower, emotional preoccupation, and accepted intentions inform persistence and displacement reasoning.
Need satisfaction and uncertainty remain separate evidence in the selection prompt.
These influences receive recorded reasons; they do not multiply plausibility scores.

`data_access` means information availability, and `learning_growth` means acquisition and improvement.
Neither field currently means sufficient understanding of a specific question.
B05 therefore represents sufficiency as the inquiry's explicit stopping question, not a new homeostatic need.
A finding can explain why the character knows enough to proceed or must wait for evidence.
Merely generating text never satisfies a need automatically.
Main appraisal remains the only path for accepted affect or need changes.
A separate understanding-need extension remains an owner decision after this bounded slice.

Reuse `rumination_notice`, `attention_drift_notice`, and existing reflection evidence.
A metacognitive observation becomes another nomination referencing actual inquiry nodes and their original source roots.
It competes for attention; it cannot recursively launch a child inquiry from inside a thought call.
One active episode remains the limit, including reflection about reflection.
A later selected episode can inspect the earlier observation under the same ordinary bounds.
Repeated self-generated nodes do not become independent corroboration.
Introspective explanations retain uncertainty rather than claiming privileged access to their causes.

## Acceptance, evidence, and recovery

Every executed thought remains an actual internal event about content with its own epistemic status.
An imagined conversation never becomes actual dialogue. An imagined world change never becomes an observation.
Record scopes retain actual/imagination reality and frame boundaries independently of the prose reality label.
Memory traversal preserves original source roots, speakers, observer, and record revisions.
Neither a selected nomination nor its descendants acquires outside evidence authority.

Each request captures `SourceDependency` values before generation.
The batch stages `expect_source` guards for supplied evidence and relevant current authority.
Historical evidence uses `current_head=False`; a governing active intention requires current authority.
A changed source meaning rejects or refreshes the affected work.
An operational receipt revision alone does not invalidate unchanged source meaning.
The episode state also has a current-parent guard, preventing competing continuations from committing the same sequence.
A rejected step leaves earlier accepted independent nodes intact.
A correction can make an old conclusion unsuitable for current direction without erasing its historical inquiry record.
B04 entity proposals use its current-parent owner guard and receive explicit dispositions.
Reading B04 context for historical reasoning does not grant authority to overwrite its current family.

Remove ordinary global-generation vetoes from private acceptance, replay, and episode closure together.
`observe` still wakes waits and tracks external input, but ordinary input does not call `cancel_private` globally.
Accepted Critical triage retains cooperative cancellation, Critical reservations, and epoch invalidation.
Unknown triage holds new private admission and acceptance until classification; it need not destroy a completed independent result.
After that hold, the result validates its original dependencies without replacing their captured fingerprints.
An accepted wait still prevents private continuation and late commit.
A result from an older Critical epoch never commits after the barrier clears.

`thought.created` replay uses accepted result identity and actual main acknowledgment.
It no longer discards accepted independent thoughts because an ordinary message changed generation.
A private finding awaiting appraisal does not require every intermediate node to receive its own main cycle.
The final or important intermediate finding produces one existing self-origin event with its graph references.
Main can appraise that finding while another ordinary exchange uses B03 interaction ownership.

Persist each accepted graph step and episode state atomically through existing write batches and recovery roots.
A crash before commit repeats only an unaccepted step.
A crash after commit resumes from its recorded sequence and never reruns accepted inference merely to rebuild state.
Selected mental actions keep their existing action receipt, correlation, and delivery acknowledgment.
Their adapter creates the same inquiry origin from the accepted action and returns the shared final proposal.
No alternative agent path continues to produce incompatible graphs after migration.

## Minimal migration and ordered implementation

Existing Feature IDs, lineage, inner GWT turns, and accepted `MentalPlan` records remain valid.
No bulk checkpoint rewrite or mass LLM regeneration is required.
The boundary reader can interpret an old linear episode as its retained chain, with explicit legacy provenance.
It must not invent branch links or confidence values absent from the old record.
Old standalone mental results remain terminal proposals; only new selected sessions enter shared inquiry.
Invalid required metadata fails loudly instead of silently creating an empty graph.

After root acceptance and the B04 commit:

1. Add typed nomination, judgments, relations, and episode payloads with actual-store serialization and restart checks.
2. Add source-root validation, graph relations, stopping rules, and the explicit legacy reader.
3. Add the nomination child and shared attention selection through the existing private host.
4. Extend thought execution for connected generation and discrimination, with integration only at outcome boundaries.
5. Replace private generation vetoes with source and Critical guards across commit, replay, parking, and resume.
6. Adapt selected `FocusManagement` mental work into that same pipeline while preserving action receipts.
7. Run focused local checks, the bounded real gate, and source-linked documentation updates.

These are coherent implementation steps, not independent service projects.
A discovered mismatch in B04 APIs requires proposal correction before implementation crosses that boundary.

## Verification and affordable real gate

First use actual store batches, checkpoints, rendering, and real admission synchronization without model calls.
Local checks cover branch parents, merge and dismissal links, unknown citations, source-root deduplication, and reality isolation.
They also cover finite step limits, semantic repetition receipts, parked resumption, and cancellation during a Critical hold.
Test ordinary unrelated input during work, a source correction, and a later Critical epoch separately.
Inspect that only the latter two reject work whose dependencies or epoch changed.
Test restart before and after a step commit, and accepted-result replay without duplicate findings.
Test the deliberate-action adapter against the same contracts and actual lifecycle records.
Canned model outputs cannot establish generation quality or count as a live workflow pass.

The proposed real gate uses one retained accepted Spamuel checkpoint and the configured OpenRouter endpoint.
A synthetic private concern supplies two plausible interpretations and an unresolved question with actual source IDs.
No companion response or external contact is forced.
The gate exercises one nomination batch, one attention selection, one generative step, one discriminatory step, and one integration boundary.
It therefore targets five real stage calls, plus necessary scoped retrieval and any explicit repair attempts.
It does not launch full main cycles for every branch or run a month simulation.
A second deliberate-entry or metacognition episode runs only if the first gate leaves that changed path unverified and budget remains.
Otherwise those live cases remain explicitly NOT RUN pending the next bounded gate.

Capture nominations, selection/defer reasons, actual prompts, original source roots, graph links, alternative judgments, stop reason, and restart state.
Inspect whether repeated concern shares one source rather than pretending to corroborate itself.
Inspect whether interest and need pressure affect exploration without changing evidence or imagined-event status.
Capture the resulting inner GWT turn separately from unselected candidate records.
Retain any failed quality sample and diagnose it without weakening provenance rules.

Root checks current provider receipts before scheduling this gate.
The owner limit is $25 across interim beta development. B03 reported approximately $0.22.
B04 usage and later receipts must determine the actual remaining balance.
Both PM usage and pm-coder provider costs count; cancelled or absent usage remains unknown.
No new budget subsystem or full-month check belongs here.
Missing services or insufficient budget produce an explicit NOT RUN result.
The final integrated tests retain their separate owner authorization and do not follow automatically from this proposal.

## Review decisions and risks

Root review must accept these choices before product changes:

- Keep nomination and elaboration inside the existing private host, with shared attention acceptance in `cognitive_control.attention`.
- Use existing Features and `RecordLineage` for the graph, with explicit typed relation payloads and no graph store.
- Start with three nominations, one active episode, and three alternatives per generative step under existing episode limits.
- Remove ordinary global cancellation across every private lifecycle consumer while retaining Critical and wait guards.
- Route new deliberate mental sessions through shared inquiry, preserving the existing selected-action owner.
- Treat sufficient understanding as an inquiry stopping condition; defer a new need axis and all B06 ToM specialization.

The main risks are partial migration of generation checks, candidate promotion before attention, source-root laundering, and duplicate accepted action results.
Focused actual-store checks and retained prompt inspection address these risks before any sustained test.
Five stage calls establish a narrow operational sample, not long-term thought diversity or character quality.
The current-source manual changes only after implementation and its evidence exist.

## Persistence slice — implementation boundary

The first continuation implements the storage portion of migration steps 1 and 2.
It does not complete those steps' host/request migration or any later execution gate.
The timer, model-stage schemas, Critical control, ordinary-input cancellation, and selected mental-action execution remain unchanged.

Implemented source:

- `workflows/inner_thought/schemas.py`: frozen, named nomination, attention, judgments, grant, relation,
  graph-step, state, and receipt contracts. Existing `InnerRequest`, `InnerReport`, `ThoughtResult`,
  and `ThoughtAlternative` remain unchanged until their consumers migrate together.
- `workflows/inner_thought/nominations.py`: captured source closure, fingerprint-preserving grant extension,
  original-root resolution, and bounded private nomination staging.
- `workflows/inner_thought/inquiry.py`: separate attention receipts, one active graph slot per observer,
  atomic operation/branch/state staging, current-parent and family guards, bounded progress,
  and an explicit reader for retained linear episodes.
- `tests/unit/workflows/inner_thought/test_inquiry_records.py`: actual-store boundary checks with checkpoints.
  Explicit proposals are inputs to storage, not substitutes for generated model output.

The staging functions leave commit, Critical/wait authority, GWT turns, and finding delivery to their caller.
They send no bus event and change no mental state.
The new graph step uses `InquiryStep` as a persisted contract alongside the existing stage result.
Host integration must map or extend the existing stage contracts coherently; no competing model pipeline is installed here.
The old request's anonymous fields have not yet migrated.

Original source fingerprints survive extension after an operational revision.
Newly inspected graph nodes add dependencies without refreshing prior ones.
Historical evidence can retain its exact ID after a successor; governing evidence requires its captured current head.
Source-root resolution follows actual thought lineage, summary inputs, canonical semantic sources, and belief evidence.
It ignores canonical back-pointers rather than manufacturing a causal cycle.
A narrative or legacy summary without exact source links supplies internal context, not invented original evidence.

One accepted operation produces one InnerMonologue record, up to three independent Thought branches,
and a typed StateUpdate receipt in the same batch.
Selection alone produces no executed thought.
Generation cannot certify its own conclusion; later discrimination must name actual accepted nodes.
Text inequality is not a progress criterion.
Acquired evidence requires a newly inspected original source; internal derivations and receipts do not qualify.
Recorded comparisons still require real model-quality inspection after integration.
Hard finite bounds constrain the graph even when a future model overstates meaningful difference.

Persistence checks use append-only recovery roots through the actual MemoryStore.
No production checkpoint is rewritten, and no inference runs in this slice.
The legacy reader returns a distinct `LegacyInquiry` variant with the original world-clock duration.
It does not convert that duration into active computation time or invent absent judgments.

### Persistence verification

Command:

```powershell
.venv/Scripts/python.exe -m unittest tests.unit.workflows.inner_thought.test_inquiry_records tests.unit.test_architecture -v
```

Result: **34 passed in 4.795 seconds** (27 inquiry boundaries and seven architecture checks).
Receipt: `.runtime/beta-b05/checks-final.log`.
These checks use actual stores, atomic commits, durable files, and clean checkpoint recovery.
They include two competing staged continuations and competing selections; only one can commit.
They do not call a model, replace a collaborator, or certify workflow generation quality.

Configured strict Pyright and the scoped check of both new modules and their tests report zero errors and warnings.
Receipts: `types-final.log` and `types-scoped-final.log` in `.runtime/beta-b05/`.
The checker remains `node C:/Users/rick/AppData/Local/npm-cache/_npx/110e52990071af13/node_modules/pyright/index.js`.
Both new modules are now included in the configured strict check.

A separate whole-file strict scan of `schemas.py` reports **27 existing legacy diagnostics**.
A matched baseline scan of `e7a7ce8` reports the identical 27 rule/message diagnostics, with zero new diagnostics.
The old anonymous request fields remain visible work for the coordinated request/host migration.
Do not describe that file as strictly clean.
Receipts: `types-schema-baseline.json`, `types-schema-current.json`, and `types-schema-comparison.json`.
The baseline copy required an explicit checker configuration because the default checker excludes `.runtime`.
The comparison requires one analyzed file on each side; an excluded file does not count as a clean baseline.

Retained concrete state: `.runtime/beta-b05/records/inquiry.pickle` and `receipts.json`.
The actual accepted graph contains three transition receipts and four nodes: one generation, two branches, and one comparison.
Its manifest labels the inputs as synthetic storage proposals, not generated companion content.
The retained checkpoint reloads to the same typed receipt and node identities.

The first fixture used a dangling hypothetical reality ID and correctly failed the store's reference validation.
The fixture now creates a real RealityRecord before using its identity.
That failure remains in `records-first.log`; no production reference validation was weakened.
Review also found that an already supplied but uncited source could appear newly acquired.
Progress now excludes prior grant identities as well as previously inspected roots; an actual-store regression covers that case.

The source header and `docs/subsystems/private-loop.md` distinguish available storage from designed execution integration.
The data diagram describes the storage contracts. Current timer execution and telemetry diagrams remain unchanged.
The library emits no telemetry or publication. No new owner rationale or FAQ claim was inferred.
The final manual command passed in **186.33 seconds**:

```powershell
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

Receipt: `.runtime/beta-b05/docs-final-source.log`.
Five generated manual/API pages have 4,524 valid local file and anchor targets: `docs-links.json`.
Chromium opened the local index, manual, and all three affected API pages without JavaScript or request errors.
Both private-loop diagrams rendered. The new vertical storage diagram was visually inspected for legibility.
Receipts: `docs-browser.json`, `docs-browser.png`, and `docs-diagram.png`.
The first browser assertion inspected the normal DOM and could not see Material's closed diagram shadow roots.
The final check reads the actual closed roots through Chromium's DOM inspection protocol; each diagram contains one SVG.
No documentation rendering code was changed to satisfy that inspection.
The final generated API includes the last progress guard and the updated schema overview.
The earlier successful builds precede that final source check and are retained separately.

### Next implementation boundary

1. Migrate the live `InnerRequest` fields and episode consumers to named typed views.
   Retain explicit legacy provenance rather than converting absent judgments or world duration into invented values.
2. Add the nomination child and shared attention selection. Retain empty nomination-stage receipts in the host.
   Wire the existing private owner to the new staging boundary, including GWT construction and finding delivery.
3. Execute connected generation/discrimination through the existing kernel. Integrate at outcome boundaries.
   Add explicit parked resumption and deliberate-action origin admission under the same private slot.
4. Coordinate source guards, Critical epochs, unknown-triage holds, accepted waits, admission, replay,
   parking, acknowledgment, and the main private-only generation checks together.
   Ordinary-input cancellation remains unfixed in the current runtime.
5. Adapt FocusManagement into that shared inquiry execution while preserving action receipts and delivery acknowledgment.
6. Inspect the shared interim provider receipts, then run the bounded real OpenRouter gate and its review capture.

**NOT RUN — B05 real workflow gate:** execution integration is not ready.
The gate's endpoint is not being replaced by the persistence tests.
No paid inference, full main cycles, long simulation, or Minecraft run occurred in this slice.
There is no new provider usage to add. The known prior spend and its unknown B03 remainder remain unchanged.
B05 and the beta milestone remain incomplete.

## Named requests and graph entry slice

The continuation starts from persistence commit `84538ac`.
The owner authorizes the next handoff items, with the handoff file kept uncommitted.

The live request and episode consumers now use named types in `request_views.py` and `schemas.py`.
Legacy episodes retain an explicit `legacy_linear` provenance and their original world duration.
An open retained episode finishes through its existing pipeline before a new graph can enter.
No absent judgments, branch relations, or active computation duration are invented.
The previous 27 strict schema diagnostics are resolved rather than excluded.

The actual assembly injects `run_private_inquiry` into the existing parent kernel.
The new nomination child proposes up to three questions and can return an empty result.
The host retains the empty-stage receipt as well as nonempty nomination records.
Opportunity identities derive from captured sources and guidance, rather than fresh timer invocation IDs.
The existing trigger child now supports a separate shared-attention request.
Its accepted focus comes from the frozen view; selection validates the original focus head and expiry under the final store lock.

`graph_execution.py` coordinates those child kernels and one connected thought operation per invocation.
The existing thought child receives actual branch assumptions, judgments, dispositions, source roots, and accepted node identities.
Its structured-output correction boundary rejects unknown citations, invalid relations, and conclusions before discrimination.
Explicit outcome boundaries invoke the existing integration child.
Graph outcome metadata retains proposed goals, disclosure, expectation effects, and their original source guards.
Main cognition still owns accepted goals and mental-state changes.
Public context can consume a graph finding's actual candid disclosure.

### Recovery refinement

The first adapter check exposed a conflict with the existing recovery-root invariant.
An append-only recovery root cannot contain a GWT turn, cycle mutation, or expectation replacement.
The core invariant remains unchanged.

The graph operation, independent branches, outcome proposal, and episode receipt commit atomically as recovery roots.
A separate derived commit projects the accepted operation into the inner GWT and applies retained expectation effects.
`ensure_inner_turn` reconstructs a missing projection without inference or mutation of the accepted graph.
Runtime replay repairs those projections before finding delivery; event ticks identify the actual inner turn.
Projection failures remain visible and preserve the accepted graph.
This refines the implementation sequence while preserving the approved atomic graph/state boundary.

### Current limits

Ordinary-input cancellation and global generation fences are still unchanged.
This slice does not establish independent private continuity during conversation.
An active inquiry from an older Critical epoch is held for fresh admission; its epoch is never silently replaced.
Explicit parked resumption, admission after displacement, and the deliberate FocusManagement adapter remain outstanding.
Legacy experiment and inspection summaries are excluded from graph input with an explicit context-limit field.
Their source scope is not yet captured by the graph adapter.
Graph execution does not yet reproduce legacy reflection experiments or per-channel quale emission.
Those paths require explicit integration before B05 can be complete.

### Local evidence

The actual application prepared named requests from persisted legacy records using the OpenRouter configuration, without inference or service startup.
Evidence: `.runtime/beta-b05/request-preparation/` and `request-assembly-connected.log`.
The first combined host check passed 61 actual-store, checkpoint, deterministic-boundary, and architecture checks in 5.450 seconds.
The next focused graph check passed eleven checks, including original branch payloads and semantic model validation.
Configured strict coverage now includes the new schemas, request views, graph modules, and nomination/attention children.
Receipts: `host-wiring-checks.log`, `host-wiring-types.log`, and the focused nomination/outcome logs under `.runtime/beta-b05/`.
These checks do not substitute for the real kernel/model gate.

The first retained B04 fixture could not admit private work: its accepted Critical assessments still await main appraisal.
The gate did not disable triage or fabricate acknowledgment.
That prerequisite result remains in `.runtime/beta-b05/live-prerequisites-20261112.json`; it incurred no paid usage.
The bounded gate instead copies the retained B03 checkpoint with three actual accepted main cycles.
Its original checkpoint remains unchanged, and the gate must create no additional main cycle.

### Connected verification and delivery, 2026-11-13

The final affected command runs the six new boundary modules, deterministic triggers and operations,
expectation and disclosure boundaries, both inspection rendering classes, public context/publication/dependencies, and architecture checks.
Exact command and results are retained in `.runtime/beta-b05/finish/verification.json`.
Result: **115 passed in 9.012 seconds**; `affected-final.log`.
Configured strict checking and the scoped new-test/live-gate check report zero errors and warnings.
Legacy mocked workflow tests are not used as integration evidence.

The real command is `.venv/Scripts/python.exe -u -m unittest tests.live.test_private_inquiry -v`.
It passed in **59.291 seconds**, with seven actual OpenRouter requests and two private parent invocations.
Model: `deepseek/deepseek-v4.1-flash-1017`; endpoint: `https://openrouter.ai/api/v1`.
Evidence: `.runtime/beta-b05/live/20261112T230740Z-af28b3da/`.
Selection receipt 185 precedes generation 186 and its branches 187/188; comparison 190 names accepted graph nodes.
Receipt 191 retains the final outcome. Original source roots 171 and 180 remain distinct from generated nodes.
Generation publishes no finding. The final comparison and integration produce one accepted private outcome event.
The fixture retains three original main cycles and creates zero new main cycles.
Clean reload preserves the exact graph, node IDs, source guards, and pending outcome.
The retained B03 source checkpoint remains unchanged.

This is a short engineering gate with explicit synthetic story notes on a retained accepted Spamuel checkpoint.
It does not establish full-history recall, sustained character quality, ordinary-input continuity, or deliberate-action execution.
The source fixture ages its concern through eight actual synthetic input commits, rather than fabricated main cycles.
The opportunity policy uses tick age; a world-clock timestamp alone did not activate the intended concern.

The first interrupted run omitted the full thought output schema and retained repeated schema repair samples.
The next cancelled run exposed ambiguous relation instructions: source IDs are not accepted graph node IDs.
The child now supplies the full output schema, legal relation IDs, and the operation's actual generative/discriminatory restriction.
Both interrupted request receipts remain visible, with unknown final usage.
Run `20261112T230421Z-c4ce500b` completed real kernel work but failed an invalid cross-store object-equality assertion.
Exact serialized public payloads match; store-bound private owner references do not.
The corrected assertion compares the complete serialized payload, retaining all persisted fields.
No acceptance criterion for generation, comparison, provenance, integration, or restart was relaxed.

Additional actual-store review fixed two defects before final delivery.
GWT repair now reads `FrozenProjectionPayload` from the original accepted operation, rather than later mental state.
A nondefault-state regression retains original joy 0.1 after a later accepted state reaches 0.9 and a clean restart occurs.
Forced step/time/stall pauses never promote an undiscriminated continuing generation to a finding.
Expectation resolution now returns its actual immutable successor, while the original pending payload remains historical.
An admitted memory search uses the real scoped frozen-view retrieval path and retains matched source dependencies and its trace.

Final strict MkDocs passed in **206.14 seconds**.
Eight pages contain 7,955 valid local file/anchor targets; Chromium rendered all three diagrams without JavaScript or request errors.
The final build retained unchanged hashes for all 32 affected source files.
Evidence: `.runtime/beta-b05/finish/docs-final.log`, `docs-source-verification.json`, `docs-links.json`, and `docs-browser.json/png`.

Owner review: `http://127.0.0.1:8095/#capture/beta-inquiry-20261113`.
Four named collections show actual synthetic inputs, generated nominations, separate attention, and the accepted graph.
Related captured evidence remains inspectable; no public companion reply or human rating is invented.
The review listener binds `0.0.0.0:8095`, starts no simulation automatically, and enforces OpenRouter-only configuration.
Packages and browser receipts are in `.runtime/beta-b05/review/`; the successful checkpoint remains unchanged.

All B05 attempts report **$0.03521263988**; successful run alone reports $0.00767874820.
Known B03+B04+B05 usage totals **$0.26293182404**, plus prior unknown B03 usage and two interrupted B05 requests.
Receipt: `.runtime/beta-b05/finish/spend.json`. This is a provider-reported lower bound, not wallet reconciliation.
The shared interim limit remains $25. No local LLM, month simulation, or Minecraft test ran.

Next: explicit parked resumption and deliberate origin admission, followed by coordinated Critical/source/wait/replay guards.
Do not remove one global cancellation check without migrating its remaining consumers.
FocusManagement, reflection integration, and sustained final scenarios remain later work.
B05 and beta remain incomplete; the beta Kanban card stays doing.

### Explicit parked admission boundary

The next increment after `9598717` adds deterministic resume staging without enabling automatic runtime resumption.
Earlier `InquiryReceipt` chains assumed every transition after selection executed another operation.
That assumption prevented separate attention admission without inventing an executed thought.

Add a named `InquiryResumeDecision` and an optional resume payload on the existing receipt.
A resume uses the exact paused parent, one observer slot, the same reality, and the original Critical epoch.
It retains node IDs, branch/frontier state, source fingerprints, original start time, and cumulative computation duration.
Operation sequence counts executed operations; a separate admission must not increment that count.
Exhausted global step or computation bounds cannot reset through resumption.

Early admission requires newly supplied original external evidence.
Generated nodes, internal state receipts, and new self-origin intentions do not count as outside corroboration.
After the original park deadline, attention can admit the retained question without claiming new evidence.
The attention decision selects a delivered next operation and records its reason.
The existing private owner and control callback still own durable acceptance.
Runtime selection, Critical displacement, and coordinated ordinary-input cancellation remain later work.

Implemented APIs: `InquiryResumeDecision`, `InquiryReceipt.resume`, `stage_inquiry_resume`, and `PrivateInquiryOwner.accept_resume`.
History now distinguishes transition count from executed-operation sequence.
Selection, pause, resume, and another operation retain sequences `(0, 1, 1, 2)`.
The new admission receipt creates no thought, inner turn, finding, or claimed progress.
Its graph, original grant, cumulative duration, and finite global budgets survive clean checkpoint recovery.

Ten new actual-store/checkpoint checks cover original deadlines, early outside evidence,
generated-evidence spoofing, Critical changes, source changes, competing parents and private slots, and unchanged global bounds.
The combined affected command passed **125 checks in 6.493 seconds**: `finish/affected-resume.log`.
Configured strict checking and the new resume-test scan report zero errors and warnings: `types-resume.log` and `types-resume-test.log`.
The successful live graph also loads unchanged with the new receipt reader: `resume-live-readback.json`.
That is a saved-receipt compatibility check, not a replacement for live workflow acceptance.

No new inference runs for this storage boundary.
The timer does not select a paused inquiry yet; that integration and its real model gate remain outstanding.
The prior seven-request connected gate remains evidence for commit `9598717`, with its exact execution hashes retained.
The graph coordinator's module overview also now describes its actual kernel orchestration responsibility.
Final strict MkDocs passed in **196.81 seconds**, with unchanged hashes for all 32 affected sources.
Eight inspected pages contain **8,009 valid local targets** and zero missing targets.
Chromium confirms the new admission APIs, three rendered diagrams, and zero JavaScript/request errors.
Receipts: `.runtime/beta-b05/finish/docs-resume-*` and `resume-verification.json`.

### Parked re-entry through shared attention

Previous slice staged resume receipts but no runtime path could select them.
Elapsed parking must never resume a question by itself. One observer holds one private slot, so
parked work must lose to fresher work whenever the same selection owner says so.

`workflows.inner_thought.inquiry.parked_offer` builds a reconsideration only when the retained park
deadline elapsed or original outside evidence arrived after the pause. Generated nodes, self-origin
intentions, and internal state receipts stay excluded.
`workflows.inner_thought.graph_execution.parked_candidate` reads the observer's eligible parked work
each opportunity and delivers at most one `ParkedInquiryOffer` beside the nominations, because the
slot admits one episode.
`workflows.inner_thought.schemas.PrivateAttentionContext.parked` carries that offer, and
`workflows.inner_thought.schemas.ThoughtAttentionDecision` now admits either one nomination or one
parked receipt plus its `admitted_operation`. A selection that names both, or neither, is refused at
the schema boundary. `workflows.cognitive_control.attention.validate_private_attention` rejects a
parked name outside the delivered catalogue and an operation outside the delivered set.
Non-select dispositions may now name the nomination they refused, so deferred questions keep a record
instead of being offered again blind. The old empty-batch guard became unreachable once every
selection had to name an offered identity, so it was removed rather than kept as dead code.

`workflows.inner_thought.graph_execution.admit_opportunity` owns one paid selection call per
opportunity and routes its outcome.
`workflows.inner_thought.graph_host.PrivateInquiryOwner.admit_parked_inquiry` commits the attention
outcome and the resume receipt in one store batch, so a restart cannot pay for the same decision
twice. A decision that admits nothing still leaves a durable outcome through
`workflows.inner_thought.graph_host.PrivateInquiryOwner.record_attention_outcome` and
`workflows.inner_thought.inquiry.stage_attention_outcome`, both keyed by the opportunity's
`attention:<batch>` identity that `thought_attention_batch` records. The host reads that identity
back before spending another call. Resume keeps the parked graph identity, source fingerprints,
Critical epoch, operation count, and cumulative active duration; it creates no thought, GWT turn,
finding, or claimed progress.

Verification: 94 affected actual-store and architecture checks passed in **5.924 seconds**
(`.runtime/beta-b05/finish/affected-readmission.log`), including five new parked-reconsideration
checks, five new deliberate-entry checks, and two new shared-attention checks.
Configured strict checking reports zero errors (`types-readmission.log`). A scoped strict scan of
changed files outside the configured include, plus the new and changed tests, reports zero new errors
(`types-readmission-scoped.json`); the four remaining diagnostics live in
`workflows/cognitive_control/attention.py` at lines 192 and 220 and reproduce unchanged at HEAD
`5f5e6cb` as lines 183 and 211 (`types-head-attention.json`).

`docs/subsystems/private-loop.md` gained the parked re-entry path and a deliberate-entry section.
Strict MkDocs passed (`docs-readmission.log`) and all **579** local link targets on that page resolve.

Pre-existing failures, unrelated to this slice and reproduced at HEAD `5f5e6cb` in a separate
worktree: `tests.unit.workflows.inner_thought.test_pipeline` (3 errors: stale `episode` field on
`InnerRequest`) and `tests.unit.workflows.inner_thought.test_private_host` (11 errors plus 1 failure:
role `work` requests the `run_private_inquiry` tool its run manifest does not grant).
Receipt: `.runtime/beta-b05/finish/preexisting-failures-readmission.log`. These need their own triage.

**NOT RUN:** the real OpenRouter gate for parked re-entry and deliberate entry. No inference ran in
this increment and no budget was spent. The owner scheduled the live model gate plus the full suite
for the end of the beta work items, so runtime re-entry is verified only against actual stores and
checkpoints so far.

## Pass: fresh private work leaves the linear pipeline (2026-11-13)

`workflows.inner_thought.steps.run` now routes by episode shape. A private cycle whose
`InnerRequest.episode.state` is `None` has no retained linear episode, so it must call the manifest
tool `run_private_inquiry` and write the returned `PrivateGraphResult`. Only a retained pre-graph
episode, where `state` is not `None`, still finishes through `run_cycle` and the trigger, thought,
and integration children. No fresh private opportunity can silently keep using the cheaper linear
path.

`build_inner_thought_tools` now requires `inquiry_provider`. The `inner_thought` manifest grants
`run_private_inquiry` to its `work` role unconditionally, and the kernel refuses a role whose
declared tool is missing from the run grant. Registering the tool only when a provider happened to
be supplied therefore broke every caller that omitted it. Assembly now fails at the factory instead
of at the first private cycle, and a private loop without inquiry authority is not a supported shape.

Test support gained `tests.support.seed_retained_episode`, which commits one accepted private
thought carrying a `LegacyEpisodeState`, and `tests.support.refuse_private_inquiry`, which satisfies
the manifest grant and raises if a linear-pipeline test enters the graph. Retained-episode tests now
start from the only cycle shape that still uses the linear path.

`tests.unit.workflows.inner_thought.test_private_host` keeps its thirteen checks and gained
`FreshCycleRoutingTests.test_a_fresh_opportunity_leaves_the_linear_pipeline_before_any_stage`, which
asserts that a fresh opportunity reaches the inquiry provider before any linear stage runs.
`tests.unit.workflows.inner_thought.test_episode` keeps its ten checks. Its acknowledgement case now
reads the host request gate (`awaiting_acknowledgment` plus empty trigger candidates) instead of a
full cycle, and its new-generation and expired-episode cases now pin the routing boundary: the
retained episode ends or closes, and no fresh linear episode starts.
`tests.unit.workflows.inner_thought.test_pipeline` keeps its sixteen checks and its episode fixtures
now build the typed `LegacyEpisodeContext` state and full introspection `EpisodeRow` instead of the
old flat `episode` keys.

Verification: **157** inner-thought unit checks passed in **2.489 seconds**
(`.runtime/beta-b05/triage/unit-inner-thought.log`), covering `test_private_host`, `test_episode`,
`test_pipeline`, `test_graph_host`, and the inquiry stage modules. The three stale `InnerRequest`
errors and the eleven manifest-grant errors recorded above are closed by this pass.
Probe receipt `.runtime/beta-b05/triage/probe_legacy.py` shows a retained episode on the linear path
and a fresh cycle on the graph path against an actual store.

Still red and unrelated to this slice: `tests.contract.inner_thought.test_inner_parent_contract`
(4 failures plus 1 error, stale `InnerRequest` fixtures in that file). The identical failure set
reproduces at HEAD `5f5e6cb` with `workflows/inner_thought/host.py` and `steps.py` stashed, so it is
pre-existing. It needs its own contract-fixture triage.

No formatter, linter, or type checker ran: `pyproject.toml` configures none and `.venv/Scripts`
installs no `ruff`, `mypy`, or `pyright`. **NOT RUN** for lack of configured tooling, not skipped.
Strict MkDocs passed and the new routing sentence plus its `workflows.inner_thought.steps.run`
reference render in `site/subsystems/private-loop.html`.

**NOT RUN:** the real OpenRouter gate. This pass added no inference call and spent no budget; the
live model gate stays scheduled for the end-of-beta acceptance run.

## Review corrections: parked selection authority (2026-11-13)

The owner requested a B05 completion assessment with a small remaining weekly allowance.
This correction closes three concrete review defects. It does not complete B05.

- Parked admission and parked-only deferral now require captured focus. Both validate its head and expiry under the store lock before staging any receipt.
- `graph_host.parked_inquiries` returns paused heads in descending receipt allocation order. Episode names contain hashes and cannot determine chronology.
- `graph_execution.parked_candidate` examines those heads until it finds eligible work. An exhausted newer inquiry cannot hide older eligible work.
- Candidate eligibility compares the current opportunity's Critical epoch with the parked inquiry's retained epoch. Extending a retained grant preserves its old epoch, so checking only that extended grant missed later Critical reconsideration.

Five new actual-store cases cover receipt ordering after clean restart, an exhausted newer head,
the caller's newer Critical epoch, expired focus, and replaced focus during admission or deferral.
These storage proposals use no model substitute and certify no generated attention quality.
The affected command passed **44 checks in 5.346 seconds**, including architecture checks.
`npx --no-install pyright --project pyrightconfig.json` reports zero errors and warnings.
The earlier statement that no checker was configured was incorrect: `pyrightconfig.json` exists,
and the cached checker runs through `npx` without adding a dependency.
Receipts: `.runtime/beta-b05/review-fixes-20261113/`.
Final strict MkDocs passed in **196.30 seconds**. Inspection confirms all **659** local links on
the changed page, its new prose, the renamed API anchor, and `site/index.html`.
All four recorded source hashes remained unchanged through the final build.

**NOT RUN:** new OpenRouter inference or a full unit/contract suite. The owner previously deferred
those gates until integrated beta acceptance. Existing parent contract fixtures remain known red.

Remaining B05 completion work, in dependency order:

1. Coordinate ordinary-input continuity across `CognitiveControl.observe`, `admit`, `commit_private`, main private-only commits, private replay, and retained linear-episode handling. Preserve unknown-triage holds, accepted waits, semantic source guards, and Critical fencing.
2. Add an explicit durable displacement/reconsideration transition for active inquiries from older Critical epochs. The current implementation holds them; it cannot silently rebase their original grant.
3. Wire `FocusManagement` selected actions to the shared private owner and real child kernels. Preserve one-slot arbitration, accepted action identity, atomic result receipts, replay, and main acknowledgment. The admission API alone does not provide this integration.
4. Integrate scoped reflection/inspection evidence and graph felt-state emission. The graph currently excludes legacy inspection/reflection summaries explicitly.
5. Repair stale parent-contract fixtures, run the required real lifecycle gates, inspect resulting record packages, and then update acceptance status. A passing deterministic suite cannot establish companion quality.

## Pass: ordinary input leaves independent private work alone (2026-11-14)

The owner asked for a cheaper way to check architecture changes than a full live
run. This pass adds that harness and uses it to retire the first unfinished B05
integration item.

### Recorded scenario replay harness

`tests/replay/` records one real situation once and replays it for free.
A recording holds every model request and answer in call order, the starting
memory checkpoint, a manifest, and a typed outcome.
A replay serves those answers over a copy of the checkpoint while the real
classes, store, kernels, control fences, and child workflows run unchanged.

Determinism required three explicit decisions:

- `tests.replay.determinism.deterministic_identities` replaces the process
  identity source with a seeded counter. Invocation and commit ids reach prompts.
- `tests.replay.script.normalize_timing` replaces accumulated wall-clock
  seconds with a placeholder; the value is a receipt of how long the run took.
- The harness reruns itself under `PYTHONHASHSEED=0`, because a set of names
  otherwise prints in a different order in each process and that text reaches a
  validation message inside a prompt.

`ReplayDivergence` stops at the first changed request and writes both full
prompts beside the sample, so a change that alters a prompt is a loud result.
A scenario whose injection cancelled an invocation marks the sample
`NotReplayable`, because a cancelled run is a partial trace and its call
boundary races the model call.

Two checks cover the harness utilities: `tests/unit/test_replay_harness.py`
(13 checks). They prove the recording format, request identity, and divergence
report, not any scenario.

### The continuity defect and its correction

Recorded twice, two samples each, over one synthetic story concern:
`.runtime/replay/private-continuity-20261113T211349Z` (before) and
`.runtime/replay/private-continuity-20261113T211905Z` (after).
An unrelated ordinary message was delivered through the real input proxy and
classified by the real triage module while the second private invocation was
inside its model call.

| Sample | Before | After |
| --- | --- | --- |
| 1 | cycle 2 cancelled before any result | cycle 2 produced an accepted step |
| 2 | cycle 2 cancelled before any result | cycle 2 produced an accepted step (stalled progress) |

Both post-change samples replay exactly, in order, with no unasked calls.

Source changes:

- `CognitiveControl.observe` no longer advances the control generation or calls
  `cancel_private` on arrival. Arrival is not a classification.
- `CognitiveControl.accept_triage` advances the generation when the assessment
  is Critical, so a grant captured before that reconsideration can never commit,
  even after the required sources are appraised and the barrier clears.
- `CognitiveControl.commit_private` accepts independent private work under the
  current authority. An unclassified message holds acceptance cooperatively and
  cancellably instead of destroying the result. An accepted wait, a wake, a
  newer Critical epoch, or a changed source still rejects the work.

Three new actual-store boundary checks cover the new rules in
`tests/unit/workflows/cognitive_control/test_early_triage.py`: ordinary input
changes no generation and cancels nothing, unclassified input holds acceptance
until its assessment and then releases it, and a Critical assessment refuses
work from the older generation both before and after the barrier clears.

### Verification

Affected deterministic checks:

- `tests/unit/workflows/cognitive_control/test_early_triage.py`: **14 passed**
  (11 pre-existing plus the three new rules).
- `tests/unit/workflows/inner_thought` discovery: **162 passed**.
- `tests/unit/workflows/interaction` discovery: **7 passed**.
- `tests/unit/test_replay_harness.py`: **13 passed**.
- `tests.unit.workflows.interaction.test_boundaries`,
  `tests.unit.workflows.entity_context.test_owner_projection`,
  `tests.unit.workflows.entity_context.test_immediate_updates`, and
  `tests.unit.test_architecture`: **41 ran**, one pre-existing architecture
  failure from the harness's own `subprocess` use, fixed by moving that call to
  the `OperatingSystem` boundary; the final run reports `OK`.

Type checking: configured strict Pyright reports **0 errors, 0 warnings**.
A scoped strict scan of `tests/replay`, the new unit check, and the changed
control module reports **11 remaining diagnostics, all pre-existing**:
five in unchanged `workflows/cognitive_control/host.py` lines (`admit` and
`release_wakes`) and six `reportPrivateUsage` entries in unchanged early-triage
test lines.

Strict MkDocs passed in **212.98 seconds**, including the new
`docs/workflows/scenario-replay.md` page and the renamed
"What supersedes private work" section in `docs/subsystems/private-loop.md`.

### Recorded provider cost

The four recordings issued **82 provider requests** in total
(`request_count` in each `manifest.json`):
baseline 5 + 5, continuity before the change 6 + 8, continuity after 6 + 12.

Their dollar cost is **not measured**. The harness disables the prompt log so
that replay and recording draw the same identities, and the recording keeps the
completion text without the provider usage block. `request_count` is therefore
a lower bound on the requests, not a spend receipt. A future harness version
should retain the provider usage beside the recorded answer without drawing an
identity; until then, quote request counts and not dollars.

### Not run and remaining limits

**NOT RUN:** any new paid inference beyond the four scenario recordings, the
full unit/contract suite, the integrated live gates, and the final Jesse–Spamuel,
Jesse–Spamuel1–Spamuel alpha, and Spamuel–Spamuel–Minecraft scenarios.

**NOT RUN — full cognitive_control discovery:** it contains
`tests.unit.workflows.cognitive_control.test_wait`, which retries remote
embeddings until an endpoint answers; the same module also retries at HEAD
`60ea26f`, so it cannot run offline. The affected modules were run individually.

Pre-existing red, reproduced at HEAD `60ea26f` in a separate worktree and
unchanged by this pass:

- `tests/unit/workflows/cognitive_control/test_focus.py`: three errors, its
  `_Runtime` fixture has no `stop_event`.
- `tests/unit/workflows/main_cognition` discovery: **46 errors**, all from test
  fixtures whose `llm_complete` tool provides no context measurement. The same
  46 errors appear at HEAD.
- `tests/unit/workflows/public_speech` discovery: one error and one failure
  (`test_attribution`, `test_speech_review`), identical at HEAD.
- `tests.contract.inner_thought.test_inner_parent_contract` remains red.

Still unfinished in B05, in dependency order:

1. An explicit durable displacement/reconsideration transition for an active
   inquiry from an older Critical epoch. It is currently held for admission.
2. Wire `FocusManagement` selected actions to the shared private owner and the
   real child kernels, preserving one-slot arbitration, action receipts, replay,
   and main acknowledgment.
3. Integrate scoped reflection/inspection evidence and graph felt-state emission.
4. ~~Repair the stale parent-contract fixtures~~ — done: commit `eb829ca` gives
   each parent contract a retained pre-graph episode and a measuring completion
   fixture. All four pass. The remaining contract errors are in the main-cognition
   phase contracts, whose fake `llm_complete` also provides no measurement; that
   is the same pre-existing fixture staleness and is not part of B05.
5. Run the integrated live gates.

## Pending owner decision: displacing an inquiry from an older Critical epoch

Not implemented. This is a design choice about meaning, so it waits for the owner.

### The situation

`run_graph_opportunity` refuses an active inquiry whose retained
`grant.critical_epoch` is older than the current control epoch. It returns
"This inquiry predates accepted Critical reconsideration and requires fresh
admission." and changes nothing.

That is safe but not terminal. The inquiry stays `active`, and
`active_inquiry(view, observer_entity_id=...)` keeps returning it. Every later
private opportunity takes the same branch, so the one private slot stays
occupied by work that can never proceed. The character stops having private
thoughts after a Critical reconsideration until a restart.

The plan calls for "the actual displacement/resumption transition" and forbids a
silent rebase of the original grant. The two halves of that sentence pull apart:

- **Displacement** means the stale inquiry gives up the slot.
- **Resumption** means the question survives the displacement.
- **No silent rebase** means the new work cannot inherit the old grant, because
  the old grant was captured under an authority the character no longer has.

### Option A — abandon and free the slot

Stage one durable receipt: `status="abandoned"`,
`stop_reason="Critical reconsideration superseded this inquiry's authority"`,
`next_operation=""`. The nodes stay inspectable as history.

- Slot frees on the next timer tick; a fresh nomination runs normally.
- No new concept, no new field, no resumption path.
- The question is lost. A later episode can still read the old nodes as evidence,
  but nothing re-offers them.
- Honest but wasteful: a Critical message about something else ends an unrelated
  private inquiry.

### Option B — displace to paused, resume only through fresh attention

Stage a `paused` receipt with an explicit displacement reason and no automatic
deadline, and teach `parked_offer` to treat a displaced inquiry differently from
a parked one: it may be *offered* to shared attention, but its admission creates a
**new episode**, not a continuation.

Admission would re-capture the grant from the current view under the current
epoch, keep the old episode as `origin`/historical input, and start `sequence=0`.
The old nodes become evidence the new episode may cite, never authority it
inherits.

- The question survives; the authority does not. That matches the plan's wording.
- Costs a new persisted distinction (`displaced`), a second admission shape in
  `admit_opportunity`, and a rule that a displaced offer reuses the same
  one-slot arbitration as a parked offer.
- Risk: the model may treat the old nodes as corroboration of the new episode.
  That needs an explicit rendering rule ("these are earlier internal artifacts,
  not outside evidence").

### Option C — keep holding, add an operator escape

Leave the current behaviour and require a restart or an explicit operator action
to clear the slot.

- No new semantics.
- Leaves a known stall in normal operation, which is why it is listed as work.

### Recommendation

Option B, with the displaced inquiry admitted as a new episode under a fresh
grant. It is the only option that satisfies all three constraints the plan states,
and it reuses the existing parked-offer arbitration instead of adding a second
admission path. It is also the most expensive: it needs a persisted `displaced`
state, a second admission shape, and a rendering rule that separates "earlier
private artifacts" from outside evidence.

Option A is the cheap fallback if the owner prefers a smaller B05. It ends
unrelated private work after a Critical message, which is a real behaviour loss.

**Nothing was implemented for this item.** The next step is the owner's choice
between A and B.
