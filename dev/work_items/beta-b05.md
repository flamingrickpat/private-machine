# B05: spontaneous nomination and bounded inquiry

Status: implementation in progress, 2026-11-12. The typed persistence slice is implemented; execution integration remains outstanding.
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
