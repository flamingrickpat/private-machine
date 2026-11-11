# Beta capability audit: B03–B07

Status: source audit and proposed implementation boundaries, 2026-11-09.
Source baseline: `2b17cf84555d4f325a36e2b3812e8a24cd38d46d`.

This work record supports the [beta plan](beta.md).
It belongs beside that plan because it records implementation evidence and unresolved design choices.
The [milestone specification](../../docs/continuous-character-milestone.md) defines the required behavior.
This audit does not change that specification or establish behavioral acceptance.

**Evidence scope:** CODE means current source supports the statement.
PROPOSED means a boundary still needs a concrete design before implementation.
NOT VERIFIED means this audit did not exercise the behavior through the real runtime.
Concurrent B01/B02 edits are outside this baseline audit.

## Capability map

| Slice | CODE: existing mechanisms | Gap against the milestone | Next boundary |
| --- | --- | --- | --- |
| B03 ordinary async interaction | Separate bus modules, endpoint admission, source guards, durable public publication | Ordinary input still enters full cognition before dialogue dispatch. Broad cancellation can discard useful private work. | Separate inference priority from cognition ownership. Validate relevant dependencies at acceptance. |
| B04 persistent entity context | Identity records, narratives, beliefs, graph facts, entity workspace shards | No dedicated typed owner covers attributed current claims, freshness, availability, corrections, and observer-relative knowledge together. | Extend accepted records through one entity-model owner. Project current views without another authoritative narrative. |
| B05 thought emergence and inquiry | Triggered private loop, operation catalogue, bounded episodes, deliberate mental work, lineage | Production, selection, and elaboration remain coupled. Episodes lack explicit branch relationships and resumable inquiry contracts. | Separate candidate nomination from coordinated selection. Share one typed inquiry lifecycle. |
| B06 ToM and metacognition | Perspective operations, a ToM codelet, reflection experiments, self-inspection | No distinct action-selected ToM workflow appears in the assembly. Perspective-specific evidence and shared inquiry integration remain incomplete. | Add a specialized inquiry client. Preserve entity-owner and main-cognition acceptance boundaries. |
| B07 interval appraisal and sustained acceptance | Appraisal, intention successors, expectation checks, causal commits, simulation evidence | No explicit interval ledger ties each effect to governing historical intention revisions without duplicate appraisal. | Unify outcome attribution and expectation ownership. Then exercise integrated behavior. |

## B03: ordinary async interaction

Milestone references: sections 7.1–7.4, ARCH-02, ARCH-12, SC-02, SC-12, SC-15, and SC-17.

### Existing execution and contracts

- [`MainCogLoop.deliver` and `handle_batch`](../../workflows/main_cognition/host.py) reserve foreground inference through the full cycle and reply dispatch.
- [`InferenceAdmission`](../../pm/llm/concurrency.py) admits one HTTP completion at a time for each endpoint.
  Background calls wait while foreground reservations exist.
  An active call keeps its slot until it returns.
- [`CognitiveControl.observe`](../../workflows/cognitive_control/host.py) increments one generation for external input and requests private cancellation.
  `commit_private` rejects results from an older generation under the memory-store lock.
- [`PrivateThoughtLoop._commit_cycle`](../../workflows/inner_thought/host.py) checks captured record heads and revisions before acceptance.
- [`commit_public_reply` and `validate_public_context`](../../workflows/public_speech/publication.py) commit speech and exchange resolution before bus publication.
  Public validation already tolerates a revision change if the captured source fingerprint remains equal.
- [`PublicContext`](../../workflows/public_speech/public_context.py) carries source revisions, fingerprints, input IDs, and state context.
- [`MainCogLoop.publish_output`](../../workflows/main_cognition/host.py) registers captured dependencies and checks stimulus, focus, expectation, and selected-memory revisions.

### Concrete gaps

New ordinary messages still require main-cognition action selection before normal dialogue dispatch.
Separate threads therefore do not establish the required fast interaction path.
The endpoint gate also prevents background inference while a full foreground cycle retains its reservation.

The private generation fence invalidates work after any external input.
It does not ask whether that input changes the proposal's actual evidence or authority.
Under sustained conversation, useful private work can repeatedly lose acceptance.

Main cognition checks raw stimulus revisions.
Public exchange resolution changes lifecycle metadata on those records.
B03 must distinguish that resolution from an actual correction of the source evidence.
The existing public fingerprint mechanism supplies a concrete precedent.

### Proposed smallest change

The existing dialogue module can remain the speech owner.
An ordinary-input admission path can supply accepted guidance plus current exchange evidence after B01/B02.
Main cognition keeps authority over accepted intentions and deliberate state changes.

One typed dependency contract must distinguish source content, current record heads, authority revisions, and the evidence frontier.
The contract must also record the governing guidance version and any local suspension of that guidance.
Public commit, private commit, recovery, and replay must consume the same relevant semantics.

Inference admission needs explicit treatment of critical reconsideration, ordinary speech, ordinary full cognition, and private inquiry.
These are scheduling roles, not new owners of memory.
The detailed priority policy remains a design decision.
It must retain progress for useful background work and obey the critical barrier.

**NOT VERIFIED:** foreground latency, background progress, late-result rejection, topic changes, or restart during concurrent real inference.
SC-02, SC-12, SC-15, and SC-17 need real kernels and the configured OpenRouter endpoint.
The receipt must name the model, hardware, workload, first visible output time, and total response time.

## B04: persistent entity context

Milestone references: sections 12–13, ARCH-06–07, ARCH-10–11, SC-01, SC-03, SC-08, SC-16, and SC-19.

### Existing data and owners

| Source | Actual contract | Existing meaning |
| --- | --- | --- |
| [`memory_records.py`](../../pm/records/memory_records.py) | `Entity` | Identity anchor with aliases and entity class. |
| Same module | `Narrative` | Entity-linked interpretation with narrative type and last refinement tick. |
| [`cognitive_records.py`](../../pm/records/cognitive_records.py) | `BeliefRecord` | Entity references, confidence, support, counterevidence, activity, and supersession. |
| [`graph_records.py`](../../pm/records/graph_records.py) | `GraphNode`, `GraphEdge` | Consolidated graph facts with source memory and validity times. |
| [`workspace.py`](../../pm/records/workspace.py) | `ShardKey`, `WorkspaceShard`, `GlobalWorkspace` | Entity-specific record references, bounded content, and recent tails. |
| [`workspace_selection.py`](../../workflows/main_cognition/workspace_selection.py) | Main-cognition workspace selection | Selects conscious records rather than owning a durable entity model. |

`Entity` does not carry a changing situation model.
`BeliefRecord.about_entity_ids` identifies subjects, not the observer whose knowledge the claim represents.
Graph validity windows do not alone express freshness or the ability to receive and respond.
Existing narratives remain interpretations rather than a typed current-state authority.

### Proposed smallest change

The memory store remains the durable owner of accepted records.
A workflow-side entity-model owner validates proposals and stages accepted revisions through existing write batches.
It must reuse identity and evidence records rather than duplicate character homes or graph stores.

The proposed boundary needs named contracts for an attributed claim, an update proposal, and a compact entity view.
Each claim needs its subject, observer, epistemic kind, source IDs, source revisions, uncertainty, and applicable time.
Current applicability and historical truth must remain distinct.
Correction and expiry must preserve the original evidence.

The compact view must distinguish ability to receive, ability to respond, current activity, and contact preferences.
Unknown fields stay unknown for people, AI peers, and game agents.
Time can mark a claim stale or due for review.
Time cannot establish an unobserved replacement state.

Direct evidence must reach this owner independently of deliberate ToM selection.
B02 guidance then reads its accepted projection.
The richer model stays outside GWT, while relevant shards retain compact grounding and uncertainty.

The exact record extension and migration belong in the B04 proposal.
Before implementation, that proposal must identify how existing beliefs and narrative references resolve to their successors.
World time, UTC, time zones, and event order need explicit contracts.

**NOT VERIFIED:** accepted correction propagation, freshness behavior, observer isolation, or current availability across entity types.
Real SC-01, SC-03, SC-08, SC-16, and SC-19 runs must inspect accepted records and actual prompt context.

## B05: thought emergence and inquiry

Milestone references: sections 8–9 and 11, ARCH-03–05, ARCH-09–10, SC-05–06, SC-10–11, and SC-14.

### Existing execution and persistence

[`PrivateThoughtLoop`](../../workflows/inner_thought/host.py) freezes committed state and builds candidates before model calls.
[`evaluate_triggers`](../../workflows/inner_thought/triggers.py) supplies deterministic opportunities such as unresolved commitments, surprise, idle time, and repeated concern.
An empty trigger set produces no thought call.

[`run_cycle`](../../workflows/inner_thought/pipeline.py) runs trigger selection, one thought operation, integration, and optional reflection.
It validates citation IDs, reality labels, continuations, and expectation proposals.
[`Operation`](../../workflows/inner_thought/operations.py) distinguishes generative and discriminatory modes.
The catalogue already covers recall, projection, counterfactuals, feasibility, curiosity, rumination, and inspection.

[`InnerRequest`, `ThoughtResult`, and `InnerReport`](../../workflows/inner_thought/schemas.py) carry evidence, interpretation, uncertainty, alternatives, and proposed continuation.
`InnerRequest` also supplies accepted mental state, goals, recent thoughts, and channels.
Some fields still use anonymous mappings, including episode state and goal records.
Those fields need concrete contracts during any relevant revision.

`PrivateThoughtLoop._next_episode` persists identity, sequence, elapsed limits, progress, parking, and closure in thought metadata.
`_resolve_episode` waits for main appraisal before the next continuation.
[`RecordLineage`](../../pm/records/lineage.py) preserves parents, source events, intention IDs, operation, sequence, and reality.
Accepted thoughts enter the inner GWT slot and emit `thought.created`.
Main cognition then appraises the result.

[`FocusManagement`](../../workflows/focus_management/host.py) supplies a separate action-selected mental-work session.
[`MentalWorkRequest`](../../workflows/focus_management/schemas.py) includes the selected instruction, workspace, character, experience, affect, and timeout.
The host commits a private `MentalPlan` proposal and action receipt.
It does not directly change goals or affect.

### Concrete gaps

Current candidates identify records worth thought, rather than diverse generated candidate thoughts with separate selection receipts.
The private workflow selects and elaborates within the same cycle.
The producer therefore lacks the milestone's explicit competition boundary with broader attention.

Episode continuation requires a main-appraisal round trip.
The milestone permits several inquiry branches without that round trip for every branch.
Current alternatives contain text, disposition, and reason.
They do not define durable branch identity, joins, premise dependencies, or revisits.

`concrete_progress` accepts a distinct thought text as an artifact.
That condition cannot by itself distinguish a meaningful result from a paraphrase.
It needs a stronger result contract before sustained inquiry acceptance.

### Proposed smallest change

Candidate emergence remains a logical workflow distinct from deliberate inquiry.
Its typed nominations enter coordinated attention before elaboration begins.
A nomination receipt records selection, deferral, displacement, or dismissal separately from later conscious thought.

One inquiry contract can serve spontaneous and selected mental work.
It needs stable episode and branch identities, typed relationships, separate judgments, source dependencies, and explicit lifecycle outcomes.
An unresolved question or useful reformulation can count as an outcome without a final answer.
Progress must name a substantive difference in evidence, alternatives, assumptions, or decisions.

The inquiry host owns bounded continuation and checkpoints.
Main cognition still owns accepted intention changes and consequential appraisal.
Public information requests remain proposals for action selection.
Tool acquisition needs attempted, completed, failed, and observed-evidence records.

Existing need and mental-state consumers need an audit before any new understanding need.
The strength of a need can affect selection without changing evidential support.
The exact mapping from current axes to exploration remains an open design choice.

**NOT VERIFIED:** diverse emergence, branch selection, substantive progress, suspended inquiry resumption, or useful information acquisition through real inference.
Existing test files describe local cases but do not establish these integrated capabilities.

## B06: ToM and metacognition

Milestone references: sections 10 and 14–15, ARCH-08–10, SC-03, SC-07–11, and SC-16.

### Existing mechanisms and mismatch

[`ToMVignetteSimulator`](../../workflows/main_cognition/codelets.yaml) is a codelet inside main cognition.
Its prompt asks for a reading with conviction and discourages hedging.
That prompt creates an epistemic risk for uncertain perspective estimates.
Source inspection does not prove that a runtime result becomes an accepted fact.

[`perspective_shift` and `rumination_notice`](../../workflows/inner_thought/operations.py) already name perspective and metacognitive operations.
[`evaluate_triggers`](../../workflows/inner_thought/triggers.py) detects repeated source IDs among recent thoughts.
This is a useful foundation for concern repetition, not proof of full metacognitive appraisal.

[`ReflectionDecision`, `ExperimentPlan`, `Hypothesis`, and `ExperimentMetadata`](../../workflows/reflection/schemas.py) define typed private experiments.
[`stage_decision`](../../workflows/reflection/experiments.py) stages their lifecycle changes through accepted record owners.
The contracts retain competing explanations, predicted effects, observations, source revisions, reversal state, and unresolved outcomes.
[`self_model.py`](../../pm/introspection/self_model.py) records revisable capability beliefs and stale-source information.

[`_cognition_workflow_options`](../../workflows/companion.py) offers deliberate mental work through `focus_management`.
The baseline assembly contains no distinct ToM manifest, host registration, or selectable ToM workflow.
A perspective codelet does not satisfy the required distinct workflow boundary.

### Proposed smallest change

A ToM workflow becomes a specialized client of B05 inquiry machinery.
Its request names the target entity, observer, question, selected action, permitted evidence, and current model revisions.
Its report preserves alternative motives, knowledge differences, predictions, counterevidence, and unresolved questions.
These remain attributed estimates rather than access to another mind.

Entity revisions pass through B04 acceptance.
Consequential findings return to main cognition for appraisal.
Public questions require separate action selection and existing speech publication.
ToM inquiry completion cannot itself authorize contact.

Metacognitive observations can enter the same proposal path as other private results.
Existing reflection experiments remain useful for predictions and reversible interventions.
Their lifecycle must share inquiry lineage and source semantics rather than become another thought store.
Recursion limits and attention compete with other work.
No fixed calm or agreeable result follows from reflection.

**NOT VERIFIED:** action-selected ToM, observer-relative prediction, contradictory social evidence, bounded metacognition, or absence of perspective leakage.
SC-03 and SC-07–11 require real model output plus accepted-record inspection.

## B07: interval appraisal and sustained acceptance

Milestone references: section 6, section 16, ARCH-01, ARCH-13, SC-04, SC-13, and integrated sections 18–20.

### Intention and outcome foundations

[`Intention`](../../pm/records/cognitive_records.py) has activity, status, salience, fulfillment, parent links, and a tick-based timeout.
[`GoalChange`](../../workflows/main_cognition/schemas.py) supplies a stable key, reason, priority, and change kind.
[`apply_goal_changes`](../../workflows/main_cognition/cycle_commits.py) stages lifecycle changes with appraisal and source evidence.
[`goals.run`](../../workflows/main_cognition/phases/goals.py) treats private goal proposals as suggestions for main acceptance.

The present goal contract does not explicitly represent semantic horizons, suspension and resumption, activation conditions, conflicts, or completion evidence.
An inactive deferred intention is absent from `active_intentions`.
The current change kinds include no explicit resume transition.
Beta must preserve deferred commitments without treating a tick timeout as their semantic horizon.

[`ExpectationRecord`](../../pm/records/cognitive_records.py) keeps anticipation, goals, originating action, evidence, observations, resolution tick, and fulfillment.
[`main_cognition.expectations`](../../workflows/main_cognition/expectations.py) retrieves relevant open expectations and computes fulfillment effects from current stimuli.
[`inner_thought.expectations`](../../workflows/inner_thought/expectations.py) supplies a separate open/resolve lifecycle for private expectations.
The main host filters its expectation inputs by origin.
That split needs explicit reconciliation before interval acceptance.

[`apply_appraisal`](../../workflows/main_cognition/cycle_commits.py) records appraisal evidence and affect deltas.
[`RecordLineage`](../../pm/records/lineage.py) links records to intention IDs, but lacks an explicit governing intention-revision field.
Generic commit identity prevents repeated acceptance of one commit.
It does not establish idempotence for the same outcome reviewed in a later, distinct appraisal.

### Proposed smallest change

The interval request needs the actual event range, prior evaluation frontier, governing intention revisions, relevant expectation revisions, and delivered actions.
The evaluation report needs attempted moves, observed outcomes, unresolved conditions, proposed direction changes, and appraisal effects.
Historical intention revisions must govern their actual interval, even after a later correction or replacement.

A durable evaluation identity must prevent repeated effects for the same outcome and governing revision.
Corrections need an explicit new evaluation relationship rather than a second reward or punishment.
Main cognition owns acceptance of deliberate intention changes and appraisal effects.
The entity and belief owners retain their existing state boundaries.

The intention extension must cover immediate, episode, and ongoing direction without mandatory goals for every horizon on every pass.
Its lifecycle needs explicit suspension, resumption, completion, abandonment, and supersession semantics.
Avoidance motives and firm constraints need distinct representation.
The B07 proposal must identify how historical goal keys and successor records migrate without invented past intent.

### Required evidence and limits

The [simulation](../../simulation/engine.py) already captures attempt evidence.
The [analysis module](../../simulation/analysis.py) supplies cited model judgments.
These tools can support short comparisons before sustained runs.
Model ratings remain separate from owner quality acceptance.

The interrupted `20261104-eventful-month` remains a partial baseline.
It is not evidence of a completed month or sustained beta acceptance.
The cleanup audit preserves its current checkpoint, source, transcripts, and failure evidence.

**NOT VERIFIED:** historical attribution, nonduplicate outcome effects, reconciled expectation ownership, or sustained integrated beta behavior.
SC-04 and SC-13 need real changes, restart, and repeated review of the same observed event.
The remaining milestone scenarios need inspectable runs after their owning slices exist.
Source inspection alone establishes none of those gates.

## Verification status of this audit

This audit used source reads and symbol searches against the stated baseline.
It changed no product code, checkpoint, simulation state, or dependency.
No LLM call, local model, or live companion run occurred.
New behavioral tests are NOT RUN because this task covers source-based planning.

The local target check found all 40 relative links.
The simple-English review found no selected banned patterns or prose sentences beyond the 25-word limit.
The root task owns the final strict MkDocs build and rendered-page inspection.
This work record stays outside the conceptual site narrative.
