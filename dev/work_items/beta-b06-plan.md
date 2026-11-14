# Beta B06 — implementation plan: theory of mind and metacognition

Written 2026-11-14 on branch `beta` at `872da60`. This file changes no product code.

This file does the work of "Pass 0" in `beta-handoff-b06-20261114.md`: the read-only audit and the
architecture proposal. It then lists small implementation steps for a coding agent.

**Status: proposed. Nothing in sections 5 and 6 is accepted until the owner answers section 2.**

How to use this file:

- **Owner:** read sections 1 and 2. Answer section 2. That is all you need.
- **Coding agent:** read `AGENTS.md`, then sections 3 to 9. Do not start section 6 before the owner's
  answer is written into section 2.

The handoff stays the authority for guard rails, known red suites, and standing rules.
Do not write `beta-b06-proposal.md`. This file replaces it.

---

## 1. What B06 adds, in plain words

B06 gives the companion two abilities. Both reuse machinery that already exists.

**1. Think about another mind on purpose (theory of mind, "ToM").**
Illustrative example, not recorded evidence:

> Evening. Spamuel wants to greet Jesse tomorrow morning, but does not know when a message is welcome.
> Spamuel decides: "I want to work out when Jesse is usually up."
> Over a few private thinking steps it reads what Jesse really said: "up at six on workdays",
> "weekends don't count", "nothing before coffee". It weighs alternatives and exceptions.
> It ends with a finding: workdays probably after 07:00, weekends unknown, no messages before coffee.
> Main cognition takes the finding in. Spamuel's picture of Jesse gets three new entries.
> Each entry is marked as an estimate or as a quoted statement, and links to the message it came from.
> No message is sent. Whether to greet Jesse tomorrow is a separate, later decision.

**2. Notice its own thinking (metacognition).**
Illustrative example, not recorded evidence:

> Jesse answers a long message with "Yeah. Maybe." Spamuel's private thoughts return to it three
> times and find nothing new. The next private step receives the facts: "three of your last four
> thoughts return to this one message, none found new evidence."
> Spamuel notices: "Why do I keep coming back to this?"
> That noticing reaches main cognition, once. Main cognition then chooses as the character would:
> stop for now, attend to something else, think about why it matters, think about Jesse's side,
> or (a separate decision) ask Jesse. No outcome is forced, and nothing guarantees calm.

What exists today and what B06 adds:

| Piece | Today | B06 adds |
| --- | --- | --- |
| Choose to think something through | Exists: the `focus_management` action | A second menu entry for thinking about another entity |
| Private multi-step thinking with evidence rules | Exists: the inquiry graph (B05) | The subject's accepted model and its original messages as reading material |
| A model of another entity with reports, estimates, uncertainty | Exists: entity context (B04) | Private thinking can propose entries, labelled `tom` |
| Noticing dwelling | A trigger exists. The noticing step gets no facts. Main cognition hears only if the thinking also stops. | Facts for the noticing step. Main cognition always hears, once per stretch. |
| Hard stop of stalled thinking | Exists | Unchanged |

No new store, no new inquiry system, no new agent, no message sent by any of this.

---

## 2. Owner decisions

Each row has a recommended answer. The plan in section 6 assumes the recommended answers.

| # | Question | Recommended answer | What it means for you | The other option |
| --- | --- | --- | --- | --- |
| D1 | How does the companion start "thinking about someone"? | Its own entry in the action menu, named `theory_of_mind`, beside `focus_management`. The same module and the same private slot run it. No new `ActionType`. | The action model sees a plainly described choice. The review browser shows the action by name. | Hide it as an option inside `focus_management`. Less code. Harder for the model to pick: in the B05 gate, main cognition picked `focus_management` in 0 of 3 attempts. |
| D2 | When does a new belief about Jesse enter Spamuel's picture of Jesse? | Private thinking only **proposes**. Main cognition writes the entry when it next takes in the finding, through the existing `EntityContextOwner`, with `producer="tom"`. | A ToM result shapes behaviour one main cycle later, not instantly. Every entry has a receipt with its outcome. | The private loop writes directly. Not recommended: see "Why D2" below. |
| D3 | What does "reconsidering its own thinking" mean? | Two roles. The **private loop notices**, with recorded facts about its own activity. **Main cognition appraises** the noticing and owns every consequence. Old thoughts are never rewritten. | You can see the noticing, the facts it was based on, and what main cognition did next. | Only one of the two roles. Then either main never hears, or the private loop never gets facts. |
| D4 | Keep the hard stop of stalled thinking? | **Yes, keep it.** This is your stated preference from 2026-11-14. No code. | No behaviour change. | A graded "wish to stop". A contract change with its own design. Not in this plan. |
| D5 | What can a ToM step read and cite? | Only what this companion itself has, in the same reality: the records the action named, its own memory search, and the original messages behind what it already believes about that entity. | No leak from another entity's conversation, another character's model, or an imagined story. | Wider or narrower grants. Each needs a new leak review. |
| D6 | First recorded ToM scenario? | SC-09, the morning routine. Synthetic Jesse lines: a workday pattern, a weekend exception, a contact preference. Metacognition evidence: re-record the existing `private-dwelling` scenario. | Two things you can open and rate in the review browser. | Another bounded question. Name it and the plan adjusts one step. |

Three smaller choices this plan made. You can veto each one:

| # | Choice | Reason | Effect of a veto |
| --- | --- | --- | --- |
| V1 | The ToM subject is the entity that this part of the story is about (the action's partner). Naming a third entity is later work. | The action model sees names, not record IDs. A named subject needs extra validation. | Add one step before 2.4: a validated `arguments.subject` name. |
| V2 | "Notice once": after a noticing about a source, the `repeated_concern` trigger stays quiet for that source while the noticing is among the last 8 private thoughts. This adjusts the accepted B05 rule. | Without it, noticing dwelling can itself count as dwelling. That is the recursion SC-07 forbids. | Keep the B05 rule unchanged. Then the recursion risk stays open and is recorded. |
| V3 | Main cognition reads the noticing in words. The counters stay on the record for review. No main-cognition prompt changes. | Smallest safe change. | Add the counters to the main prompts. Larger change, new live checks. |

Limits of ToM writing that follow from D2 and D5. They are deliberate:

- ToM can add an **estimate** (`inference`), a **possibility** (`hypothetical`), or a **quoted statement** it
  found (`report`, with the exact quote).
- ToM can **challenge** an old entry. It can **remove** one only when it cites a quoted statement.
- ToM cannot write `observation`. Thinking is not observing.
- ToM cannot change the "can receive / can respond" estimate or the current-situation layer.
  Ordinary message handling keeps that job.
- A repeated estimate with no new source is refused. Dwelling is not corroboration.

**Why D2.** Two reasons, both checked in the source.
First, goals and focus advice from private thinking already work this way: propose, then main accepts.
Second, a technical hazard. Every private graph commit is a crash-recovery root, and the store refuses a
recovery root that depends on an unaccepted commit (`pm/memory/store.py`, "Recovery root depends on
unaccepted transactions"). An entity entry written from the private loop's derived commit would be the
newest entry of that family and unaccepted. The next incoming message updates the same family inside a
recovery root, and that update would fail.

**How to answer.** Write `accept all`, or list your changes, for example `D1: other option, V2: veto`.

Owner answer (fill in before implementation):

> **accept all** (2026-11-14). D1–D6 take the recommended answers; V1–V3 stand as written. Implementation
> may start at section 6.

---

## 3. Rules for the coding agent

`AGENTS.md` is mandatory. The rules below are the ones this work breaks most easily.

1. **One step, one commit.** Do the steps in section 6 in order. Do not start a step before the checks of
   the previous step pass. Do not push.
2. **Touch only the files a step names.** If you think another file must change, stop and ask.
3. **The golden rule.** A private step that is neither ToM nor a noticing must send the same prompt bytes
   and store the same record bytes as before B06. The replay check in section 7 proves it.
   - Do not add, remove, or rename a field of `InquiryStep`, `InnerRequest`, `InquiryState`,
     `InquiryOrigin`, or `InquiryGrant`.
   - A new optional field on a stored or serialized model uses `Field(default=None, exclude_if=_absent)`.
     `GraphThoughtRequest.reflection_evidence` shows the pattern.
4. **Real stores, no mocks.** Unit checks use a real `MemoryStore`, a real `BusSystem`, and real owners.
   `tests/unit/workflows/focus_management/test_deliberate_owner.py` shows the setup. No model runs there.
5. **Typed contracts.** No `dict[str, Any]` domain data, no `.get(field, default)` on a required field,
   no `getattr` probing, no `# type: ignore`.
6. **Do not relax a validator** to make a model answer pass. Keep the failed sample and report it.
7. **Do not fix the known red suites** (section 7) in a B06 commit. Do not run the complete suites.
8. **OpenRouter only.** Never start a local model. Only Pass 4 spends money.
9. **Verify names first.** Every symbol in this file was read at `872da60`. Open the file and confirm the
   symbol before you edit. Report any difference in the work record.
10. **Record each pass** in `dev/work_items/beta-b06.md` with the report format of section 8.

Stop and ask the owner when:

- a step needs a contract change this file does not name;
- any path would contact a person or change an entity model outside `EntityContextOwner.stage_update`;
- a change would alter the hard-stop behaviour (`pause`, `stalled_steps`, `episode_steps`, `episode_seconds`);
- a replay diverges at a step that is neither ToM nor a noticing;
- a live gate fails for a reason that is not clearly one model-quality sample;
- the same symptom needs a third patch.

Not in B06. Do not implement:

- the four gamma G01 items in `dev/work_items/gamma.md`;
- a graded stop bias (D4);
- a named third-party ToM subject (V1), unless the owner vetoes V1;
- ToM entries in the `situation` layer or with an `interaction` estimate;
- beliefs about beliefs about beliefs;
- changes to `GwtTurnRecord`, the recovery-root rule, or `stage_decision`;
- the final Jesse–Spamuel, three-party, and Minecraft scenarios. They come after B07.

---

## 4. Code map (read at `872da60`)

| File | Role today | Symbols this plan uses |
| --- | --- | --- |
| `workflows/companion.py` | Builds the action menu and the modules | `_cognition_workflow_options` |
| `workflows/main_cognition/phases/prompts/route.yaml` | Rules that turn a decision into an action | the `focus_management` rule |
| `workflows/main_cognition/host.py` | Main cycle. Publishes `action.request` with `target=dispatch.workflow_name`. Stages entity proposals at its final commit. | `MainCogLoop.publish_output`, `prepare_input` |
| `workflows/main_cognition/entity_updates.py` | Turns main belief updates into entity proposals | `prepare_entity_beliefs`, `MainEntityUpdates` |
| `workflows/focus_management/host.py` | Owns a selected mental action: admission, lifecycle, delivery. No model. | `FocusManagement`, `requested_action`, `RequestedMentalAction`, `_admit_requested` |
| `workflows/inner_thought/schemas.py` | Typed private-loop contracts | `DeliberateAdmission`, `IntegrationResult`, `InquiryStep` (do not change) |
| `workflows/inner_thought/inquiry.py` | Graph staging | `stage_deliberate_inquiry`, `deliberate_admission`, `outside_source` |
| `workflows/inner_thought/graph_host.py` | Durable acceptance of graph steps | `PrivateInquiryOwner.accept_step`, `accepted_source_ids`, `pending_findings` |
| `workflows/inner_thought/graph_execution.py` | Runs one private opportunity | `run_graph_opportunity`, `GraphThoughtRequest`, `graph_thought_instruction`, `with_origin_evidence`, `REFLECTION_OPERATIONS` |
| `workflows/inner_thought/graph_outcomes.py` | Keeps the integration on the finding | `capture_graph_outcome`, `attach_graph_outcome` |
| `workflows/inner_thought/pipeline.py` | Integration validation | `validate_integration`, `IntegrationError` |
| `workflows/inner_thought/stages/thought/steps.py` | Thought stage prompt | the `stage_kind == "inquiry_operation"` branch |
| `workflows/inner_thought/stages/integrate/` | Integrate stage prompt | `IntegrationPrompts`, `build_messages`, `focus_advice_template` |
| `workflows/inner_thought/triggers.py` | Deterministic triggers | `_repeated_source`, `RUMINATION_REPEATS` |
| `workflows/inner_thought/request_views.py` | Views of recent thoughts | `RecentThoughtView`, `InquiryThoughtView` |
| `workflows/entity_context/schemas.py` | Entity drafts, proposals, views | `EntityClaimDraft`, `EntityExtractionDraft`, `EntityProducer` (already has `"tom"`) |
| `workflows/entity_context/owner.py` | The only acceptance boundary for entity entries | `EntityContextOwner.stage_update`, `active_claims` |
| `workflows/entity_context/projection.py` | Compact entity view and its text | `project_entity_context`, `render_entity_context` |
| `workflows/input_triage/entity_updates.py` | Turns model drafts into proposals | `capture_entity_extraction`, `hydrate_entity_extraction` |
| `tests/replay/` | Recorded scenarios | `scenarios.py::_dispatch_mental_action`, `_run_deliberate`, `SCENARIOS`; `situations.py::private_dwelling` |

Audit result against the handoff findings F1–F11: all eleven hold at `872da60`. Additions found:

- **A1.** `action.request` is routed by `target=dispatch.workflow_name`, and `BusModule.accepts` takes only
  `target in {None, module_id}`. A new menu name needs `FocusManagement.accepts` to take it.
- **A2.** `EntityContextOwner._valid_claim` already refuses an inference that repeats a claim with higher
  confidence and no new source. Milestone section 15.3 has a foundation in the owner.
- **A3.** A retraction needs report or observation evidence in `correction_evidence`. Thinking alone cannot
  delete an entry.
- **A4.** A noticing step with `outcome == "continue"` is not a finding, so main cognition does not hear it.
- **A5.** The private loop reads no entity context at all today.
- **A6.** The recovery-root hazard described under "Why D2" in section 2.
- **A7.** Record metadata is fingerprinted, and grant fingerprints are part of later prompts. A new stored
  field that serializes as `null` changes old recordings. Hence rule 3.

---

## 5. Target design

### 5.1 Flow of one ToM action

1. Main cognition decides and routes: `workflow_name="theory_of_mind"`, `action_type="ToolCall"`,
   `arguments.instruction` = one bounded question.
2. `FocusManagement` accepts the request. The subject is `ActionRequest.partner_entity_id`.
3. `DeliberateAdmission` carries `subject_entity_id`. The episode enters the private slot exactly like
   other deliberate work: it parks spontaneous work, with a receipt.
4. Each private tick runs one operation. A ToM step also receives **perspective evidence**: the accepted
   view of the subject, and the original messages behind it as citable candidates.
5. At the outcome, the integrate stage is also asked for an optional **perspective revision**: at most
   four `EntityClaimDraft` rows.
6. The finding keeps the revision in its metadata. `FocusManagement` delivers the finding as
   `thought.created` under the action's correlation. Nothing changes here.
7. Main cognition appraises the finding. At its final commit it hydrates the drafts into
   `EntityContextProposal` rows with `producer="tom"` and stages them through
   `EntityContextOwner.stage_update`. Each outcome (`staged`, `deferred`, `rejected`) is recorded on the
   decision record, as for `producer="main"` today.

### 5.2 Flow of one noticing

1. Stalled private steps return to one source. `repeated_concern` fires (existing).
2. Nomination and attention can admit `rumination_notice` or `attention_drift_notice` (existing).
3. New: that step receives **activity evidence**, built without a model from the recent private steps.
4. New: the step record keeps those facts in metadata, and the record is always delivered to main
   cognition, also when the private thinking continues.
5. Main cognition appraises it like any private finding (existing). Its choices are the existing actions.
6. New: "notice once". The trigger does not fire again for that source while the noticing is in the window.

### 5.3 New and changed types (proposed)

```python
# workflows/inner_thought/schemas.py — one new field on an existing model
class DeliberateAdmission(InquiryContract):
    ...
    subject_entity_id: int | None = Field(default=None, gt=0, exclude_if=_absent,
        description="Identity anchor of the other entity this inquiry models. Null for deliberate work "
                    "that is not a perspective inquiry, and for every admission stored before B06.")

class IntegrationResult(BaseModel):
    ...
    perspective_revision: EntityExtractionDraft | None = Field(default=None, exclude_if=_absent,
        description="Proposed entries for the subject's entity model. Only requested at the outcome of a "
                    "perspective inquiry; null otherwise.")
```

```python
# workflows/entity_context/schemas.py — the stored draft that main cognition reads
class PerspectiveRevision(BaseModel):
    """Entity-model entries that a perspective inquiry proposed. Main cognition stages them."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    subject_entity_id: int = Field(gt=0, description="Identity anchor of the entity the inquiry modelled.")
    claims: tuple[EntityClaimDraft, ...] = Field(min_length=1, max_length=4,
        description="Model drafts. The host supplies fingerprints, claim IDs, and the family parent later.")
```

```python
# workflows/inner_thought/perspective.py — new file, the ToM specialization of the shared inquiry
class PerspectiveContext(InquiryContract):
    """What one perspective-inquiry step reads about its subject."""
    subject_entity_id: int
    subject_name: str
    subject_class: str                      # EntityClass value, so a non-human subject is visible
    entity_view: EntityContextView          # the accepted view, captured for this step
    evidence_source_ids: tuple[int, ...]    # original sources behind that view: current, accepted, same scope

class ExistingClaimRef(InquiryContract):
    claim_id: str
    claim_key: str
    statement: str
    source_ids: tuple[int, ...]

class PerspectiveRevisionRequest(InquiryContract):
    """Ask the integrate stage for optional entity-model entries at the end of a perspective inquiry."""
    subject_entity_id: int
    subject_name: str
    citable_source_ids: tuple[int, ...]     # original outside sources only; never a private thought
    existing_claims: tuple[ExistingClaimRef, ...]

def capture_perspective_context(view: Snapshot, *, subject_entity_id: int, grant: InquiryGrant,
                                accepted_ids: frozenset[int], as_of_utc: datetime) -> PerspectiveContext: ...
def render_perspective_context(context: PerspectiveContext) -> str: ...
def perspective_rule(evidence: str, *, companion_name: str, subject_name: str) -> str: ...
def revision_request(view: Snapshot, context: PerspectiveContext, grant: InquiryGrant) -> PerspectiveRevisionRequest: ...
def validate_perspective_revision(draft: EntityExtractionDraft, request: PerspectiveRevisionRequest,
                                  source_texts: Mapping[int, str]) -> None: ...   # raises IntegrationError
```

```python
# workflows/inner_thought/request_views.py — facts about the companion's own recent thinking
class RepeatedSourceActivity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: int = Field(description="A source that several recent private steps cite.")
    step_ids: tuple[int, ...] = Field(description="Recent private steps that cite it, newest first.")
    stalled_step_ids: tuple[int, ...] = Field(description="Those of them that accepted no progress.")
    episode_ids: tuple[str, ...] = Field(description="Inquiry episodes those steps belong to.")

class ThoughtActivityView(BaseModel):
    """Recorded facts about recent private thinking. No judgment and no cause."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    window_step_ids: tuple[int, ...]
    repeated: tuple[RepeatedSourceActivity, ...]
    stalled_step_count: int = Field(ge=0)
    new_evidence_step_count: int = Field(ge=0)

# workflows/inner_thought/metacognition.py — new file
METACOGNITION_OPERATIONS: frozenset[str] = frozenset({"rumination_notice", "attention_drift_notice"})
def thought_activity(recent: Sequence[RecentThoughtView]) -> ThoughtActivityView: ...
def render_thought_activity(activity: ThoughtActivityView) -> str: ...
def activity_rule(evidence: str) -> str: ...
```

```python
# workflows/inner_thought/graph_execution.py — two optional prompt fields, absent for every other step
class GraphThoughtRequest(BaseModel):
    ...
    perspective_evidence: str | None = Field(default=None, exclude_if=_absent, description="...")
    activity_evidence: str | None = Field(default=None, exclude_if=_absent, description="...")
```

```python
# workflows/main_cognition/entity_updates.py
def prepare_perspective_revisions(owner: EntityContextOwner, view: Snapshot, stimulus_ids: Sequence[int], *,
                                  observer_entity_id: int) -> MainEntityUpdates: ...
```

Why two new files: `perspective.py` and `metacognition.py` each own one specialization of the shared
inquiry (milestone 14.2 calls these "specialized inquiry context"). Their conceptual parent is
`workflows/inner_thought/host.py`. `graph_execution.py` stays the coordinator and only calls them.

### 5.4 Ownership and persistence

| State | Who proposes | Who accepts | Where it is stored |
| --- | --- | --- | --- |
| ToM subject of an action | Main cognition (the action's partner) | `FocusManagement` at admission | `DeliberateAdmission` on the entry receipt |
| Perspective evidence | Host, no model | — (prompt material) | Not stored. The cited sources are in the step's grant. |
| Perspective revision | Integrate stage (model) | Validated in the private loop; **accepted by main cognition through `EntityContextOwner`** | Draft: finding metadata `perspective_revision`. Accepted: entity receipt, `producer="tom"`. Outcome: decision metadata `entity_update_outcomes`. |
| Activity facts | Host, no model | — | Noticing record metadata `thought_activity` |
| Consequence of a noticing | Main cognition | Main cognition | The main turn that claims the noticing, and its action |

No independent commit is added. No external side effect exists in B06. Nothing is retried except the
existing model repair turns. A stale or invalid stored draft becomes a recorded rejection, not an error.

### 5.5 Behaviour preserved and changed

Preserved: every prompt and record of a private step that is neither ToM nor a noticing; the hard stop;
deliberate admission, parking, lifecycle, and delivery; entity acceptance rules; main appraisal of findings.

Changed on purpose:

- A new menu entry and one new route rule (main-cognition prompt text grows by a few lines).
- A noticing step reads activity evidence, keeps it, and is always delivered to main cognition.
- `_repeated_source` gets the "notice once" rule (V2).
- Recordings: `private-dwelling` is expected to diverge after Pass 3 and is re-recorded in Pass 4.
  The other three recordings must replay exactly through all passes.

---

## 6. Steps

Each step lists: **Edit**, **Test**, **Check**, **Commit**. "Check" always includes the section 7 baseline.

### Pass 1 — a ToM action can enter the private slot

Nothing is selectable by main cognition until Step 2.4.

#### Step 1.1 — subject on the admission

Edit:

- `workflows/inner_thought/schemas.py`: add `DeliberateAdmission.subject_entity_id` (section 5.3).
  Put a module-level `_absent` helper here, as in `graph_execution.py`.
- `workflows/inner_thought/inquiry.py::stage_deliberate_inquiry`: when `admission.subject_entity_id` is
  set, require that the record is an `Entity` and is not `grant.observer_entity_id`. Raise `ValueError`
  otherwise.
- `workflows/focus_management/host.py`:
  - add `PERSPECTIVE_WORKFLOW = "theory_of_mind"` and
    `MENTAL_WORKFLOWS: frozenset[str] = frozenset({"focus_management", PERSPECTIVE_WORKFLOW})`;
  - override `FocusManagement.accepts`: also accept a data envelope whose topic is in `DATA_TOPICS` and
    whose target is in `MENTAL_WORKFLOWS`; otherwise call `super().accepts(envelope)`;
  - add `subject_entity_id: int | None` to `RequestedMentalAction`;
  - in `requested_action`: when `request.workflow_name == PERSPECTIVE_WORKFLOW`, the subject is
    `request.partner_entity_id`. A missing partner raises `ValueError`. Any other workflow gives `None`;
  - in `_admit_requested`: pass the subject into `DeliberateAdmission`.
  - update the module docstring: one short paragraph on the second catalogue name.

Test (`tests/unit/workflows/focus_management/test_deliberate_owner.py`, real store and bus):

- a request with `workflow_name="theory_of_mind"` and target `theory_of_mind` is admitted, and the entry
  receipt's `deliberate.subject_entity_id` is the partner;
- a `focus_management` request stores no `subject_entity_id` key in the receipt metadata;
- a subject that is the observer, or is not an `Entity`, raises.

Check: section 7 baseline, including the four replays.

Commit: `feat(focus-management): carry a perspective subject into deliberate admission`

#### Step 1.2 — perspective evidence for the thought step

Edit:

- new `workflows/inner_thought/perspective.py` with `PerspectiveContext`, `capture_perspective_context`,
  `render_perspective_context`, and `perspective_rule`. Start the file with a module docstring that points
  to `workflows/inner_thought/host.py` as the overview.
  - `capture_perspective_context` builds `EntityContextKey(subject, observer=grant.observer_entity_id,
    scope=grant.scope)`, reads the keys of `active_claims`, and calls `project_entity_context` with
    `token_budget=WORKSPACE_TAIL_TOKENS` (`pm.constants`).
  - `evidence_source_ids` are the record IDs of `entity_view.source_dependencies` that are in
    `view.all_knoxels`, in `accepted_ids`, and in `grant.scope`.
  - `render_perspective_context` writes one subject line (name, entity ID, class) and then
    `render_entity_context(entity_view)`.
- `workflows/inner_thought/graph_execution.py`:
  - add `GraphThoughtRequest.perspective_evidence`;
  - in `run_graph_opportunity`, after `supplied = graph_thought_request(...)`: read
    `deliberate_admission(step_view, parent.receipt.state)`. When it has a subject: capture the context,
    extend `execution_grant` with `extend_inquiry_grant(step_view, execution_grant, evidence_source_ids)`,
    add each evidence source to `request.candidates` as `ThoughtCandidate(kind="entity_evidence")` when the
    request does not hold it yet (same shape as `with_origin_evidence`), and set
    `supplied = supplied.model_copy(update={"request": ..., "grant": ..., "perspective_evidence": ...})`;
  - in `graph_thought_instruction`: add a `_perspective_rule(request)` block that returns `""` when
    `perspective_evidence` is absent.
- `workflows/inner_thought/stages/thought/steps.py`: add `"perspective_evidence"` to the `exclude` set of
  the echoed envelope. The text is already in the instruction.
- `pyrightconfig.json`: add `workflows/inner_thought/perspective.py` to `include`.

`perspective_rule` text must say all of this, in short sentences:

- This inquiry models `{subject_name}` as `{companion_name}` understands them.
- The lines are accepted entries. A report is what was said. An inference is an estimate. Entries marked
  contested, expired, or due for review are uncertain.
- Keep four things apart: what the subject said or did (cite the record), what the companion infers,
  what the subject may know, believe, or want (attributed and fallible), and what is unknown.
- The companion's knowledge is not the subject's knowledge.
- Consider one alternative explanation and one exception when the evidence allows it.
- Do not give a non-human entity human sleep, work, motives, or feelings without evidence.
- A routine is an estimate. It is never an observation of today.
- Repeating an earlier thought adds no support.
- This inquiry cannot contact the subject. If asking is the best way to learn something, state it as an
  information gap. Main cognition decides whether to ask.

Test (`tests/unit/workflows/inner_thought/test_perspective.py`, new, real store and real
`EntityContextOwner`):

- a subject with two accepted claims: the evidence text holds both claim lines, and both source IDs are in
  the step grant and in the candidates;
- a subject without any accepted claim: the evidence says so and nothing raises;
- a source in another reality or frame is not in `evidence_source_ids`;
- a deliberate episode without a subject serializes no `perspective_evidence` key.

Check: section 7 baseline, including the four replays.

Commit: `feat(inner-thought): give a perspective inquiry the subject's accepted model`

### Pass 2 — a ToM finding revises the entity model

#### Step 2.1 — revision contract and its validation

Edit:

- `workflows/entity_context/schemas.py`: add `PerspectiveRevision` (section 5.3).
- `workflows/inner_thought/schemas.py`: add `IntegrationResult.perspective_revision`.
- `workflows/inner_thought/perspective.py`: add `ExistingClaimRef`, `PerspectiveRevisionRequest`,
  `revision_request`, and `validate_perspective_revision`.
  - `revision_request`: `citable_source_ids` are the grant sources for which
    `inquiry.outside_source(view, record_id, grant=grant)` is true. `existing_claims` come from
    `context.entity_view.claims`.
- `workflows/inner_thought/pipeline.py::validate_integration`: add the keyword
  `perspective: PerspectiveRevisionRequest | None = None` and `source_texts`. Call
  `validate_perspective_revision` when a revision is present.

`validate_perspective_revision` raises `IntegrationError` for each of these:

1. a revision is present but was not requested;
2. a claim's `subject_entity_id` is not the requested subject;
3. `epistemic_kind == "observation"`;
4. `interaction` is set, or `layer == "situation"`;
5. a `source_ids` entry is outside `citable_source_ids`;
6. a `report` whose `exact_quote` is not in the text of one cited source;
7. a correction or challenge target that is not an existing claim ID;
8. `perspective_entity_id` is set and is not the subject;
9. a claim with the same `claim_key` and the same statement (whitespace-normalized) as an existing claim,
   whose `source_ids` add no source to that claim.

Each message tells the model what to do instead, in one sentence. Look at the messages in
`validate_graph_step` for the style.

Test (`test_perspective.py`): one case for each of the nine rules, and one valid draft. Pure; no store
needed for the rules except `revision_request`, which uses a real store.

Check: section 7 baseline, including the four replays.

Commit: `feat(inner-thought): validate a proposed perspective revision`

#### Step 2.2 — ask for the revision and keep it on the finding

Edit:

- `workflows/inner_thought/stages/integrate/prompts.yaml`: add `perspective_revision_template`. Model it on
  `focus_advice_template`. It names the optional field `"perspective_revision": {"claims": [...]}`, lists
  the fields of one claim, the allowed source IDs, and the existing claim IDs. Take the field wording from
  `workflows/input_triage/prompts.yaml` (`extraction`), and state the nine rules of Step 2.1 briefly.
  An empty or null revision is valid: not every inquiry changes the model.
- `workflows/inner_thought/stages/integrate/steps.py`: add the template to `IntegrationPrompts`; append it
  in `build_messages` only when the request holds `perspective_revision`; pass the request and the
  candidate texts to `validate_integration` in `run`.
- `workflows/inner_thought/graph_execution.py::run_graph_opportunity`: for a perspective episode, set
  `payload["perspective_revision"] = revision_request(...).model_dump(mode="json")` before the integrate
  stage, and pass the same request to the host-side `validate_integration`.
- `workflows/inner_thought/graph_outcomes.py`:
  - `capture_graph_outcome`: add every revision `source_ids` entry to `selected_ids`, so the original
    guards are kept;
  - `attach_graph_outcome`: new keyword `perspective_subject_id: int | None`. When the integration holds
    a revision with at least one claim, store
    `operation.metadata["perspective_revision"] = PerspectiveRevision(...).model_dump(mode="json")`.
- `workflows/inner_thought/graph_host.py::accept_step`: pass `deliberate.subject_entity_id` (or `None`)
  to `attach_graph_outcome`.

Test (`tests/unit/workflows/inner_thought/test_graph_outcomes.py` and `test_perspective.py`, real store,
explicit boundary inputs in place of the model stages, as `test_deliberate_owner.py` does):

- a perspective episode's finding stores `perspective_revision` with the subject and the claims;
- a non-perspective finding stores no such key;
- the integrate messages hold the template only when requested (`build_messages` is pure).

Check: section 7 baseline, including the four replays.

Commit: `feat(inner-thought): retain a perspective revision on its finding`

#### Step 2.3 — main cognition stages the revision

Edit:

- `workflows/main_cognition/entity_updates.py`: add `PerspectiveRevisionMetadata` (a read model with
  `extra="ignore"` and one field `perspective_revision: PerspectiveRevision`) and
  `prepare_perspective_revisions`.
  - For each stimulus ID whose record metadata holds `perspective_revision`: collect the cited source IDs,
    call `capture_entity_extraction(owner, view, subject_ids=[subject], observer_entity_id=...,
    source_ids=cited, evidence_ids=cited, as_of_utc=<newest cited source time>,
    token_budget=WORKSPACE_TAIL_TOKENS)`, then
    `hydrate_entity_extraction(extraction, EntityExtractionDraft(claims=list(drafts)), producer="tom")`.
  - `hydrate_entity_extraction` raises `ModelFormatError` when the stored draft no longer fits the model,
    for example a challenged claim was removed in the meantime. Catch exactly that error and append
    `{"source_id": <finding id>, "reason": <message>}` to `rejected`. Comment why: this is a stored model
    draft checked against a changed world, not an internal bug. Catch nothing else.
- `workflows/main_cognition/host.py::publish_output`: directly after `prepare_entity_beliefs(...)`, call
  `prepare_perspective_revisions(self.entity_context_owner, self._memory_view, self._stimulus_ids,
  observer_entity_id=request.self_entity_id)` and extend `entity_updates.proposals` and
  `entity_updates.rejected`. The existing loop then stages, records outcomes, and refreshes the workspace.
  Change nothing else in this method.

Test (`tests/unit/workflows/entity_context/test_main_and_reselect.py` or a new
`test_perspective_updates.py`, real store and owner, no model):

- a finding with a valid revision: after staging, `receipt_chain` ends in a receipt with
  `proposal.producer == "tom"`, and the claim cites the original message;
- a revision whose challenge target was retracted meanwhile: one `rejected` row, no receipt, no exception;
- a second staging of the same proposal returns the first receipt (owner idempotency);
- an inference that repeats an accepted claim with higher confidence and no new source ends as
  `rejected` / `unsupported_interpretation` (existing owner rule, audit A2).

Check: section 7 baseline; also
`.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/entity_context -t . -b`.

Commit: `feat(main-cognition): stage perspective revisions through the entity owner`

#### Step 2.4 — switch it on

Edit:

- `workflows/companion.py::_cognition_workflow_options`: add, directly after `focus_management`:
  name `theory_of_mind`; description: "Private thinking about another person or agent: what they usually
  do, what they know or believe, what they may want, and how uncertain that is. Set
  arguments.instruction to one bounded question. It sends no message and asks nobody. Its result returns
  through ordinary appraisal."
- `workflows/main_cognition/phases/prompts/route.yaml`: add one rule after the `focus_management` rule:
  use `theory_of_mind` with `ToolCall` when the companion decides to work out another person's or agent's
  habits, knowledge, motives, or likely reaction before acting; write one bounded question into
  `arguments.instruction`; it thinks about the one this part of the story is about; it never contacts
  anyone; asking them is a separate Reply decision.

Read only, and report: `workflows/interaction/steps.py` uses the same catalogue. State whether ordinary
interaction can dispatch a non-dialogue workflow. If it can, stop and ask.

Test: extend the wiring assertion that already lists `focus_management` in
`tests/unit/workflows/test_companion_wiring.py` only if that test runs without an embedding endpoint at
this revision. It is a known endpoint-dependent suite. If it cannot run, report `NOT RUN` with the reason.

Check: section 7 baseline.

Commit: `feat(main-cognition): offer theory_of_mind as a selectable action`

### Pass 3 — metacognition

#### Step 3.1 — activity evidence for a noticing step

Edit:

- `workflows/inner_thought/request_views.py`: add `RepeatedSourceActivity` and `ThoughtActivityView`.
- new `workflows/inner_thought/metacognition.py`: `METACOGNITION_OPERATIONS`, `thought_activity`,
  `render_thought_activity`, `activity_rule`. Module docstring points to `host.py`.
  - `thought_activity` is pure. Input: `request.recent_thoughts`. A step is stalled when
    `inquiry.progress_kind == "none"`. A linear thought (`loop == "inner"`) counts as a step and is never
    counted as stalled, because it records no such judgment. `repeated` lists each source that at least two
    steps cite. Use plain loops and named intermediate variables.
- `workflows/inner_thought/graph_execution.py`: add `GraphThoughtRequest.activity_evidence`; set it in
  `run_graph_opportunity` only when `parent.receipt.state.next_operation in METACOGNITION_OPERATIONS`;
  add an `_activity_rule(request)` block to `graph_thought_instruction`.
- `workflows/inner_thought/stages/thought/steps.py`: add `"activity_evidence"` to the `exclude` set.
- `pyrightconfig.json`: add `workflows/inner_thought/metacognition.py`.

`activity_rule` text must say:

- These lines are recorded facts about your own recent private thinking. They are not evidence about
  anyone else.
- They do not show why you return to this. Any reason you give is your own reading, and it can be wrong.
- Returning to a thought adds no support for it.
- You can pause, name the evidence that is missing, or continue with one specific different question.
- Nothing requires you to feel calmer, and nothing requires you to stop.

Test (`tests/unit/workflows/inner_thought/test_metacognition.py`, new): counts for a mixed window; a
noticing request carries the evidence; a `memory_search` request does not serialize the key.

Check: section 7 baseline. `private-dwelling` can diverge from here on, at a noticing step only. Report
the first differing call. The other three recordings must replay exactly.

Commit: `feat(inner-thought): show a noticing step its own recorded activity`

#### Step 3.2 — keep the facts and always tell main cognition

Edit:

- `workflows/inner_thought/graph_host.py::accept_step`: new keyword
  `activity: ThoughtActivityView | None = None`. When set, store
  `operation.metadata["thought_activity"] = activity.model_dump(mode="json")`.
  Set `"private_inquiry_finding"` to true when `step.outcome != "continue"` **or**
  `step.operation in METACOGNITION_OPERATIONS`. Leave the `MENTAL_ACTION_KEY` condition as it is.
- `workflows/inner_thought/graph_execution.py::run_graph_opportunity`: pass `activity` to `accept_step`;
  return `finding_id` also for a metacognition operation.
- Update the docstrings of `accept_step`, `pending_findings`, and the `graph_host.py` module header: a
  noticing is delivered as an intermediate result, and the episode can continue.

Before editing, list every reader of `private_inquiry_finding` with
`rg private_inquiry_finding workflows pm simulation`. At `872da60` there are three: `graph_host.py`
(writer and `pending_findings`) and `focus_management/host.py::resume`. If you find more, read each and
report.

Test (`tests/unit/workflows/inner_thought/test_graph_host.py`, real store):

- a `rumination_notice` step with `outcome="continue"` is returned by `pending_findings` and keeps
  `thought_activity`;
- a `future_projection` step with `outcome="continue"` is not;
- inside a deliberate episode, the same noticing carries no `MENTAL_ACTION_KEY`, and
  `focus_management.host.mental_outcome` still returns `None` while the episode is active.

Check: section 7 baseline (three recordings exact).

Commit: `feat(inner-thought): deliver a noticing to main cognition while thinking continues`

#### Step 3.3 — notice once

Only when V2 is not vetoed.

Edit `workflows/inner_thought/triggers.py::_repeated_source`:

- a thought whose `operation` is in `METACOGNITION_OPERATIONS` never counts toward the repeat count;
- a source that such a thought cites is not reported while that thought is in the window.

Use two plain loops: first collect the noticed source IDs, then count. Update the function docstring with
the reason: noticing dwelling must not itself count as dwelling.

Test (`tests/unit/workflows/inner_thought/test_triggers.py`):

- three stalled steps on one source fire `repeated_concern` (existing case, unchanged);
- the same window plus a `rumination_notice` on that source does not fire;
- three `rumination_notice` thoughts on one source do not fire.

Check: section 7 baseline (three recordings exact).

Commit: `fix(inner-thought): do not count a noticing as the concern it notices`

### Pass 4 — recorded evidence and one integrated gate

This pass spends provider money. Report the provider-reported cost of every recording from its manifest.
Expect cents, as in B05. The shared interim limit in `dev/work_items/beta.md` applies.

#### Step 4.1 — scenario `perspective-morning` (SC-09)

Edit:

- `tests/replay/situations.py`: add a deterministic builder `perspective_morning(now_utc, checkpoint)`,
  modelled on `private_dwelling`. It commits identity anchors and four or five synthetic Jesse dialogue
  lines on different world days, addressed to Spamuel: a workday waking pattern, one weekend exception,
  one contact preference ("nothing before coffee"), one unrelated line. Label the text as synthetic input
  in the docstring. It is not persona history.
- `tests/replay/scenarios.py`:
  - let `_dispatch_mental_action` take `workflow_name` and the partner; keep its current defaults so the
    existing scenarios do not change;
  - add `run_perspective_morning` and a typed `PerspectiveInquiryOutcome(ScenarioOutcome)`;
  - register it in `SCENARIOS`.
  - Instruction (from milestone 14.1): "Understand when Jesse usually becomes available in the morning,
    and how uncertain that pattern is, so I can judge whether and when a greeting would be welcome."
  - Name an explicit first operation that needs no embedding request, for example `narrative_stitching`.
    The harness does not record embedding requests.

The outcome reports facts and requires no model answer: the executed operations; the final status; the
finding ID or the notice; whether each thought request carried perspective evidence; the stored revision
claims with kind, layer, confidence, and source IDs; whether any claim is an `observation` (must be none);
the owner's disposition for each claim, obtained by `prepare_perspective_revisions` and
`EntityContextOwner.stage_update` in a batch on the final store; every `action.request` and every public
utterance the session produced (must be none).

Run:

```powershell
python -m tests.replay record --scenario perspective-morning --samples 3
Copy-Item -Recurse .runtime/replay/<new folder> .runtime/replay/copy-<new folder>
python -m tests.replay replay --snapshot .runtime/replay/copy-<new folder>
```

A sample without a finding or without a revision is a recorded quality result. Keep it. Do not change a
validator or a prompt to rescue it without asking.

Commit: `test(replay): record a morning-routine perspective inquiry`

#### Step 4.2 — re-record `private-dwelling`

Edit `tests/replay/scenarios.py`: extend `PrivateDwellingOutcome` with: the noticing steps (operation,
outcome); whether each received activity evidence; whether each was delivered as `thought.created`;
whether `repeated_concern` fires after the noticing (expected: no).

Run three samples, copy, replay the copies. A model that never notices is a recorded quality result.

Commit: `test(replay): record noticing with activity evidence in private-dwelling`

#### Step 4.3 — integrated live gate

Add `tests/live/test_perspective_integration.py`, modelled on `tests/live/test_private_integration.py`:
the real application from `build_app`, a real store, real OpenRouter chat and embeddings. Each part
writes evidence under `.runtime/beta-b06/integration/<timestamp>-<id>-part-<x>/`.

| Part | Input | Passes when | `NOT RUN` when |
| --- | --- | --- | --- |
| A — ToM effect | The Step 4.1 situation, and a ToM action committed at the dispatch boundary. Label it synthetic input, as the replay scenario does. | The finding is claimed by a real main turn; the subject's family holds a receipt with `producer == "tom"`; the ToM action has no public reply; `FocusManagement` published no speech. Record whether main cognition chose to speak afterwards: that is a separate action and is allowed. | The episode ends without a revision in three attempts (model quality). |
| B — ToM selection | A real Jesse message that invites thinking about Jesse's habits. No forced action. | Main cognition routes `theory_of_mind`. | Main cognition selects another action in three attempts. Keep the three decision records. |
| C — noticing reaches main | The `private-dwelling` situation in the real application. | A noticing with `thought_activity` is claimed by a real main turn. Record the action main cognition chose. | No noticing arises within the tick limit, or main commits an inference wait. Do not clear a wait. |

Never commit an action to turn Part B green. Never weaken an assertion after a failed run.

Commit: `test(live): gate perspective revision, selection, and noticing`

#### Step 4.4 — owner review package

Add `dev/work_items/beta_b06_review.py`, modelled on `dev/work_items/beta_b05_review.py`. Capture from the
replayed final store of the most informative `perspective-morning` sample and the most informative
`private-dwelling` sample: Jesse's lines, the action, every executed step with its reasons, the finding,
the proposed claims, the owner's outcomes, the resulting entity view, the noticing with its activity
facts. Register with `uv run --locked python -m simulation.review register-package`. Put the direct URL
in the work record. Invent no rating.

Commit: `docs(beta): register the B06 review package`

### Pass 5 — documentation and close-out

Read the project-documentation skill first (`~/.agents/skills/project-documentation/SKILL.md`).
Describe only what Pass 4 verified as implemented. Label the rest.

- `docs/subsystems/entity-context.md`: deliberate ToM is a producer; what it can and cannot write.
- `docs/subsystems/deliberate-mental-work.md`: the `theory_of_mind` entry, the subject, the perspective
  evidence, the revision path, a data-flow diagram.
- `docs/subsystems/private-loop.md`: activity evidence, the intermediate noticing, "notice once", and the
  unchanged hard stop with the owner's rationale.
- `docs/workflows/scenario-replay.md`: the new scenario row and the changed `private-dwelling` row.
- `docs/concepts/glossary.md`: "perspective inquiry", "perspective revision", "thought activity". Check
  `docs/concepts/naming.md` first.
- Module headers and docstrings of every changed module, in the same commits as the code where possible.

```powershell
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

Inspect the changed pages and their API links in `site/`.

Then: finish `dev/work_items/beta-b06.md`, add one dated note to card 7 in `dev/kanban/board.json` (keep the
card `doing`), and stop. B07 is not part of this plan.

Commit: `docs(beta): record B06 theory of mind and metacognition`

---

## 7. Checks

Baseline after every step. All of these must end as they did before the step:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/inner_thought -t . -b
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/focus_management -t . -b
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/entity_context -t . -b
.venv/Scripts/python.exe -m unittest tests.unit.test_architecture -b
npx --no-install pyright --project pyrightconfig.json
```

Replay check (no endpoint, no cost). Never replay an original folder. Copy first:

```powershell
$step = "s11"   # the step you just finished
foreach ($name in "private-continuity-20261114T180018Z", "deliberate-entry-20261114T180133Z",
                  "deliberate-reflection-20261114T180458Z", "private-dwelling-20261114T181741Z") {
    Copy-Item -Recurse ".runtime/replay/$name" ".runtime/replay/b06-$step-$name"
    python -m tests.replay replay --snapshot ".runtime/replay/b06-$step-$name"
}
```

Expected: all four replay exactly through Pass 2. From Step 3.1, `private-dwelling` can diverge at a
noticing step or at a changed trigger list. Any other divergence is a stop condition.

Known red suites. Do not fix them in B06 and do not count them as your failures. The full list is in the
handoff, section "Known defects outside B06". The ones you will meet:

- `tests/unit/pm/introspection`: three failures;
- `tests.unit.workflows.test_companion_wiring` and `tests.unit.workflows.cognitive_control.test_wait`:
  retry remote embeddings until an endpoint answers;
- main-cognition discovery fixtures, `test_cognition_wiring` (11 errors), `test_focus` (3);
- `tests.live.test_private_inquiry`: failed 3/3 on model quality on 2026-11-14;
- every recorded sample shows one rejected early-triage answer at `entity_extraction.claims[0]`.

If a required endpoint is missing, report `NOT RUN — missing prerequisite` with the command. Never report
a pass for a check that did not run.

---

## 8. Report format for each pass

Append to `dev/work_items/beta-b06.md`, in this order (AGENTS.md section 19):

1. **Changed:** components, contracts, boundaries.
2. **Why:** the requirement served (name the decision D1–D6 or the scenario SC-07 / SC-09).
3. **Verified:** exact commands, real endpoint or store used, observed result, evidence path.
4. **Not verified:** every gate not run, with the reason.
5. **Documentation:** pages, headers, diagrams changed.
6. **Open concerns:** risks, differences from this plan, owner decisions still needed.

---

## 9. Risks and limits

| # | Risk or limit | Handling |
| --- | --- | --- |
| R1 | Main cognition may not select `theory_of_mind`. It did not select `focus_management` in the B05 gate. | Part B reports `NOT RUN`. Part A proves the effect with a dispatch-boundary input. |
| R2 | The entity model can change between the thinking and the appraisal. | The owner returns `deferred` or the draft is rejected. Both are recorded. No retry in B06. |
| R3 | The perspective evidence makes the thought prompt longer (budget 6000 tokens for the view). | The existing context measurement applies. A prompt that does not fit fails visibly. |
| R4 | A stored field can silently invalidate recordings. | Rule 3 and the replay check after every step. |
| R5 | Every noticing costs one main cycle. | "Notice once" bounds it. Report the count in Step 4.2. |
| R6 | Repeated ToM runs can add near-duplicate estimates with new wording. | Rule 9 of Step 2.1 and owner rule A2 stop exact repeats. Wording drift is a model-quality item for review. |
| R7 | Main cognition reads a noticing as text only (V3). | The facts are on the record for review and for later reflection work. |
| R8 | ToM about a third entity is not possible (V1). | Recorded limit. One added step if vetoed. |
| R9 | ToM cannot cite the companion's own utterances for an entry (`outside_source` excludes self-origin). | Recorded limit of the first version. |

## 10. Where AGENTS.md section 5 is covered

| Required item | Section |
| --- | --- |
| Current module graph and execution path | 4, 5.1, 5.2 |
| Current problems with files and symbols | 4 (audit A1–A7), handoff F1–F11 |
| Target responsibilities and folder structure | 5.3, 5.4 |
| Existing and proposed types and signatures | 5.3 |
| Data ownership, persistence, side effects, error boundaries | 5.4 |
| Behaviour preserved or changed | 5.5 |
| Live prerequisites and verification plan | 6 (Pass 4), 7 |
| Ordered small steps | 6 |
| Risks, open decisions, owner choices | 2, 9 |
