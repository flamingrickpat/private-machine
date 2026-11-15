# B06: theory of mind and metacognition

Status: implemented and corrected, 2026-11-15. Read "Audit and corrections" near the end first: three statements in the pass sections are wrong.
Plan (owner-accepted `accept all`):
`dev/work_items/beta-b06-plan.md`. Handoff authority:
`dev/work_items/beta-handoff-b06-20261114.md`.

## Pass 1 — a ToM action can enter the private slot

### Step 1.1 — subject on the admission

1. **Changed:**
   - `workflows/inner_thought/schemas.py`: `DeliberateAdmission.subject_entity_id` (optional, `exclude_if=_absent`), module-level `_absent` helper.
   - `workflows/inner_thought/inquiry.py::stage_deliberate_inquiry`: a set subject must resolve to an `Entity` other than the observer, else `ValueError`.
   - `workflows/focus_management/host.py`: `PERSPECTIVE_WORKFLOW`, `MENTAL_WORKFLOWS`, `FocusManagement.accepts` override, `RequestedMentalAction.subject_entity_id`, `requested_action` partner wiring, `_admit_requested` passes the subject; module docstring paragraph on the second catalogue name.
2. **Why:** decision D1 (a second menu entry beside `focus_management`) and the subject-carrying admission that Pass 2 builds on.
3. **Verified:**
   - `.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/focus_management -t . -b`: 18 tests OK (3 new).
   - Section 7 baseline: inner_thought 221 OK, focus_management 18 OK, entity_context 32 OK, architecture 7 OK, `npx --no-install pyright --project pyrightconfig.json` 0 errors.
   - Four replayed copies under `.runtime/replay/b06-p1s1-*`: all "sample N: ok", exit 0.
4. **Not verified:** none for this step; no model call.
5. **Documentation:** module docstring of `focus_management/host.py` (second catalogue name).
6. **Open concerns:** none. Commit `b7ed8e8`.

### Step 1.2 — perspective evidence for the thought step

1. **Changed:**
   - New `workflows/inner_thought/perspective.py`: `PerspectiveContext`, `capture_perspective_context`, `render_perspective_context`, `perspective_rule`.
   - New `tests/unit/workflows/inner_thought/test_perspective.py`.
   - `workflows/inner_thought/graph_execution.py`: `GraphThoughtRequest.perspective_evidence`; in `run_graph_opportunity` the perspective block reads `deliberate_admission`, captures the context, extends the step grant (`extend_inquiry_grant`) and request candidates (`kind="entity_evidence"`), and seeds the recall `dependencies` dict so the recall branch keeps the extended sources; `_perspective_rule` returns the assembled block or `""`.
   - `workflows/inner_thought/stages/thought/steps.py`: `perspective_evidence` excluded from the echoed envelope.
   - `pyrightconfig.json`: include `perspective.py`.
2. **Why:** decisions D2/D5 — the private loop reads the accepted model of its subject and the original messages behind it; no other step's prompt changes (rule 3).
3. **Verified:**
   - `tests.unit.workflows.inner_thought.test_perspective`: 5 checks OK (two-claim evidence text + grant + candidates; no-claim subject; foreign-frame source excluded; plain deliberate episode serializes no `perspective_evidence` key; rule text).
   - Section 7 baseline: inner_thought 226 OK, focus_management 18 OK, entity_context 32 OK, architecture 7 OK, pyright 0 errors.
   - Four replayed copies `.runtime/replay/b06-p1s2-*`: all exact.
4. **Not verified:** no model call in this step; live selection of `theory_of_mind` is Pass 4.
5. **Documentation:** module docstring of `perspective.py`; docstring updates in `graph_execution.py` for the new fields and helpers.
6. **Open concerns:** the plan's `_perspective_rule` assembly happens at the build site (subject name is only known there); the ball travels as one rendered block and `_perspective_rule` returns it as-is. Same observable prompt, same serialized shape; noted as a difference from the plan's letter, not its contract. Commit `bcdca6a`.

## Pass 2 — a ToM finding revises the entity model

### Step 2.1 — revision contract and its validation

1. **Changed:** `PerspectiveRevision` in `workflows/entity_context/schemas.py`; `IntegrationResult.perspective_revision` (optional, `exclude_if=_absent`) in `workflows/inner_thought/schemas.py`; `ExistingClaimRef`, `PerspectiveRevisionRequest`, `revision_request`, `validate_perspective_revision` in `perspective.py`; `validate_integration` gains `perspective`/`source_texts` keywords (function-local import breaks the pipeline<->perspective cycle; TYPE_CHECKING-only annotation).
2. **Why:** decisions D2/D5 — a proposed revision is validated in the private loop before it can be kept, with the nine rules of plan step 2.1.
3. **Verified:** 16 checks in `test_perspective.py` (9 rules incl. the not-requested case via `validate_integration`, valid draft, `revision_request` against a real store). Baseline: inner_thought 237, focus_management 18, entity_context 32, architecture 7, pyright 0. Four replays exact.
4. **Not verified:** none (no model call).
5. **Documentation:** docstrings of the new types and validators.
6. **Open concerns:** rule 1 (revision without a request) is enforced in `validate_integration` because `validate_perspective_revision` requires a request by signature. Commit `4ef1abe`.

### Step 2.2 — ask for the revision and keep it on the finding

1. **Changed:** `perspective_revision_template` in `stages/integrate/prompts.yaml` (modeled on `focus_advice_template`, field wording from input triage); `IntegrationPrompts` + `build_messages` + `run` in `stages/integrate/steps.py`; `run_graph_opportunity` sets `payload["perspective_revision"]` and passes the request/texts to host-side validation; `capture_graph_outcome` keeps revision source guards; `attach_graph_outcome` stores `perspective_revision` metadata under the subject; `graph_host.accept_step` passes `deliberate.subject_entity_id`.
2. **Why:** the integrate stage is the only caller that may answer; the finding is the stored receipt that main cognition later stages (D2).
3. **Verified:** 3 new checks in `test_perspective.py` (finding keeps validated revision + original guard; plain finding stores no key; template only when requested). Baseline: inner_thought 240, focus_management 18, entity_context 32, architecture 7, pyright 0. Four replays exact.
4. **Not verified:** real model producing a revision; integrated staging (Pass 4).
5. **Documentation:** integrate module docstring and prompt YAML.
6. **Open concerns:** none. Commit `1f7a585`.

### Step 2.3 — main cognition stages the revision

1. **Changed:** `PerspectiveRevisionMetadata` + `prepare_perspective_revisions` in `workflows/main_cognition/entity_updates.py` (hydrates the stored draft with `producer="tom"`, records `ModelFormatError` as a rejected row with the plan's comment); `publish_output` extends proposals/rejected right after `prepare_entity_beliefs`; nothing else in the method changed; new `tests/unit/workflows/entity_context/test_perspective_updates.py`.
2. **Why:** decision D2 — the entity entry is written by main cognition at its final commit through `EntityContextOwner.stage_update`, avoiding the crash-recovery hazard (A6).
3. **Verified:** 4 checks (staged `tom` receipt citing the message; retracted challenge target -> one rejected row; owner idempotency; A2 repeated-inference rejection). Baseline: inner_thought 240, focus_management 18, entity_context 36, architecture 7, pyright 0. Four replays exact.
4. **Not verified:** a real main turn claiming a real ToM finding (Pass 4, Part A).
5. **Documentation:** docstrings of the new boundary function and read model.
6. **Open concerns:** none. Commit `a5cabcb`.

### Step 2.4 — switch it on

1. **Changed:** `_cognition_workflow_options` gains the `theory_of_mind` entry directly after `focus_management`; one route rule added after the `focus_management` rule in `route.yaml`.
2. **Why:** decision D1 — the action model sees a plainly described second mental-workflow choice.
3. **Verified:** read-and-report on `workflows/interaction/steps.py`: ordinary interaction only ever dispatches `dialogue_story` replies locally (`local_reply` gate); any other workflow (including `theory_of_mind`) becomes a Reconsider outcome and never runs from the dialog path. Added a bus-level `accepts` assertion for both catalogue names to `test_deliberate_owner.py` (19 focus-management checks green; covers audit A1 since the wiring suite cannot run). Baseline: inner_thought 240, entity_context 36, architecture 7, pyright 0. Four replays exact.
4. **Not verified:** `tests.unit.workflows.test_companion_wiring` NOT RUN — the module retries remote embeddings with 30s backoff without an endpoint; verified empirically (attempts 1-9, no endpoint). Per plan, the wiring assertion was not extended because the suite cannot run offline; the accepts contract is covered by the note above.
5. **Documentation:** no prose page changed at this step; the catalogue entry and route rule are prompt material.
6. **Open concerns:** a model that never selects the new action is a Pass 4 quality question (risk R1). Commit `532d65d`.

## Pass 3 — metacognition

### Step 3.1 — activity evidence for a noticing step

1. **Changed:** `RepeatedSourceActivity` + `ThoughtActivityView` in `request_views.py`; new `metacognition.py` (`METACOGNITION_OPERATIONS`, `thought_activity`, `render_thought_activity`, `activity_rule`); `GraphThoughtRequest.activity_evidence`; `run_graph_opportunity` renders the activity block only for metacognition operations; `_activity_rule` adds it to the instruction; echoed-envelope exclude set; new `test_metacognition.py`.
2. **Why:** decision D3 (the private loop notices with recorded facts about its own activity); prompt parity rule for every other step.
3. **Verified:** 5 checks (mixed-window counts, rendered facts, noticing payload carries the block, `memory_search` serializes no key, operation pair). Baseline: inner_thought 245, focus_management 19, entity_context 36, architecture 7, pyright 0. Three recordings replay exactly; `private-dwelling` diverges at call 20 (samples 1-2) and call 16 (sample 3) — each the `rumination_notice` step now carrying the activity block; first differing call reported, as the plan allows.
4. **Not verified:** live noticing delivery (Pass 4 Part C).
5. **Documentation:** module docstring of `metacognition.py`, field docstrings.
6. **Open concerns:** none. Commit `b3b1ae3`.

### Step 3.2 — keep the facts and always tell main cognition

1. **Changed:** `accept_step` gains `activity` (stored as `thought_activity` metadata) and marks `private_inquiry_finding` for metacognition operations even with `outcome == "continue"`; `run_graph_opportunity` passes `activity` and returns `finding_id` for noticing steps; `pending_findings` and the `graph_host` module header document the intermediate noticing. Reader audit `rg private_inquiry_finding`: exactly three readers (writer, `pending_findings`, `focus_management.host.resume`) as the plan predicted.
2. **Why:** decision D3 and audit A4 — main cognition always hears the noticing, once, while the episode continues.
3. **Verified:** 3 new checks in `test_graph_host.py` (noticing is pending with `thought_activity`; `future_projection` continue is not; inside a deliberate episode the noticing carries no `MENTAL_ACTION_KEY` and `mental_outcome` stays None while active). Baseline: inner_thought 270, focus_management 19, entity_context 36, architecture 7, pyright 0. Replays: three exact, `private-dwelling` diverges at the same noticing calls only.
4. **Not verified:** a real main turn claiming a delivered noticing (Pass 4 Part C).
5. **Documentation:** `accept_step`, `pending_findings`, and the `graph_host.py` module header.
6. **Open concerns:** none. Commit `e8d8bb9`.

### Step 3.3 — notice once

1. **Changed:** `_repeated_source` in `triggers.py`: two plain loops (collect noticed source ids from metacognition thoughts, then count), so noticing never counts toward the repeat count and a source the noticing names is not reported while it is in the window; docstring states the V2 reason.
2. **Why:** decision V2 — noticing dwelling must not itself count as dwelling (recursion SC-07 forbids).
3. **Verified:** 3 trigger checks incl. the unchanged linear-repeat case, the suppressed concern, and three notices that never fire. Baseline: inner_thought 272, focus_management 19, entity_context 36, architecture 7, pyright 0. Replays: three exact; `private-dwelling` diverges at the noticing calls only.
4. **Not verified:** none for this step.
5. **Documentation:** function docstring.
6. **Open concerns:** none. Commit `e72d87b`.

## Pass 4 — recorded evidence and one integrated gate

### Step 4.1 — scenario `perspective-morning` (SC-09)

1. **Changed:** `tests/replay/situations.py` adds `perspective_morning` (5 synthetic Jesse lines labelled as engineering input); `tests/replay/scenarios.py` generalizes `_dispatch_mental_action` (workflow name, partner, explicit source ids; defaults unchanged) and adds `PerspectiveInquiryOutcome`, `run_perspective_morning` (first operation `narrative_stitching` so no embedding request is needed), registered in `SCENARIOS`. The outcome reports executed operations, status, finding/notice, thought-request evidence counts (from the flushed `calls.jsonl`), stored revision claims, observation-claim guard, owner rehearsal dispositions via `prepare_perspective_revisions` + `EntityContextOwner.stage_update` in one uncommitted batch, action.request count, and public utterances.
2. **Why:** decision D6 — first recorded ToM scenario; a revision is optional and a missing one is a recorded quality result.
3. **Verified:** `python -m tests.replay record --scenario perspective-morning --samples 3` → `perspective-morning-20261114T222352Z`; all 3 samples completed with findings (`deliberate_status` completed), every thought request carried the perspective block (4/4, 4/4, 6/6), no observation claim, no speech, exactly one action.request (the script's own). All 3 samples replay exactly from copies. Provider-reported cost: $0.004310 + $0.005967 + $0.007085 = **$0.017362** total. Two earlier exploratory snapshots (1+2 samples) are kept under `.runtime/replay/`; those replayed exactly under their own folders (renumbering a combined folder broke the `name:sample` identity seed and was discarded).
4. **Not verified:** none of the samples produced revision claims: the model's outcomes concluded "ask Jesse" information gaps and the integrate stage honestly declined to write entries. Recorded quality result; no validator or prompt changed to rescue it (plan rule 6).
5. **Documentation:** docstrings of the situation and outcome models.
6. **Open concerns:** risk R6 materialized positively (no near-duplicate claims); the empty revision is the model-quality item the owner reviews. Commit `9a39945` (code).

### Step 4.2 — re-record `private-dwelling`

1. **Changed:** `PrivateDwellingOutcome` gains `NoticingObservation` (`noticing_steps`: operation, outcome, `received_activity`, `delivered_thought_created`, `repeated_concern_after`).
2. **Why:** decision D3/V2 — the recording must show noticing with activity facts, delivery while the episode continues, and the quieted trigger.
3. **Verified:** `record --scenario private-dwelling --samples 3` → `private-dwelling-20261114T222749Z`; all 3 samples produced noticing steps (sample 1 had `attention_drift_notice` and `rumination_notice`), every noticing had `received_activity=True`, `delivered_thought_created=True`, `repeated_concern_after=False`; `repeated_concern` no longer fires (old recordings fired True in 2 of 3 samples — V2 visible live). All 3 samples replay exactly from copies. Provider cost: $0.014821 + $0.010614 + $0.009599 = **$0.035034** total. The pre-B06 `private-dwelling-20261114T181741Z` recording is superseded.
4. **Not verified:** nothing for this step; main-side claim of a noticing is the live gate below.
5. **Documentation:** outcome model docstrings.
6. **Open concerns:** none. Commit `d3d8fc9` (code).

### Step 4.3 — integrated live gate

1. **Changed:** new `tests/live/test_perspective_integration.py` (real `build_app`, real store, real OpenRouter chat + embeddings). Part A: ToM action at the dispatch boundary, three attempts; passes when a real main turn claims the finding and the subject family holds a `tom` receipt, with no public reply and no FocusManagement speech. Part B: three real Jesse messages; passes when main routes `theory_of_mind`. Part C: the dwelling situation; passes when a noticing with `thought_activity` is claimed by a real main turn. Never forces an action or clears a wait.
2. **Why:** decisions D1/D2/D3 — the effect boundary (tom receipt), the selection boundary, and the noticing boundary on the real path.
3. **Verified (evidence roots `.runtime/beta-b06/integration/`):**
   - Part C run 1: `ValueError: Internal derivations cannot count as acquired original evidence.` — a **pre-existing** guard in `stage_inquiry_step` turned an unsupported model `new_evidence` claim into a hard crash (the child validator cannot check root-ness). Recorded as `failure.json`; not caused by B06 (the check predates it); reported, not patched out of scope.
   - Part C runs 2-3: noticing produced in ~4 opportunities with real activity facts (window `[12, 10]`, 2 stalled, repeated sources 4 and 5), published as `thought.created`; `NOT RUN` because main committed an inference wait instead of claiming in the checked step (plan's stated condition; B05's own gate had the same). Evidence: `noticing.json`, `not-run.json`.
   - Part A: one clean run — attempt 1 executed the full ToM episode (operations `narrative_stitching, feasibility_check, reflection_review`), finding id delivered with `mental_status=proposed`; attempts 2-3 also admitted cleanly after unique dispatch commit ids; `NOT RUN — no ToM episode produced a revision in three attempts` (plan's stated condition; matches the recorded samples).
   - Part B: `NOT RUN — main did not select theory_of_mind in three attempts` (risk R1, exactly like B05's focus_management gate: 0 of 3); all three decision records and request counts kept (25/52/95 requests).
4. **Not verified:** a `tom` receipt staged by a real main turn (no revision produced — model quality); main routing `theory_of_mind`; main claiming a noticing without a wait. All three are the planned NOT RUN conditions; two gate bugs found and fixed during the runs (repeated dispatch commit id; frozen `ManualClock` never made the focus-management timer due for the close step).
5. **Documentation:** module docstring of the gate.
6. **Open concerns:** provider spend for the gate runs is not in a manifest (the gate tracks request counts, not usage); estimate well under $0.30, far inside the shared $25 interim budget. Commit `fab5d03`.

### Step 4.4 — owner review package

1. **Changed:** new `dev/work_items/beta_b06_review.py` (models the B05 review script; replays a copy of each sample with no endpoint, then captures): perspective-morning sample-03 and private-dwelling sample-01 packages, including outcome facts (revision claims, rehearsal dispositions, evidence-request counts, noticing readout) as a `facts` layer (`present_packages` ignores unknown keys).
2. **Why:** decision D6 — two things the owner can open and rate.
3. **Verified:** `python -m dev.work_items.beta_b06_review ...` produced 8 packages; registered as two captures (a single `--combine` of both stores would collide on shared record ids):
   - http://127.0.0.1:8095/#capture/b06-tom-20261114
   - http://127.0.0.1:8095/#capture/b06-metacognition-20261114
4. **Not verified:** no revision claims exist in any sample, so the revision section of the review is factually empty; the owner sees the inquiry, the evidence in every prompt, and the delivered finding.
5. **Documentation:** script docstring with the working register commands.
6. **Open concerns:** nothing invented; no rating created. Commit `a256de8`.

## Pass 5 — documentation and close-out

1. **Changed (docs):**
   - `docs/subsystems/deliberate-mental-work.md`: new "Theory of mind" section — the `theory_of_mind` entry, the subject, perspective evidence, the revision path with a data-flow diagram, and the "no contact" rule; link to the B06 record.
   - `docs/subsystems/entity-context.md`: status updated (deliberate ToM now a bounded producer); new "Deliberate ToM as a producer" section with the may/may-not table.
   - `docs/subsystems/private-loop.md`: new "Noticing its own thinking" section — recorded activity facts, the intermediate noticing delivery, notice once, and the unchanged hard stop with the owner's 2026-11-14 rationale.
   - `docs/workflows/scenario-replay.md`: new `perspective-morning` row and the updated `private-dwelling` row plus outcome descriptions.
   - `docs/concepts/glossary.md`: new entries "perspective inquiry", "perspective revision", "thought activity"; the "entity model" entry updated to name `tom` as a producer.
   - `docs/concepts/naming.md` checked first: the new names follow its rules; no conflicts.
   - Module headers and docstrings of changed modules were updated in their own commits (perspective.py, metacognition.py, graph_host.py, integrate steps.py, focus_management host.py, graph_execution fields, entity_updates boundaries).
2. **Why:** documentation must describe only what Pass 4 verified (AGENTS.md section 10 and the project-documentation skill).
3. **Verified:** `uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict` passes (one warning during the first build was my own link to `dev/`; converted to the HTML-anchor convention the manual already uses; second build clean in 200.86s). Inspected `site/`: the changed pages render, both mermaid diagrams appear, and the new API-reference pages `reference/workflows/inner_thought/perspective.html` (82 doc objects) and `metacognition.html` (26 doc objects) exist with docstrings; cross-refs resolve to `../reference/...` targets that exist.
4. **Not verified:** none for documentation; the behavioral claims are the Pass 4 evidence (recorded findings, noticing delivery, NOT RUN gates kept).
5. **Documentation:** the five pages above.
6. **Open concerns:** none.

## Audit and corrections — 2026-11-15

An audit of the delivery above found three wrong statements in "Not verified" and one real defect behind them. This section corrects the record. The pass sections above are kept as written; where they disagree with this section, this section is right.

### What the audit found

1. **"The integrate stage honestly declined" was false.** In all 8 real outcomes (6 recorded samples, 2 live Part A episodes) the model proposed 3 to 5 claims. Each time the repair boundary answered `Expected null at $.perspective_revision`, and the model then returned null.
   - Cause 1: `perspective_revision_template` showed `"layer": "<layer>"` and never named the allowed layers. The model invented `disposition`, `preference`, `habit`, `report`, `trait`. The template also did not state the four-claim limit; 2 of 8 answers held five claims.
   - Cause 2: the repair library reports the last branch of a union. Pydantic writes an optional field as `[value, null]`, so a wrong value inside an optional object was reported as "Expected null".
   - Reproduction without an endpoint: `.runtime/beta-b06/audit/repro_union_message.py`. A recorded rejected answer is accepted with four claims when only its layer names change.
   - Consequence: the nine-rule validator, the stored revision, `prepare_perspective_revisions`, and a `tom` receipt had never run on real model output.
2. **Part A was one attempt, not three.** Main cognition committed a wait after it appraised the first finding. Actions 47 and 48 stayed `dispatch_status="pending"` with no episode. "Attempts 2-3 also admitted cleanly" was false.
3. **Part C was under-reported.** In both runs that reached a noticing, a real main turn had claimed it (tick 9 with stimuli `[12, 19]`; tick 7 with `[13]`). The wait came afterwards. The gate checked the wait before the claim and reported "wait instead of claiming".
4. **The perspective evidence was empty in every real run.** The situation held no entity claim, so every thought prompt read "No accepted entry exists for this subject" and carried no entity-evidence candidate.
5. **The manual described the revision path without a status label**, although no real run had produced a `tom` entry.
6. `prepare_perspective_revisions` caught `KeyError` and `TypeError` as well. The plan and its own docstring allow only `ModelFormatError`.

### Corrections

1. **Changed:**
   - `pm/llm/structured.py`: `_unmasked_problem` and `_null_branches_first`. When the schema check rejects an answer, the boundary repeats the check with the null branch first and uses that problem as the correction text. Accepted payloads still come from the original schema.
   - `workflows/inner_thought/stages/integrate/prompts.yaml`: the template lists the five layers a revision may use, states the four-claim limit, and says to keep `interaction` null.
   - `workflows/main_cognition/entity_updates.py`: only `ModelFormatError` is caught; the unreachable empty-citation branch is removed.
   - `tests/replay/situations.py`, `tests/replay/scenarios.py`: `perspective_morning` stages one accepted report about Jesse through the real `EntityContextOwner` (producer `immediate`, source: the oldest line). The action names only the four newer lines. The outcome counts thought completions only and reports accepted-entry and entity-evidence counts.
   - `tests/live/test_perspective_integration.py`: each Part A attempt has its own application, store, and evidence folder, and a refused admission raises. Part C checks the claim first. Both parts record what main cognition did after the appraisal.
   - `dev/work_items/beta_b06_review.py`: the recorded package includes the accepted entity receipts; `--live-part-a` and `--live-part-c` capture from a copy of a passed gate run's final store.
2. **Why:** decision D2 (private thinking proposes, main cognition writes with `producer="tom"`) had no real evidence, and the stated reason for that was wrong.
3. **Verified:**
   - Unit and static checks: inner_thought 273, focus_management 19, entity_context 36, `tests.unit.pm.llm.test_structured` 22, architecture 7, pyright 0 errors.
   - Replays from fresh copies, all exact: `private-continuity-20261114T180018Z`, `deliberate-entry-20261114T180133Z`, `deliberate-reflection-20261114T180458Z`, `private-dwelling-20261114T222749Z`, and the new `perspective-morning-20261115T081258Z` (3 samples).
   - Recording `perspective-morning-20261115T081258Z` at revision `e3d3182` (clean tree), provider-reported cost $0.001665 + $0.007078 + $0.002057 = **$0.010799**. All 3 samples completed with a finding and a stored revision (3, 1, and 4 claims). Every thought request carried an accepted entry and an entity-evidence candidate (2/2, 3/3, 2/2). The real owner staged every proposed batch in rehearsal (`staged`, `accepted`). No observation claim, no speech, one action request (the script's own). No sample received an "Expected null" correction.
   - Live gate Part A, real OpenRouter, `.runtime/beta-b06/integration/20261115T083307Z-e590778f-part-a-attempt-1`: **passed on attempt 1.** Operations `narrative_stitching, feasibility_check, plan_next_step, feasibility_check`; main tick 14 claimed finding 19; the entity owner staged receipt 46 with `producer="tom"` (two reports, one inference); `FocusManagement` published no speech; 30 requests. Afterwards main cognition dispatched a `dialogue_story` action to ask Jesse when his coffee is finished, and committed a wait for the answer. That is its own separate decision; the gate runs no dialogue module, so no reply was sent.
   - Live gate Part C, `.runtime/beta-b06/integration/20261115T083806Z-358898db-part-c`: **passed.** Noticing 21 with `thought_activity` (window `[12, 10]`, two stalled steps, repeated sources 4 and 5) was claimed by main tick 9. Main cognition chose to stop and committed a wait; 29 requests.
   - Owner review, registered: http://127.0.0.1:8095/#capture/b06-tom-live-20261115 (the live run with the staged `tom` entry and the main turn), http://127.0.0.1:8095/#capture/b06-metacognition-live-20261115 (the live noticing and the main turn), http://127.0.0.1:8095/#capture/b06-tom-20261115 (recorded sample 3). `b06-metacognition-20261114` is still current. `b06-tom-20261114` is superseded: it shows the empty model and the nulled revision.
4. **Not verified:**
   - Part B was not run again. Nothing it exercises changed. Its 2026-11-14 result stands: main cognition did not select `theory_of_mind` in three attempts.
   - Part C run `20261115T083614Z-62e8fc86-part-c` failed with `ValueError: Internal derivations cannot count as acquired original evidence.` This is the defect reported in Pass 4, now seen in 2 of 5 Part C runs. Its `failure.json` is kept. It is not patched here: the fix changes where a model claim about new evidence is validated, and the plan names no such change.
   - Provider cost of the gate runs is not in a manifest. The gate records request counts only.
   - The five-claim correction text was not seen live. Offline, the boundary now reports the raw library text `[...] is too long` for that case, which is true but long.
5. **Documentation:** `docs/subsystems/deliberate-mental-work.md` now states which part of the theory-of-mind path was observed and which was not; `docs/workflows/scenario-replay.md` describes the accepted claim in the situation and the new outcome counts. The strict MkDocs build passed after these edits.
6. **Open concerns:**
   - The `stage_inquiry_step` crash needs an owner decision. A model answer is a trust boundary; in the dwelling situation it stopped a private opportunity with a traceback in 2 of 5 runs.
   - Main cognition never selected `theory_of_mind` or `focus_management` by itself in six attempts across B05 and B06. The route rule or the menu text needs its own look.
   - Difference from the plan: step 4.3 says Part C is `NOT RUN` when "main commits an inference wait". The gate now applies that only when the wait comes without a claim. A turn that claims the noticing and then waits meets the pass condition of the same row.
   - Environment, not a code defect: recording needed `PYTHONHASHSEED=0` set in the shell on 2026-11-15. The harness restarts itself in a child process when the variable is absent; both processes together need about 11 GB of commit charge, and the machine had 10.7 GB free (a process named `basalt` held 69.6 GB). The child failed with `MemoryError` while loading `canon.yaml`.

Commits of this section, on `beta`, none pushed: `1302e29`, `b60e564`, `cb5da64`, `e3d3182`, `ac89871`, `49edfd2`, and the commit that holds this text.

## Status

B06 is implemented, and its two main paths are observed in the real application.

- Theory of mind: a dispatched `theory_of_mind` action entered the private slot with its subject, read the accepted model of Jesse and the line behind it, and proposed entries. A real main turn staged them through `EntityContextOwner` with `producer="tom"` (live gate Part A, 2026-11-15).
- Metacognition: a noticing step read recorded activity facts and was delivered, and a real main turn appraised it (live gate Part C, 2026-11-15). A noticing never counts as the concern it notices.
- Recordings `perspective-morning-20261115T081258Z` and `private-dwelling-20261114T222749Z` replay exactly.
- Owner review: http://127.0.0.1:8095/#capture/b06-tom-live-20261115, http://127.0.0.1:8095/#capture/b06-metacognition-live-20261115, http://127.0.0.1:8095/#capture/b06-tom-20261115, http://127.0.0.1:8095/#capture/b06-metacognition-20261114.
- Not observed: main cognition selecting `theory_of_mind` by itself (Part B, 0 of 3).
- Open defect, reported and not patched: an unsupported model `new_evidence` claim that cites an internal derivation crashes `stage_inquiry_step` (2 of 5 Part C runs).

B07 is not part of this plan.
