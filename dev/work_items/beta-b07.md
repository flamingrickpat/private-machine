# B07: interval appraisal and run readiness

Status: Passes 0 to 5 and 7 done, 2026-11-15; Pass 6 partly. See "Status" at the end. The owner answered Q1 to Q3 the same day (section "Owner answers").
Handoff authority: `dev/work_items/beta-handoff-b07-20261115.md` (owner decisions D1–D10 answered
2026-11-15). Evidence root: `.runtime/beta-b07/`.

## Pass 0 — verify and surface owner prerequisites

### Step 2 — findings F1 to F12 and F17 against the code at `cd641e3`

| # | Result | Note |
| --- | --- | --- |
| F1 | Correct, one precision. | `WriteBatch.replace` appends a successor with a new id and `supersedes_id`. `ActionSchemaRecord.intention_ids` is `goal_ids = [all active intentions of changes.view()]` **after** this cycle's goal changes. It names exact versions, but it is every active goal at commit, not the goals the decision named. Pass 3 shows them as "the goals that held when the action was chosen", not as "the goal this action served". |
| F2 | Correct. | `appraise.yaml` and `goals.yaml` show `active_goals` text only; no action is shown with its goal. |
| F3 | Correct. | `RecentMove(action, utterance, status)` in `situational_guidance.py`. |
| F4 | Correct. | `InteractionDecision.guidance_disposition: GuidanceDisposition`, `guidance_reason`; the receipt is a `StateUpdate` feature with `interaction_receipt` metadata. |
| F5 | Correct. | `_open_expectations` filters `provenance.source_ref == "main_cognition"`; both docstrings name A19G. |
| F6 | Correct. | `pending_expectations(view)` in `InnerThoughtLoop.prepare_input`; every pending expectation of the request is a **governing** (current-head) source of the private grant (`inner_thought/host.py::run_inquiry`, `governing_ids`). This matters for step 4. |
| F7 | Correct. | `check` rates against every stimulus text and passes every stimulus id as `observation_ids`; `relevant_expectations` embeds every stimulus. |
| F8 | Correct. | `stage_expectations` never sets `ExpectationRecord.goal_ids`. |
| F9 | Correct. | `find_intention` reads active goals; the second `complete` reports `missing`. |
| F10 | Correct. | `ALLOWED_CHANGES = ("retain", "add", "complete", "reprioritize", "defer", "refuse")`. |
| F11 | Not reproduced; reported only. | As stated. |
| F12 | Correct. | `validate_graph_step` has no original-source rule; `stage_inquiry_step` raises `ValueError`; `handle_batch` catches `ModelProposalError`, `ModelFormatError` only. |
| F17 | Correct, and wider than stated. | See the Pass 1 tables. `capture_graph_outcome` also runs on the host, outside the integrate stage's loop, and raises `ValueError` for model-only mistakes. A `ValueError` there is not caught by `handle_batch`, so it crashes the private module and ends the application. |

Additional findings by reading:

- **F18.** The integrate stage's loop has an attempt cap: `for attempt in range(2)` in
  `stages/integrate/steps.py::run`. After two rejected answers it raises `IntegrationError`;
  `handle_batch` discards the opportunity (thought step already paid). This is a cap on a model
  failure (failure rule, points 2 and 3). Reported in step 5; not changed in Pass 1.
- **F19.** Private recall widens the grant without derivation closure: `recall_memory` captures
  `[match.knoxel_id, *match.source_ids]` with `capture_source`, not `capture_inquiry_grant`. If a
  model cites a recalled derived record whose parents are outside that set, `source_roots` in
  `stage_inquiry_step` raises `ValueError` ("outside its captured grant"). Not reproduced. Closing it
  correctly changes the grant, which is serialized in the thought prompt, so it is reported and not
  fixed (Pass 1 step 0 rule).

### Step 3 — paths that can present one observed event to main cognition again

| Path | What main receives | Same record id? | Observed role? |
| --- | --- | --- | --- |
| Critical requeue (`MainCogLoop._handle_priority_batch`, `replay`) | The same `user.input` envelopes. Only an uncommitted batch is interrupted (`_active_batch_committed` guard). | Yes | `user_report` |
| Wait flush (`CognitiveControl._flush_deferred`) | The same envelopes with fresh envelope ids. | Yes | as original |
| `interaction.reconsider` (`InteractionWorkflow.publish_output`, `resume`) | The interaction receipt (`StateUpdate`, provenance Self, no `memory_kind`). | No, a new receipt id | `derived`/`background`: not observed |
| `interaction.resume` | Republishes `user.input` for interaction sources that no main cycle claimed. | Yes | `user_report` |
| `entity.action_reconsider` (`_stage_entity_reconsideration`, `resume_entity_actions`) | A new `StateUpdate` marker. | No | `derived`: not observed |
| Task reports (`workflow.event`) | `_stimulus_from` creates a **new** `WorldEvent` record per envelope (cached only for one handled batch). | **No: a re-sent event gets a new id** | `runtime_event`: observed |
| Restart: `dispatch_recovery.replay_intents`, `reply_recovery` | Action requests and task replies, not observations. The task's terminal event reaches main as a new `workflow.event`. | — | — |
| Restart: `inner_thought.resume` | Unclaimed thoughts and graph findings. | Yes | `interpretation`: not observed |
| A private finding that cites the event (`thought.created`) | The finding. | No | `interpretation`: not observed |
| `cognition.wake` | A new `WorldEvent` "N seconds passed…" (`memory_kind=executed_event`). | No | `runtime_event`: observed, and it is new evidence |

Consequence for Pass 2: the counted-id rule (D3 b) and the observed-role rule (D3 a) together close
every path except a **re-sent task event**, which arrives as a new record with the same meaning.
That residual is reported, not closed in B07.

### Step 4 — the two refusal paths that D2 makes more frequent

**(a) Main commits a check for an expectation the private loop replaced meanwhile.**
`publish_output` calls `changes.expect_revision(outcome.record_id, outcome.revision)` and
`apply_expectation_outcomes` calls `changes.replace(record)`; the store's head guard
(`record_id in batch._superseded and record_id not in self._all_knoxels`) raises `GhostConflict`.
`_handle_awake_batch` emits `cycle.conflict` and reruns the cycle on a fresh view, so the stale
expectation is no longer offered. That is a recorded stale result followed by fresh state. One
limit: after **three** conflicts in a row the conflict is raised and the main module crashes
(`attempt == 3: raise`). It is internal contention, not an external failure, so it is reported, not
changed.

**(b) The private loop projects a resolution for an expectation main replaced meanwhile.**
The private grant carries every pending expectation and every active goal as a governing
(current-head) source. `PrivateInquiryOwner.accept_step` commits the graph root with
`expect_inquiry_grant` (all grant sources) and then `ensure_inner_turn`. Both commits fail with
`GhostConflict` when any governing source got a successor. `run_inquiry` turns that into a
`PrivateGraphResult` summary ("requires fresh admission"). **But the next opportunity does not read
fresh state:** the active inquiry keeps its admission grant, and `stage_inquiry_step` forbids
refreshing its original fingerprints. Every later step is refused again on the same stale grant,
after a paid thought call, and the inquiry keeps the private slot until a Critical message displaces
it.

Reproduced on a real store, no model: `.runtime/beta-b07/pass0/repro_stale_governing.py`, output
`repro_stale_governing.txt`: three step attempts after a governing source got a successor, three
`GhostConflict` refusals ("Stale source meanings: [3]"), and the same episode still holds the slot.

This is already reachable today: main replaces a goal on every goal change and a main expectation
on every partial check, and both are governing sources of every private grant. D2 adds the
private-origin expectations to what main replaces. A second, rarer form: if a graph root committed
and the process stopped before its inner turn, `InnerThoughtLoop.resume` repeats `ensure_inner_turn`
at the start of every `handle_batch`, outside its `try`, so a permanently refused projection crashes
the private module on every start.

**Stop and ask (handoff section 6 and Pass 0 step 4).** Pass 2 is not started. See "Owner decision
needed" at the end of this file.

### Step 5 — attempt caps and timeouts on external operations

Full table: `.runtime/beta-b07/pass0/caps-and-timeouts.md` (read-only audit; I checked A3, B5 and the
module-crash chain in `pm/host/application.py::_run_loop` myself). Corrections to the handoff's
starting point: `complete_text` has no caller in `workflows/dialogue_story/steps.py`; its only caller
is `phases/model.py::complete_prose` (`attempts=10`), and empty answers already fail after two tries
inside `pm/llm/stage_model.py::complete_chat_with_retry`.

Transport failures (connection, 429, 5xx, connect timeout) retry without limit everywhere.
`simulation/client.py` has no inference or readiness deadline. These end a long run on an external
or model failure:

| # | Where | What ends |
| --- | --- | --- |
| A1 | `simulation/agents.py::Agents.call` (actor, review, judge) via `complete_chat_with_retry` | 2 tries for an empty answer, 1 for any other `ModelError` (`finish_reason=length`, provider error, 4xx). The attempt fails. |
| A2 | `simulation/engine.py::RunEngine._stage` | One bad director or actor answer (empty message, unknown fact id, wrong delivery) is not asked again. The attempt fails. |
| A3 | `inner_thought/host.py::handle_batch` | Any `ModelError` from a private stage propagates; module crash; the participant exits. |
| A4 | `public_speech/public_dialogue.py::_handle_priority_batch` | A plain `ModelError` (for example `finish_reason=length`) crashes the module. |
| A5 | `pm/memory/reconsolidation/model_calls.py::_send` | 4 calls, then `ModelError` and shutdown. Only when the companion routes to reconsolidation. |
| A6 | `simulation/client.py::PMProcess.listen` | websockets default `open_timeout=10`, no reconnect. |
| A7 | `simulation/client.py::post/state` | No deadline, but no retry on a transport error to the local participant. |
| A8 | pm-coder `PermanentProviderError` | Authentication or billing rejection; correctly not retried. |
| A9 | `pm/runs/workflow_runner.py::_render_state_md` | 5 tries on a Windows file lock, then crash. |

Caps that do not end the run but lose work: the integrate loop (F18, 2 answers), `generate_reviewed`
(2 generations), the codelet agent (3 sessions of 600 s each), and the main-cycle `GhostConflict`
cap of 3 (internal). Nothing was changed in this pass.

### Step 6 — memory

`.runtime/beta-b07/pass0/memory.txt`, captured 2026-11-15T09:29:46Z:

- Physical 127.9 GB, 61.4 GB free. Commit limit 127.9 GB (no page file), **9.2 GB commit free**.
- `basalt` 69.6 GB commit (owner process; not touched).
- The review server on 8095 (`python -m simulation`, pid 7256): **4.7 GB commit**. That is the size
  of one harness or participant Python process at rest.
- A stale `unittest` run from 2026-11-13 07:32 (pids 25892/12484,
  `tests.unit.workflows.inner_thought.test_inquiry_records … cognitive_control.test_wait`) holds
  **4.8 GB** commit. `test_wait` retries remote embeddings forever, so it never ends. It is probably
  left over from an earlier agent session. I did not stop it: it is not mine to stop without the
  owner.

Consequence, **for the owner now:** run 1 needs a controller plus one participant (about 2 × 5 GB
at start, growing with the store; the month store reached a 1.3 GB pickle). Run 2 needs a controller
plus two participants. With 9.2 GB free neither final run fits without the owner freeing memory
(for example the stale unittest process, the 8095 review server while no review is open, or memory
held by other owner applications). The Pass 5 short comparison (one participant) needs the same.

### Step 7 — why the interrupted month run stopped

Full notes: `.runtime/beta-b07/pass0/month-run-shutdown.md`. **Cause: the PC was powered off** from
the Start menu at 2026-11-06 01:22:43 local (System log, User32 1074, `Explorer.EXE` "power off",
then 6006 and Kernel-Power 109). The last participant log line is 1.7 s earlier, a chat request that
never returned. No crash, no `MemoryError`, no time limit (`ExecutionTimeLimit PT0S`). The monitor
`long-run.py` had already died at about 2026-11-05 18:48 local, 4 s after the owner returned to the
PC (Kernel-Power 566), with task result `0xC000013A` (console control event): its console window was
probably closed. The dashboard survived because it runs with `CREATE_NO_WINDOW`. Nothing wrote a
receipt, so `state.json` still says `running`.

### Baseline checks at `cd641e3`

`.runtime/beta-b07/pass0/baseline-checks.txt` and `main-cognition-baseline.txt`:
inner_thought 273 OK, focus_management 19 OK, entity_context 36 OK, architecture 7 OK, pyright 0
errors. Main cognition: `test_alpha_clamps`, `test_dispatch_recovery`, `test_phase_rules` OK;
`test_cognition_wiring` (11 errors), `test_context_recovery`, `test_context_rendering`,
`test_game_choices` (1 error each), `test_output_recovery` (5 errors) red as known;
`test_affect`, `test_codelet_registry`, `test_cognition_state`, `test_decisions`, `test_workspace`
did not finish in 300 s (they run phases against an endpoint and retry). Public speech: all OK except
`test_attribution` (1 error) and `test_speech_review` (1 failure), as known.

1. **Changed:** no product code. New record and evidence under `.runtime/beta-b07/pass0/`.
2. **Why:** handoff Pass 0.
3. **Verified:** reading, one real-store reproduction, Windows event log (read-only), memory counters.
4. **Not verified:** F11 (not reproduced, as instructed).
5. **Documentation:** this record.
6. **Open concerns:** step 4 (b) needs an owner decision before Pass 2; memory does not fit either final run.

## Pass 1 — D1: close every gap between the retry loop's checks and the store owner's checks

### Step 0 — rule-by-rule comparison

**Pair 1: `validate_graph_step` (in the thought step's loop) against `stage_inquiry_step` (store owner).**
`stages/thought/steps.py::run` passes `validate_graph_step` to `complete_json` as `validate`;
`pm.llm.structured.complete_json` asks again in a `while True` loop, starts a fresh conversation
after nine rejections, and stops only with `RepeatedRejectionError` (two complete conversations
with one identical problem). No cap was added.

| `raise` in `stage_inquiry_step` | Model answer alone? | In-loop twin |
| --- | --- | --- |
| non-finite `active_seconds`, aware `now_utc` | no (host timing) | none, stays hard |
| `GhostConflict`: not the current receipt parent | no (concurrency) | none |
| parent not `active`; step, time, or stall bound used | no (host admits only inside bounds) | none |
| observer, reality, or Critical epoch replaced; original fingerprints refreshed | no | none |
| `expect_inquiry_grant` scope or observer | no | none |
| operation is not the admitted `next_operation` | yes | yes |
| unknown `next_operation` (`validate_operation`) | yes | yes |
| relation target not an accepted node | yes | yes |
| continuation without a relation | yes | yes |
| alternatives in a non-generative operation | yes | yes |
| relation target outside the grant | no (node ids are always in the execution grant) | covered by the node rule |
| `source_roots`: evidence outside the grant | yes | yes (evidence ⊆ grant) |
| `source_roots`: a granted record whose parents are outside the grant | yes, only for recalled derived records (F19) | **no; reported, not closed** (the correct fix changes the grant, which the prompt serializes) |
| `source_roots`: lineage cycle or foreign scope | no (grant capture rejects both) | none |
| compared node without its relation | yes | yes |
| `new_evidence` without inspected ids, or inspected ⊄ evidence | yes | yes |
| **`new_evidence` that inspects an internal derivation** | **yes (the live crash)** | **closed** |
| discrimination in a non-discriminatory operation, or fewer than two nodes | yes | yes |
| finding or completion with generative mode or pending branches | yes | yes |

**Pair 2: `validate_integration` (in the integrate stage's loop) against `capture_graph_outcome` and
`stage_graph_outcome` (store owners).** `capture_graph_outcome` runs on the host after the stage,
so its `ValueError` is not caught by `handle_batch` either. Both are listed.

| `raise` | Model answer alone? | In-loop twin before | Now |
| --- | --- | --- | --- |
| capture: disclosure mode not `candid`/`withhold` | yes (`mode: str`) | none | closed |
| capture: unknown expectation action | yes (`action: str`) | none | closed |
| capture: open without anticipation or outcome | yes | yes | — |
| capture: resolve without target or observations; unsupported status | yes | yes | — |
| capture: `goal_ids` outside the grant | yes | none | closed (goals must be supplied goals) |
| capture: `observation_ids` outside the grant on `open`/`none` | yes | none (checked only for `resolve`) | closed |
| capture: focus ids, expectation id, revision sources outside the grant | yes | yes (⊆ citable ids, ∈ pending, revision rules) | — |
| capture: resolution without current expectation authority | no (every pending expectation is a governing source) | none | — |
| stage: retained payload differs; invalid action at projection | no | none | — |
| stage: goal id is not an `Intention` | yes | none | closed |
| stage: target not a pending expectation in the operation's scope | no (the request holds only pending expectations of the observer's scope) | yes (∈ pending) | — |
| stage: target without current authority | no | none | — |
| stage: observation in another scope or with a non-observed role | yes | none | closed |

Assumption the new rules rely on, stated in code: the citable ids of the integration request are a
subset of the execution grant. A breach is our bug and still raises in `capture_graph_outcome`.

### Steps 1 to 4 — the change

1. **Changed:**
   - `pm/memory/evidence.py`: `OBSERVED_EVIDENCE_ROLES`, the one named set; `stage_graph_outcome`
     reads it instead of its inline set (Pass 2 step 4 reuses it for main cognition).
   - `workflows/inner_thought/nominations.py::original_source_ids`: the store owner's rule
     (`source_roots(view, [id], ...) == (id,)`); a derived record is skipped before the walk.
   - `GraphThoughtRequest.original_source_ids` (required). `graph_thought_request` fills it; the
     host recomputes it against the final execution grant after perspective and recall widen it.
     Excluded from the echoed envelope in `stages/thought/steps.py`.
   - `validate_graph_step`: a `new_evidence` step that inspects a non-original id raises
     `ModelProposalError` naming the wrong ids, listing the original ids, and saying to use
     `progress.kind='none'` otherwise.
   - `GraphOutcomeRules` (`schemas.py`): observed citable ids and supplied goal ids;
     `graph_outcomes.graph_outcome_rules` builds it; `pipeline.validate_graph_outcome_rules`
     checks disclosure mode, expectation action, supplied goals, citable observations, and observed
     roles for a resolution. The graph path sends it as `graph_outcome_rules` beside the request;
     the integrate stage applies it in its loop, and the host applies it again after the stage.
   - Untouched: `stage_inquiry_step`, `capture_graph_outcome`, `stage_graph_outcome` checks, and
     the `except` in `InnerThoughtLoop.handle_batch`.
2. **Why:** D1 and the owner's answer that every escaping model mistake is closed.
3. **Verified:**
   - New real-store checks: `test_graph_host.GraphHostTests.test_earlier_thought_claimed_as_new_evidence_is_corrected_inside_the_loop`
     (in-loop `ModelProposalError`; `stage_inquiry_step` still raises the original `ValueError`;
     the original source passes), and in `test_graph_outcomes`:
     `test_thought_cited_as_observation_is_corrected_inside_the_loop`,
     `test_goal_that_is_not_a_supplied_intention_is_corrected_inside_the_loop`,
     `test_unknown_disclosure_mode_and_expectation_action_are_corrected_inside_the_loop`
     (each: in-loop `IntegrationError`, store owner still `ValueError`).
   - inner_thought 278 OK, focus_management 19 OK, entity_context 36 OK, architecture 7 OK,
     pyright 0 errors.
   - Replays of fresh copies (`.runtime/replay/b07-p1-*`, output `.runtime/beta-b07/pass1/replay-*.txt`):
     private-continuity-20261114T180018Z, deliberate-entry-20261114T180133Z,
     deliberate-reflection-20261114T180458Z, private-dwelling-20261114T222749Z (3 samples),
     perspective-morning-20261115T081258Z (3 samples): every sample `ok`, exact.
   - Live Part C, real OpenRouter, run 1 at `15c270b`:
     `.runtime/beta-b06/integration/20261115T100117Z-9a2d0d57-part-c`: **passed**. Noticing 32
     with `thought_activity` claimed by main tick 15; main then dispatched a `dialogue_story` action
     and committed a wait; 43 requests.
   - Live Part C run 2: `.runtime/beta-b06/integration/20261115T100606Z-31600d42-part-c`: **passed**.
     Noticing 26 claimed by main tick 11; then a `dialogue_story` action and a wait; 34 requests.
   - Neither run raised `ValueError` from a store owner. Logs: `.runtime/beta-b07/pass1/live-part-c-run*.log`.
4. **Not verified:** in neither live run did the model make one of the closed mistakes (no prompt
   holds a new correction text), so the live correction path itself is shown by the real-store
   checks only. F19 is open. Gate cost: 43 + 34 requests; not in a manifest. The pass-1 Part C runs
   still write under `.runtime/beta-b06/integration/` because that is the gate's evidence root.
5. **Documentation:** docstrings of the new functions and fields; the `stage_graph_outcome` and
   `OBSERVED_EVIDENCE_ROLES` comments. No manual page (Pass 7).
6. **Open concerns:**
   - F18: the integrate loop still stops after two rejected answers. Closing it as the owner's rule
     asks means calling `validate_integration` and `validate_graph_outcome_rules` as the `validate`
     callback of `complete_json` (endless, with `RepeatedRejectionError`). That changes the text of
     the correction turn ("This proposal was rejected: …" becomes the shared "The answer failed JSON
     or schema validation: …"), so it waits for the owner.
   - Run 1's main summary contains a leaked model token `<｜begin▁of▁sentence｜>` inside the
     subjective-experience text. Model output, not caused by B07; reported.
   Commit `15c270b`.

## Owner decision needed before Pass 2

**Q1 — an active private inquiry whose grant went stale (Pass 0 step 4 b).** Every private grant
holds all active goals and all pending expectations as current-head sources. When main cognition
replaces one of them, the active inquiry can never commit another step: each attempt costs a
thought call and is refused, and the slot stays held until a Critical message. D2 makes this more
frequent, but it already happens on any goal change or partial expectation check. Options:

- **A (recommended).** Before a step's model call, the private loop compares the active
  inquiry's current-head sources with the store. If one has a successor, it closes the inquiry with
  a terminal receipt that names the changed sources, without a model call, and frees the slot. The
  question returns to shared attention as a new episode under a fresh grant, exactly like the
  `displaced` path you chose for Critical displacement. A refused step commit ends the same way at
  the next opportunity. No refreshed fingerprint, no retry of the same refused commit.
- **B.** A continuation may refresh only its current-head sources to their new heads and keeps
  every historical fingerprint. The inquiry continues under the new goal or expectation version.
  This relaxes the rule "a continuation cannot refresh its original source fingerprints".
- Either way, the rare restart form (a graph root committed, its inner turn refused for good) needs
  a rule too. Recommended: commit the inner turn and the finding, record the refused expectation
  effect as stale on the turn, and do not apply it. Main cognition still hears the finding.

**Q2 — F18.** May the integrate stage use the endless `complete_json` correction loop (changes the
correction turn's wording only)?

**Q3 — memory.** 9.2 GB commit is free. One participant process is about 5 GB at rest, so neither
final run fits. A stale `unittest` process from 2026-11-13 (pids 25892/12484, 4.8 GB, retrying
remote embeddings forever) and the 8095 review server (4.7 GB) are candidates to stop. I did not
stop either.

## Owner answers, 2026-11-15

> continue with your recommendation and dont worry about memory

Read as: Q1 option A with the recommended restart rule; Q2 yes (it follows the owner's own failure
rule); Q3 memory is not a blocker. The owner stopped the stale `unittest` processes himself.

On memory, for the record: "commit free" is the reservation headroom Windows gives programs (RAM
plus page file). Without a page file the limit is the 128 GB of RAM, and Basalt reserves 69.6 GB it
mostly does not touch, so a `MemoryError` can appear with 60 GB of RAM physically free. A
system-managed page file would raise that limit. Nothing was changed.

## Q1 — close an inquiry whose grant went stale (commit `5ff50b8`)

1. **Changed:** `pm/memory/dependencies.py::changed_dependencies` (the commit's source rule applied
   to a view); `InquiryDisplacement.cause` (`critical_epoch` or `governing_source_changed`, default
   the old meaning, not serialized when default) and `changed_source_ids`;
   `inquiry.stage_inquiry_outdated` and `OUTDATED_REASON`; history validation for both causes;
   `PrivateInquiryOwner.close_outdated`, called by `run_graph_opportunity` after
   `displace_superseded` and by `admit_deliberate_action`; `ensure_inner_turn` projects a graph
   root whose outcome sources changed after acceptance without the expectation effect and stages a
   `stale_graph_outcome` receipt in the turn; `focus_management` words the notice per cause
   (`MentalOutcome.displacement_cause`).
2. **Why:** Pass 0 step 4 (b): every later step paid for a model call and was refused.
3. **Verified:** `tests/unit/workflows/inner_thought/test_outdated_inquiry.py`, 6 real-store checks
   (closed before any model call, question restarts under a fresh grant and executes, current work
   untouched, parked work closed, notice wording, late projection keeps the finding without the
   effect and is idempotent). inner_thought 284, focus_management 19, architecture 7, pyright 0.
   Five recordings replay exactly (`.runtime/beta-b07/q1/`).
4. **Not verified:** the closure in a live run.
5. **Documentation:** module docstrings of `inquiry.py`, `graph_host.py`, `focus_management/host.py`.
6. **Open concerns:** see Pass 2 point 6.

## Q2 — the integrate stage asks again without a cap (commit `3f07013`)

`validate_integration` and `validate_graph_outcome_rules` run as the `validate` callback of
`complete_json`. The contract test that encoded the old two-answer cap now checks three wrong
answers followed by a valid one. inner_thought 284 OK, contract inner_thought 4 OK, five recordings
exact (`.runtime/beta-b07/q2/`).

## Pass 2 — D2 and D3 (commit `7430cc2`)

1. **Changed:**
   - `ExpectationRecord.counted_observation_ids` (default empty, described as "no history recorded").
   - `OpenExpectation.origin` and `counted_observation_ids` (required).
   - `main_cognition/expectations.py`: `offered_expectations` (every pending expectation of the real,
     in-world scope; unknown producer raises; private ones are scoped to the observer, which is that
     scope), `is_observed` (shared `OBSERVED_EVIDENCE_ROLES` plus the scope), `uncounted_stimuli`;
     `relevant_expectations` and `check` rate each expectation only against its uncounted observed
     stimuli, grouped, one call per group; the A19G docstrings are replaced.
   - `CycleRequest.observed_stimulus_ids`, filled by the host.
   - `apply_expectation_outcomes` stages every checked record, appends the rated ids (also when
     nothing moved), and raises on a repeated count; `stage_expectations` carries counted ids to a
     superseded successor and sets `goal_ids` from the action's `intention_ids` (F8).
   - `inner_thought/expectations.PRIVATE_SOURCE_REF` names the private producer.
2. **Why:** D2, D3.
3. **Verified:** `tests/unit/workflows/main_cognition/test_expectation_ownership.py`, 7 real-store
   checks: both origins offered, the imagined/meta-frame one not; a thought stimulus moves nothing
   and asks for no embedding; the same observation moves once and the second pass needs no call; an
   unmoved rating still counts; a private expectation resolved by main once, the late private
   resolution refused by the head guard and reported by `changed_dependencies`; an old record reads
   as no history; complete twice gives `completed`, `missing`. inner_thought, focus_management,
   entity_context, interaction suites OK; the main-cognition modules green at `cd641e3` still green,
   the red ones red with the same causes; architecture OK; pyright 0. Five recordings exact.
   `workflows/main_cognition/expectations.py` has 24 strict errors (mostly older), so it is not
   added to `pyrightconfig.json`.
4. **Not verified:** a live main cycle (Pass 4).
5. **Documentation:** module docstring of `main_cognition/expectations.py`, field docs.
6. **Open concerns:** every main check now replaces each checked expectation (to record the count).
   Every pending expectation is a governing source of every private grant, so after each checked
   observation an active private inquiry is closed by Q1 and its question re-enters attention. This
   is correct by the current grant rule but may churn private thinking. A narrower grant (govern only
   what a step cites) is a design change for the owner.

## Pass 3 — D4, the interval review (commit `32e2d08`)

1. **Changed:** new `workflows/main_cognition/interval_review.py` (`GoverningIntention`,
   `IntervalMove` holding the shared `RecentMove`, `GuidanceSuspension`, `IntervalReview`,
   `goal_family_heads`, `build_interval_review`, `render_interval_review`);
   `situational_guidance.recent_move` extracted from `build_situational_guidance` (one projection,
   two readers); `GuidanceDisposition` moved to `situational_guidance.py` (an import cycle
   `cycle_work → interval_review → interaction.schemas → cycle_work` otherwise);
   `CycleState.interval_reviewed_through` (None, not serialized while unset);
   `CycleRequest.interval_review`; the host builds it from the previous accepted main state's
   frontier and `_accepted_state` stores the new frontier; appraise and goals render it, with the
   three section-16 rules in `appraise.yaml` and `goals.yaml`. Names checked against the glossary
   and naming rules: no conflicts.
2. **Why:** D4.
3. **Verified:** `tests/unit/workflows/main_cognition/test_interval_review.py`, 6 real-store checks
   (version 1 text after a goal change, frontier moves and no repeat, refused cycle keeps the
   frontier, suspension once with its reason, a state without a frontier reviews nothing, clean
   restart keeps it). The entity-context fixture that drives `publish_output` now supplies a review.
   Suites as in Pass 2, pyright 0 with `interval_review.py` added, five recordings exact.
4. **Not verified:** live (Pass 4). The token budget rule relies on the existing measurement in
   `complete_json`: an oversized initial prompt raises `RequiredContextTooLarge`, which ends the cycle
   visibly. No live sample was large enough to trigger it.
5. **Documentation:** module docstring of `interval_review.py`; prompt file headers.
6. **Open concerns:** an action's `intention_ids` are all active goals at its commit (F1), so the
   review says "the goals that held", not "the goal it served". The status is read at review time:
   a reply committed just before the cycle can show "chosen but not out yet" and is not reviewed again.

## Pass 4 — live gate (commits `test(live): add the B07 interval appraisal gate` and the review script)

1. **Changed:** `tests/replay/situations.py::story_evening` (labelled synthetic history: two accepted
   main turns with frontiers, a fast-path reply chosen under goal version 1 while the turn that
   completed version 1 was calculating, one open main-origin expectation); new
   `tests/live/test_interval_appraisal.py` (evidence root `.runtime/beta-b07/integration/`); new
   `dev/work_items/beta_b07_review.py`.
2. **Why:** handoff Pass 4. A real flow reviews an action under an older goal version only when the
   fast path commits it while a main turn that changes the goal is calculating; the situation
   commits that order. A synchronous gate cannot interleave the two loops.
3. **Verified (real OpenRouter, `deepseek/deepseek-v4.1-flash-1017`):**
   - **Part B passed**, attempt 1, `20261115T112207Z-51fc5b35-part-b-attempt-1`: main tick 3's
     three appraise calls and its goals call contain
     `[action 8] … goal then [4]: Finish the winter-sea story with Jesse this week (since completed)`,
     that goal is no longer active, and the new state's `interval_reviewed_through` is 15 (> action 8).
     21 requests.
   - **Part C passed**, attempt 1, `20261115T113235Z-31e98205-part-c-attempt-1`: after the
     conflicting wish the promise (goal 11) is still active. 36 requests.
   - **Part A NOT RUN**, 0 of 3: in every attempt the real turn moved the expectation to `supported`
     with the confirmation counted once (`counted_observation_ids: [13]`), but the goals call did not
     complete "Agree with Jesse on the evening for finishing the winter-sea story"; it added a new
     goal instead (attempt 1: "Keep Jesse thinking about the winter sea between now and Thursday at
     eight"). A valid choice, not an error (failure rule 5). The second presentation was therefore
     not run. Decision records: `part-a-first.json` and the goals prompt logs in each attempt folder.
   - Provider-reported cost from the prompt logs' usage blocks: Part B $0.009165, Part A
     $0.010620 + $0.011375 + $0.011314, Part C $0.018195; with the Pass 1 Part C runs ($0.029519 +
     $0.023726) the B07 gate spend is **$0.113914**.
   - Owner review, registered: http://127.0.0.1:8095/#capture/b07-attribution-live-20261115 and
     http://127.0.0.1:8095/#capture/b07-promise-live-20261115. The review server was not running
     after the restart; start it to open them.
4. **Not verified:** a live goal completion and the live second presentation of one observation
   (SC-13's second half). The counted-once rule is shown only by the real-store checks of Pass 2.
5. **Documentation:** gate and review-script docstrings.
6. **Open concerns:** `capture_gwt` cannot capture a main turn that names a superseded record (every
   turn that checks an expectation does); the review script captures such turns as record
   collections and names the missing ids. The model keeps adding goals instead of completing reached
   ones; that is the D7-style selection question for the owner.

## Pass 5 — run readiness

### Step 1 — short comparison (paused at my 45-minute limit)

Template `test_data/simulation/b07-sleep-comparison.json` (commit `7c1d833`): the preserved Spamuel
home, one evening scene at 2026-06-02 22:30 (Jesse chats and says good night, at least three turns)
and one morning scene at 2026-06-03 07:30. Run `b07-sleep-comparison-20261115`, real harness on
port 8097 bound to `0.0.0.0`, `--openrouter-only`, `PM_COMPLETION_CACHE=0`. Started 11:41:13Z,
paused through the harness's own `/api/runs/<id>/pause` at 12:25:40Z, state `paused`.

| Measure | Comparison run (at pause) | Baseline `20261104-eventful-month` |
| --- | --- | --- |
| Scenes done | 1 of 2 (the night scene) | 4 of 90 |
| Jesse messages | 3 | 16 |
| Spamuel messages | 6 | 115 |
| Unsolicited messages during announced absence | none yet: the run was paused at 22:35, five minutes after the good night | 14 after Jesse's last message, every 10 world-minutes |
| Main cycles (goals calls) | 7 | — |
| Logged model calls / provider-reported cost | 526 / $0.242412 | — |

Observed: before Jesse's first message the preserved history was mid-exchange, and Spamuel sent
three unsolicited messages at 22:00:30, 22:10:30, and 22:20:30, each a new variant of the same bit
("one dish from the finish line…", "six minutes of silence…", "court adjourned, i'm going dark").
That is the baseline's 10-minute pattern (SC-18 failure condition), seen before any sleep
announcement. After Jesse's good night Spamuel answered once ("go sleep. and don't dream about the
sink"). The night itself, which SC-01 and SC-18 are about, was not reached.

Summary: `.runtime/beta-b07/pass5/run-summaries.txt`. Resume from the dashboard
(http://127.0.0.1:8097/#run/b07-sleep-comparison-20261115) or with
`POST /api/runs/b07-sleep-comparison-20261115/resume`; attempt 2 freezes the current source tree.

### Step 2 — which beta modules ran (prompt logs of the comparison participant)

| Module | Evidence | Ran |
| --- | --- | --- |
| Early triage | 20 triage prompts | yes |
| Ordinary interaction (fast path) | 3 prompts with `guidance_disposition` | yes |
| Entity context | 6 prompts with `entity_extraction` | yes |
| Private inquiry | 46 nomination prompts, 2 inquiry-operation steps | yes |
| Interval review | 17 prompts contain "since last taking stock" | yes |

### Step 3 — run 1 (D8)

Run `beta-final-month-20261115` created from `test_data/simulation/eventful-month.json`
(the exact 90-scene plan of the interrupted baseline), state `ready`, 0 attempts. Launcher and
monitor copied to `.runtime/beta-final/` and changed: `final-run.py` takes run id, template, and port,
writes `<run-id>-status.json` every 30 s and `<run-id>-exit.json` on every exit (normal end,
exception with traceback, console close or Ctrl+C via `SIGBREAK`/`SIGINT`/`SIGTERM`, interpreter
exit); `launch-final.ps1` writes `<run-id>-launcher-exit.json` with the exit code and uses
`$ErrorActionPreference = "Continue"` so a stderr line cannot stop it. No automatic restart.

### Step 4 — caps that end a long run on an external failure

Changed, one commit each:

| Finding | Commit | Now |
| --- | --- | --- |
| A1 scenario agents | `d92edfc` | a `ModelError` keeps a `retrying` receipt and asks again, 30 s doubling to 300 s, no cap, until it works or the run is paused |
| A3 private loop | `ce038d2` | a `ModelError` is a `thought.rejected` diagnostic; the next timer opportunity asks again from accepted state |
| A4 public speech | `2a4d158` | any `ModelError` takes the existing reply deferral; no crash, no repeated tool action |
| A2 director and actor | `3898294` | a blank actor message and a judgment that cites unknown or out-of-stage facts, missing deliveries, or wrong times are re-asked inside the call with a message that names the wrong value |

Reported, not changed: A5 reconsolidation (4 calls, only when routed there), A6 websocket open
timeout (10 s, no reconnect), A7 no retry on a transport error to a local participant process
(a local process fault; loud failure kept), A8 permanent provider rejection (correctly not
retried), A9 five tries on a Windows file lock, the main-cycle `GhostConflict` cap of 3, the codelet
agent's 600 s × 3 sessions, `generate_reviewed`'s two generations, and
`MINECRAFT_REQUEST_SECONDS = 660` (body path; see the Minecraft prerequisites). None of the changed
paths was exercised by a real failure in this session; they are verified by reading, import, the
simulation unit suite (13 OK), the inner-thought suite, three public-speech modules, and pyright.

### Step 5 — run 2 (D9): Jesse with Spamuel 1 and Spamuel alpha

**What Jesse's actor sees** (`RunEngine._stage`): for a scene that targets one participant, the
actor's conversation is every timeline row whose `target` or `source` is that participant. So in a
Spamuel 1 scene Jesse sees her own messages to Spamuel 1 and everything Spamuel 1 said, including
what Spamuel 1 said that was routed to Spamuel alpha; she does not see Spamuel alpha's messages. The
speech rows carry `speaker` from the participant's own store, which is "Spamuel" for both; the
actor profile is per participant (`actors/<id>.json`) and both are forked from the same home. So the
harness gives Jesse both conversations, each in its own scenes. A cross-participant message is
routed only while a stage settles (`collect(route=True)`).

Generator `dev/work_items/beta_final_two_spamuels.py` (commit `af161c9`) writes
`test_data/simulation/b07-two-spamuels-check.json` (the two naming scenes) and
`beta-final-two-spamuels.json` (the naming scenes plus the first seven days of the month plan, 21
scenes, targets alternating, "Spamuel" in scene text replaced by the target's name). Both
participants fork `~/.pm/private-machine/spamuel-simulation-a`; neither store is renamed. Jesse's
naming intent asks her to treat both as equally real and to speak of each only by his name. The
connections are `spamuel-1 → spamuel-alpha` named "Spamuel 1" and the reverse named "Spamuel
alpha". The generator refuses any template that contains the four words the owner ruled out; none
appears in either file.

Check run `b07-two-spamuels-check-20261115`, started 11:49:50Z, paused 12:25:40Z at 07:20:30
world time, before the first scene (07:30). Both participants spoke on their own from the
preserved history (Spamuel alpha 2, Spamuel 1 3 messages). Three messages are queued in transit
between them (`pending`); none crossed, because no stage has settled yet. Whether Jesse's naming
lands and whether a message crosses is **not observed**. Logged calls: 249 + 260, provider-reported
$0.093995 + $0.101576.

Run `beta-final-two-spamuels-20261115` created, state `ready`, 0 attempts. The assumption that the
two Spamuels speak to each other is still the owner's to confirm; without it, remove the two
`edges` from the template and recreate the run.

### Step 6 — run 3 (D10)

`dev/work_items/beta-final-minecraft-prerequisites.md` (commit `58b7503`). Not startable.

### Step 7 — start commands, dashboards, memory

Start in a hidden window, so closing a console cannot end the monitor:

```powershell
Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File C:\source\pm\private-machine\.runtime\beta-final\launch-final.ps1 -RunId beta-final-month-20261115 -Template C:\source\pm\private-machine\test_data\simulation\eventful-month.json -Port 8098'
Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File C:\source\pm\private-machine\.runtime\beta-final\launch-final.ps1 -RunId beta-final-two-spamuels-20261115 -Template C:\source\pm\private-machine\test_data\simulation\beta-final-two-spamuels.json -Port 8099'
```

| Run | Dashboard | Processes | Memory |
| --- | --- | --- | --- |
| 1, month | http://127.0.0.1:8098/#run/beta-final-month-20261115 | monitor, dashboard, 1 participant | dashboard 6.8 GB commit measured with two paused runs loaded; one participant about 5 GB at start, growing with its store (the baseline's pickle reached 1.3 GB) |
| 2, two Spamuels | http://127.0.0.1:8099/#run/beta-final-two-spamuels-20261115 | monitor, dashboard, 2 participants | dashboard about 7 GB plus about 5 GB per participant at start |

Free commit at 12:31Z: 100.0 GB (Basalt not running after the restart). Receipts:
`.runtime/beta-final/<run-id>-status.json`, `-exit.json`, `-launcher-exit.json`, `-launch.log`,
`-dashboard.log`. The 8097 dashboard is still running and owns the two paused check runs.

1. **Changed:** see the steps.
2. **Why:** handoff Pass 5.
3. **Verified:** templates validate; runs 1 and 2 are `ready` with 0 attempts; the comparison and
   check runs ran on the real harness until paused.
4. **Not verified:** the night of the comparison (SC-01, SC-18), a message crossing between the
   two Spamuels, Jesse's naming, the launcher (not executed, because it starts a final run), and
   every changed retry path under a real failure.
5. **Documentation:** this record.
6. **Open concerns:** the 10-minute unsolicited pattern appeared again before the sleep scene; the
   two Spamuels may answer each other without end when routing is on (each speech is routed to the
   other); the final runs freeze the source at their first attempt.

## Pass 6 — optional

- **D7, selection of mental work.** Decision records of the six failed attempts: in every one the
  decide call chose to answer Jesse publicly from memory (for example "answer the test in one cold,
  exact breath … workday six … weekends not before nine"), so the route had no mental choice to map;
  the route answers were `dialogue_story` replies. The decide prompt lists real choices (silence,
  waiting, a refusal, a joke …) and names no private thinking. One prompt change (`19fc954`) added
  "thinking something through privately before saying anything" to that list. Measured: ToM Part B
  0 of 3 messages (`.runtime/beta-b06/integration/20261115T120519Z-e3fb900c-part-b`, $0.062038) and
  private Part C 0 of 3 (`.runtime/beta-b05/integration/20261115T121440Z-c5bfa192-part-c`,
  $0.071362); the decide answers still answered publicly. The line had no measured effect and was
  removed again (`ac4d115`). A model that declines is a recorded result. Likely cause (inferred): the
  situation's memories already hold the answer, so the character has nothing to work out.
- **D5, resume:** not done (time).
- **Early-triage rejected answer:** not investigated (time).

## Pass 7 — documentation and close-out (commit `e05b169`)

1. **Changed:** `docs/subsystems/cognition-state.md` (new "Interval review" with a data-flow diagram,
   frontier, attribution, and the Part B observation); `decisions-and-routing.md` (the check owner,
   observed stimuli only, counted observations, limits now true); `private-loop.md` (expectation
   ownership, "A changed grant source", failure table rows); glossary entries interval review, review
   frontier, observed evidence, counted observation. Project-documentation skill read first.
2. **Why:** the manual must describe what Passes 4 and 5 observed, with status labels.
3. **Verified:** `uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict`
   exit 0 in 290.36 s (`.runtime/beta-b07/mkdocs-build.log`). In `site/`: the cognition-state page
   renders the section and the mermaid block; its API links resolve to
   `reference/workflows/main_cognition/interval_review.html` (generated); the private-loop anchor
   `#a-changed-grant-source` resolves; the glossary holds the four entries.
4. **Not verified:** none for documentation.
5. **Documentation:** as above.
6. **Open concerns:** none beyond the passes.

## Status

B07 Passes 0 to 5 and 7 are done and reported; Pass 6 D7 was measured and reverted, D5 and the
triage item were not done. Live results: Pass 1 Part C passed twice; Pass 4 Part B and Part C
passed, Part A NOT RUN (0 of 3: the goal was never completed). The beta milestone is not finished:
the final runs and the owner's review are its evidence, so card 7 stays `doing`.

B07 provider-reported spend in this session: gates $0.113914, D7 $0.133400, comparison and check
runs $0.437983; total about $0.69. Commits since the handoff: `d8c6444` … `ac4d115`, none pushed.

## Pass 8 — start the three final runs in parallel (owner instruction, 2026-11-15)

The owner started the runs and asked for changes so all three can run at the same time without
affecting each other. Three owner decisions replaced earlier recommendations:

- **D9 clarified.** Jesse is in the loop. The setup is that Jesse made a second Spamuel; she talks with
  both, and both may also write to each other. For Spamuel 1 the other Spamuel is an ordinary entity:
  either may stay quiet, answer later, or ignore the other, and neither is obliged to reply. The
  two-Spamuels template now says that in its brief, director guidance, and actor guidance. The two
  directed edges stay.
- **D10 withdrawn.** Run 3 is startable. `C:/source/minecraft-server-real` serves the world and
  `C:/source/pm/pm-minecraft` runs the bodies.
- **Private churn** stays as it is for now. The unfixed run-enders stay as they are for now.

### Changed

- `simulation/templates.py`: `TemplateParticipant.minecraft_mcp_url` (validated HTTP endpoint, one
  endpoint per participant); `create_run` maps template participants to the engine's selection type.
- `simulation/engine.py`: new `ParticipantSelection(id, home, minecraft_mcp_url)` replaces the
  untyped participant mapping in `RunEngine.create`; `participant_body_endpoint` reads the controller
  document (an old run has no field and reads as "no body"); the body endpoint is passed to each
  `PMProcess`; `start` and `restore` refuse to restore a run that owns a body.
- `simulation/client.py`: `PMProcess` carries the endpoint and sends it in `/control/start`.
- `simulation/participant.py`: `/control/start` accepts `minecraft_mcp_url`, writes it as the
  `minecraft` section of that participant's own `runtime/model-config.yaml`, and starts the
  application with `enable_minecraft=True`; the world poll stays off (module docstring explains the
  virtual-clock reason). Start options are type-checked at that boundary.
- `simulation/server.py`: the browser keeps each participant's body endpoint through template
  validation, run forking, and raw-plan run creation.
- `test_data/simulation/beta-final-two-spamuels.json`: brief, director guidance, actor guidance, and
  analysis context now state the owner's premise. The plan and its 23 scenes are unchanged.
- New `test_data/simulation/beta-final-minecraft.json`: Jesse with Spamuel 1 and Spamuel alpha, each
  with his own body in one shared world; two days, six scenes, `minimum_turns_per_scene` 3, the same
  two directed connections.
- New `dev/tools/minecraft_final_run.py`: `up` copies the server template into
  `.runtime/beta-final/minecraft/server`, patches its ports, starts the world, writes one `.env` per
  body (MCP 6777/6778, body HTTP 6779/6780, viewer 6781/6782, separate agent homes), starts both
  pm-minecraft processes, waits for a live observation from each, and supervises them until the
  server ends. `status` and `stop` read its receipt; `stop` saves the world over RCON first. It uses
  the `OperatingSystem` boundary and refuses to start beside another listener.
- `docs/subsystems/simulation.md`: new "Minecraft bodies" section; the restore paragraph notes the
  body exception. `simulation/__init__.py`: the canonical overview describes bodies and that limit.

### Why

The owner wants the three end-to-end runs going at once as the beta milestone evidence. They need a
body path that names one endpoint per participant, a world that two bots can share, and no shared
ports, homes, leases, or run folders between runs.

### Verified

- `python -m simulation.templates validate` on all three templates: valid.
- `tests/unit/simulation` 13 OK (including the coordinated-checkpoint restore); `tests.unit.test_architecture`
  7 OK; `npx --no-install pyright --project pyrightconfig.json`: 0 errors.
- Real Minecraft stack, started 2026-11-15T13:17Z: `dev/tools/minecraft_final_run.py up` generated a
  fresh world, both players joined it (`spamuel_1`, `spamuel_alpha`, entity ids 325 and 326), and both
  MCP endpoints answered a full tool probe. Receipt: `.runtime/beta-final/minecraft/stack.json`.
- Runs created and started in parallel at 13:21Z: `beta-final-month-20261115` (port 8098, 1
  participant, 90 scenes), `beta-final-two-spamuels-20261115` (port 8099, 2 participants, 23 scenes,
  recreated from the updated template), `beta-final-minecraft-20261115` (port 8100, 2 participants
  with bodies at 6777/6778, 6 scenes). All three report `running`, each with its own dashboard and
  monitor, and the month run had already delivered its first Jesse message at 08:00 world time.
- The body configuration is real, not assumed: each body participant's
  `runtime/model-config.yaml` holds its own `minecraft.mcp_url`, and those participants loaded and
  ran real cognition afterwards, which requires the startup probe of that endpoint to have succeeded.
- The main-cognition route prompt in run 3 lists `minecraft` and `minecraft_project` with their real
  descriptions, so the body is offered as an action, not merely configured.
- The first run-3 attempt found two real defects, both fixed in `b624684`:
  - the participant wrote `minecraft.mcp_url` into its model configuration, but the read-only
    capability and hook modules resolve the endpoint from `MINECRAFT_MCP_URL` in the process
    environment, so every dialogue prompt said "Minecraft connection right now: disconnected" while
    the body stood in the world. The participant now binds that variable from the same typed start
    option.
  - the stack tool's RCON client treated an empty login body as a rejected password; the protocol
    answers a rejection with request id `-1`. `stop` now checks the id. Live: the old client raised
    against the real server, and the new one ran `list` and returned `Stopping the server`.
- Run 3 was then restarted: the attempt was stopped through the harness, its receipts were kept in
  `.runtime/beta-final/restart-1/`, the disposable world was recreated, the run folder was recreated,
  and the new attempt's dialogue prompt reads **"Minecraft connection right now: connected"**
  (`participants/spamuel-1/logs/prompts`, attempt 1 of the second launch).

### Pace (measured 13:52:53Z to 14:02:53Z, three runs at once)

| Run | Virtual time gained | Delivered rows | Note |
| --- | --- | --- | --- |
| month | 0 min (clock held at 08:00) | 6 to 8 | inside a scene: the actor kept sending turns |
| two Spamuels | 0.5 min | 3 to 3 | both participants in long main cycles |
| Minecraft | 20 min | 1 to 4 | startup cycles already finished |

The clock only moves while every participant is idle, so the rate is governed by how long a
participant's main cycle runs, not by scene spacing. At the two-Spamuels rate a seven-day plan needs
days of real time. That is the existing product pace under a virtual clock, not a regression from
this pass, and nothing here changes it.

### Not verified

- No final run had reached a second scene when this record was written; the runs were still in their
  first long main cycle at 07:00:30. Nothing here claims dialogue quality, body use, or completion.
- The body path in a delivered scene (a companion choosing a Minecraft action and the world changing)
  was not observed yet.
- The restore refusal, the two-Spamuels premise in live dialogue, and the RCON stop were not exercised
  by a real failure or by a finished run.
- The harness still does not capture world effects in an attempt report; the body evidence is the
  pm-minecraft agent homes, logs, and the world itself.

### Documentation

`docs/subsystems/simulation.md`, `simulation/__init__.py`, and this record. The Minecraft
prerequisites file is updated with what closed and what still applies.

### Open concerns

- A body run cannot be resumed after a failure; the owner creates a new run ID. That is now enforced.
- The virtual clock and real world time still disagree: a scene advance can take real minutes while
  world time stands still, and a long body task delays the next scene.
- The month run may reach its later scenes long after the other two finish; nothing here bounds that.
