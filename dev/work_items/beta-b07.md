# B07: interval appraisal and run readiness

Status: Passes 0 to 3 done, 2026-11-15; the owner answered Q1 to Q3 the same day (section "Owner answers").
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
