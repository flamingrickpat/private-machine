# B07: interval appraisal and run readiness

Status: Pass 0 and Pass 1 done, 2026-11-15. Pass 2 is stopped for an owner decision (Pass 0 step 4).
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
