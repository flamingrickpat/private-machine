# Beta handoff — B07: interval appraisal, then run readiness

Written 2026-11-15 after the B06 audit, at revision `ce3780d` on branch `beta`.

Goal for today, set by the owner: finish the beta engineering work and make the final
end-to-end runs startable. B07 is the last beta slice. It has two halves:

1. **Interval appraisal** (milestone section 16, SC-04, SC-13). Small code change. Most of the
   foundation already exists; see the audit below.
2. **Run readiness.** Prepare the three final runs so the owner can start them. Do not start them.

Read `AGENTS.md` first. Its rules are mandatory. The ones that matter most here: real stores and
real endpoints only, no mocks, typed contracts, fail loud on a broken internal contract, and no
default on a required internal field.

**The design is accepted.** The owner answered every decision on 2026-11-15; the answer is recorded
at the end of section 1. This file replaces the usual separate proposal, because the change is
small. A decision that section 1 does not hold still goes to the owner first.

## 1. Owner decisions (answered 2026-11-15)

| ID | Decision | Accepted answer | Consequence |
| --- | --- | --- | --- |
| D1 | The `stage_inquiry_step` crash (2 of 5 live Part C runs). A model claims `new_evidence` for a derived record and the store owner raises `ValueError`. | Two parts, by the owner's failure rule below. **Our bug:** the check that runs inside the model's retry loop does not know a rule that the store owner enforces. Fix that. **The model's mistake:** the wrong answer is then asked again, without an attempt cap, until the answer is valid. The store owner's check stays unchanged. | Nothing is hidden and nothing is swallowed. A wrong model claim never reaches the store owner, so it cannot stop a private opportunity. Required before any long run. |
| D2 | Who checks expectations. Today main cognition checks only expectations it recorded itself. Private expectations are checked only if the private loop happens to pick them. | Main cognition checks every pending expectation in the accepted reality, whatever its origin. The private loop keeps opening expectations and may still resolve one with observed evidence. | No category of expectation stays unevaluated. One expectation is still resolved once: a replacement of a record that is no longer the head is refused at commit. |
| D3 | Counting an observation once. | (a) Only observed records can move an expectation: a private thought about an event cannot. (b) Each expectation keeps the observations it already counted, and the same observation never moves it again. | A thought that restates an event no longer counts as new evidence. This changes behaviour: today a `thought.created` stimulus can move an expectation. |
| D4 | The interval record. | A read-only projection, built for each main cycle: the actions since the last accepted review, their delivery status, the intention versions that governed them, and any guidance the fast path set aside. One new typed field on `CycleState` holds the review frontier. | Appraisal and goals read what was actually done and under which goal. Each action is reviewed in exactly one accepted main cycle. One persistence change with an honest default. |
| D5 | A deferred goal has no way back. `defer` hides it from the goals prompt and no change kind resumes it. | Add a `resume` change kind and show goals that were put aside. Horizons, activation conditions, and avoidance constraints go to gamma. | A deferred promise can be taken up again with its history. Optional pass; drop it if time is short. |
| D6 | Part C of the B06 live gate differs from the B06 plan's wording. It reports NOT RUN for an inference wait only when no claim came first. | Keep the gate as it is now. | A turn that claims the noticing and then waits is a pass. |
| D7 | Main cognition never selected `theory_of_mind` or `focus_management` by itself (0 of 6 attempts in B05 and B06). | One time-boxed look at the catalogue text and the route rule: 60 minutes, at most one prompt change, measured by the existing Part B gates. | Without it the long runs will probably never exercise deliberate mental work. A model that still declines is a recorded result. |
| D8 | Final run 1, Jesse–Spamuel. | Reuse the exact plan of `test_data/simulation/eventful-month.json` under a new run id. | Directly comparable with the interrupted baseline `20261104-eventful-month`. |
| D9 | Final run 2, "Jesse–Spamuel 1–Spamuel alpha". | Owner's definition: one Jesse with two Spamuels, named **Spamuel 1** and **Spamuel alpha**, "so no one feels like the cheap copy". Both run the current beta code. "alpha" is a name, not the alpha baseline. | Two real participants in one run. Neither is labelled a copy, a clone, or a second instance anywhere a participant can read. See Pass 5 for one open assumption. |
| D10 | Final run 3, Spamuel–Spamuel–Minecraft. The harness starts every participant with `enable_minecraft=False`. | Today: write a prerequisite report only. Harness support for bodies is its own item. | Run 3 is not startable today. |

**Owner answer, 2026-11-15, in the owner's words:**

> d1? the idea is to always fix bugs, not hide errors, that we can fix ourselves. and try in a loop
> until the end of time if its outside our bounds. model generating crap? try again until it
> generates correct thing. model down? try again until we can reconnect. no timeouts, simple endless
> loops handling external errors forever and it works again. otherwise yes. d9 is a run where jesse
> has 2 spamuels. spamuel 1 and and spamuel alpha (so no one feels like the cheap copy).

Read as: D2 to D8 and D10 accepted as recommended. D1 follows the failure rule. D9 is defined.

Asked whether every model mistake that escapes the retry loop's check should be closed, not only
the one that crashed, the owner answered the same day:

> if something is escaping the chekc that should be fixed, yes

So Pass 1 closes every such gap it finds, including F17.

### The owner's failure rule (applies to every pass)

1. **A failure we can fix ourselves is a bug. Fix its cause.** Do not catch it, log it, and
   continue. Do not turn it into a default or an empty result.
2. **A failure outside our control is retried in a plain endless loop until it works.** A model
   answer that fails its schema or its validator: ask again. An endpoint that is down: reconnect
   again. Use a delay between attempts and show honest status. The loop ends on success or on the
   owner's cancellation, and on nothing else.
3. **No timeouts and no attempt caps on those loops.** Do not add one. Report any you find.
4. One existing stop stays, because it marks our bug and not the model's:
   `pm.llm.structured.RepeatedRejectionError`. It fires when one identical correction text fails
   through two complete repair conversations. That means our prompt or our validator message is
   defective. When you see it, fix the message. Do not raise or remove the limit.
5. This rule is about failures. It is not about choices. A valid model answer that selects a
   different action than a gate hoped for is a result, not an error. A gate does not loop until the
   model happens to choose what we want; it runs its stated attempts and reports the rate.
6. A retry never repeats an external effect whose outcome is unknown (`AGENTS.md` section 8.3).

## 2. Where things are

- Branch `beta`, head `ce3780d`. This handoff and one board note are the only uncommitted changes.
  Commit them first: `docs(beta): hand off B07 interval appraisal and run readiness`. Nothing is
  pushed. Do not push.
- B01 to B06 are implemented. Card 7 is `doing` and stays `doing` (see section 8).
- Package version is `1.0.2`. It does not change.
- Interim budget: $25 on OpenRouter for development checks. Recorded spend so far is about $0.50.
  Several gate runs and interrupted requests have no recorded cost, so the true figure is higher
  and unknown. The final runs have a separate, uncapped authorization.

The records are the authority, not this file:

- `dev/work_items/beta.md`: the plan. B07 is the row "interval appraisal and sustained acceptance".
- `dev/work_items/beta-capability-audit.md`, section "B07". Written against the alpha baseline; one
  of its statements is out of date (finding F1 below).
- `dev/work_items/beta-b06.md`, section "Audit and corrections": the open defect and gate results.
- `docs/continuous-character-milestone.md`: section 16 (interval evaluation), section 6
  (intentions), SC-04, SC-13, section 20 (acceptance evidence), section 24 (completion).
- `dev/work_items/beta-handoff-b06-20261114.md`: the list "Known defects outside B06" still
  applies. Do not fix those suites here, except where a pass below names one.
- `dev/work_items/simulation-month.md`: the harness and the interrupted month run.

## 3. Audit (by reading the code at `ce3780d`; nothing here was reproduced)

Check each finding in Pass 0. Report any that is wrong before you build on it.

| # | Finding | Evidence |
| --- | --- | --- |
| F1 | Historical attribution already exists in the data. `WriteBatch.replace` appends a successor with a **new record id** and sets `supersedes_id`; the old payload stays readable. Every action records the intention ids that governed it. Those ids name exact historical versions. No "governing revision" field is needed. | `pm/memory/write_batch.py::replace`; `workflows/main_cognition/host.py` (`ActionSchemaRecord(intention_ids=list(goal_ids))`, lineage `intention_ids`); `workflows/interaction/host.py::accept_selection`. |
| F2 | Nothing reads that attribution. The appraise and goals prompts do not show an action with the goal that governed it. | `workflows/main_cognition/phases/appraise.py`, `phases/goals.py`, `phases/prompts/goals.yaml`. |
| F3 | A recent-move projection exists: action, utterance, and status `selected / committed / delivered / failed`. Reuse it; do not write a second one. | `workflows/public_speech/situational_guidance.py::RecentMove`, `build_situational_guidance`. |
| F4 | The fast path records when it set accepted guidance aside, with a reason. | `workflows/interaction/schemas.py` (`guidance_disposition`, `guidance_reason`); `interaction/host.py::publish_output`. |
| F5 | Main cognition offers only expectations with `provenance.source_ref == "main_cognition"`. The docstrings still say "until A19G moves it onto this cycle". | `main_cognition/host.py::_open_expectations`; `main_cognition/expectations.py` module docstring. |
| F6 | The private loop reads every pending expectation, including main's, and may resolve any pending one in its scope. Its resolution requires observed evidence roles. | `inner_thought/host.py` (`pending_expectations(view)`); `inner_thought/graph_outcomes.py::stage_graph_outcome`. |
| F7 | The main check rates expectations against **all** stimuli of the cycle, including `thought.created`. Nothing records which observation already moved an expectation; `observation_ids` is set only at resolution. | `main_cognition/expectations.py::check`, `outcome_of`; `host.py::apply_expectation_outcomes`. |
| F8 | Main-recorded expectations link their action and stimuli, but `goal_ids` stays empty. | `main_cognition/host.py::stage_expectations`. |
| F9 | Completing a goal twice is already safe: `find_intention` returns only active goals, so the second `complete` is `missing`. No affect code reads a completion. The numeric duplicate risk is in expectations (F7). | `main_cognition/cycle_commits.py::apply_goal_changes`. |
| F10 | `defer` sets `active=False`. The goals prompt shows active goals only, and `ALLOWED_CHANGES` has no resume. A later `add` with the same key creates a new, unlinked intention. | `cycle_commits.py`; `phases/goals.py::ALLOWED_CHANGES`. |
| F11 | Each partial check replaces the expectation, and the successor takes the current tick. `recency` reads that tick, so a partly fulfilled expectation never ages. Not reproduced. Report only; do not change it. | `host.py::apply_expectation_outcomes`; `expectations.py::recency`. |
| F12 | The D1 crash: the child validator `validate_graph_step` does not know which supplied sources are original. `stage_inquiry_step` does and raises `ValueError`. `InnerThoughtLoop.handle_batch` survives `ModelProposalError` and `ModelFormatError` only. | `inner_thought/graph_execution.py::validate_graph_step`; `inner_thought/inquiry.py` (the `new_evidence` branch); `inner_thought/host.py::handle_batch`; `.runtime/beta-b06/integration/20261115T083614Z-62e8fc86-part-c/failure.json`. |
| F17 | A second gap of the same kind as F12, found by reading, not reproduced. `validate_integration` runs inside the integrate stage's retry loop. It accepts a resolution whose observation ids are merely known to the request, and it does not check `expectation.goal_ids` at all. `stage_graph_outcome` then raises `ValueError` for an observation with a non-observed evidence role, for a goal id that is not an intention, and for a target in another scope. A model answer alone can trigger the first two. | `inner_thought/pipeline.py::validate_integration`; `inner_thought/stages/integrate/steps.py` (the `validate_integration` call); `inner_thought/graph_outcomes.py::stage_graph_outcome`; the caller in `inner_thought/graph_host.py` (the `inquiry-gwt:` batch). |
| F13 | The harness supports several participants and directed speech connections. Planning and the actor profile use `participants[0]` only. | `simulation/templates.py` (`SimulationTemplate`, `generate_plan`, `create_run`). |
| F14 | Every harness participant starts with `enable_minecraft=False, world_poll_seconds=None`. | `simulation/participant.py::_start`. |
| F15 | The interrupted month run left no failure. `state.json` says `running`, `failures` is empty, the last status has `"error": null`. Its launcher and monitor are kept and reusable. | `.runtime/simulation/runs/20261104-eventful-month/state.json`; `.runtime/simulation-workspace-20261104/` (`launch-long.ps1`, `long-run.py`, `prepare-month.py`, `scheduled-task.xml`, `long-launch.log`). |
| F16 | Port 8080 is in use by the owner's Basalt application. Port 8095 serves the review UI. The kanban server (8767) is not running. | `netstat`, 2026-11-15. |

## 4. Passes

Work in this order. Commit after each pass with a scoped Conventional Commit subject. If a pass
needs more than twice its time box, stop and report.

### Pass 0 — verify and surface owner prerequisites (45 minutes, no product code, no model)

1. Read the owner's answer and the failure rule in section 1.
2. Check findings F1 to F12 against the code. Report differences.
3. Name every path that can present one observed event to main cognition a second time. Check at
   least: Critical requeue, `interaction.reconsider`, `entity.action_reconsider`, task reports,
   restart recovery (`dispatch_recovery.py`, `public_speech/reply_recovery.py`), and a private
   finding that cites the event. Write the list into the B07 record. Pass 2 needs it.
4. Trace the two refusal paths that D2 makes more frequent, and write down what each loop does
   next. (a) Main cognition commits a check for an expectation that the private loop replaced
   meanwhile (`publish_output`, `apply_expectation_outcomes`). (b) The private loop projects a
   resolution for an expectation that main cognition replaced meanwhile
   (`graph_host.py`, the `inquiry-gwt:` batch around `stage_graph_outcome`). Note that the old
   payload still reads `pending`, so only the head guard at commit sees the change. A refusal must
   end as a recorded stale result, and the loop must then read fresh state. A crash here, or a loop
   that repeats the same refused commit on the same stale state, is our bug: stop and ask.
5. List every attempt cap and every timeout on an external operation in the path a long run uses:
   model calls, embeddings, the harness client, and the participant controls. Start with
   `workflows/main_cognition/model_json.py::complete_text` (`attempts`) and its caller in
   `workflows/dialogue_story/steps.py`. For each one, state what ends the run when it fires. The
   harness client already has no deadline (`simulation/client.py`). Fix nothing in this pass.
6. Measure memory before any long-run preparation. On 2026-11-15 the machine had 10.7 GB of commit
   charge free and one harness process pair needed about 11 GB. Report the free commit charge and
   the size of one participant process. **Do not stop or reconfigure Basalt or any owner process.**
   If a run cannot fit, tell the owner now, not at the end of the day.
7. Read the last hour of the interrupted month run's logs (F15) for 30 minutes at most. State the
   shutdown cause, or state that it is still unknown.
8. Create the work record `dev/work_items/beta-b07.md`. Use the six-part report of `AGENTS.md`
   section 19 for each pass, as `beta-b06.md` does.

### Pass 1 — D1: close every gap between the retry loop's checks and the store owner's checks (90 minutes)

The defect is ours: `validate_graph_step` runs inside the model's retry loop
(`stages/thought/steps.py` passes it to `complete_json` as `validate`), and it lacks one rule that
`stage_inquiry_step` enforces. So a model mistake escaped the loop and reached the store owner.

0. First compare two pairs of functions rule by rule:
   - `validate_graph_step` (in the loop) against `stage_inquiry_step` (store owner);
   - `validate_integration` (in the loop) against `stage_graph_outcome` (store owner), finding F17.

   For every `raise` in the store-owner function, state whether a model answer alone can trigger
   it and whether the in-loop check has the same rule. Write both tables into the B07 record. By
   reading, the first pair has one gap (the original-source rule) in the block that checks the
   step; confirm it for the whole function. The second pair has at least two (F17).

   The owner's rule for the result: every rule a model answer alone can break must also be in the
   in-loop check, with a correction message that says what to change. A `raise` that only our own
   state can trigger stays a hard failure and gets no in-loop twin. If a rule needs facts the child
   does not have, the host supplies them as a typed request field that stays out of the prompt,
   as step 1 does. If closing a gap would change a prompt, stop and report before you do it.

   Steps 1 to 3 close the first pair's gap. Close the F17 gaps the same way: the request already
   carries the candidates; the observed-role rule needs the evidence role of each citable id, and
   the goal rule needs the supplied intention ids. Use the one named role constant that Pass 2
   step 4 introduces, so the in-loop check, the private store check, and the main check agree.
1. Add `original_source_ids: tuple[int, ...]` to `GraphThoughtRequest`. The host fills it in
   `run_graph_opportunity` with the same rule the store owner uses: a supplied source is original
   when `source_roots(view, [source_id], scope=..., permitted_ids=...) == (source_id,)`.
2. Keep the field out of the echoed envelope in `stages/thought/steps.py` (the existing `exclude`
   set), so no prompt byte changes.
3. In `validate_graph_step`, a `new_evidence` step whose `inspected_source_ids` are not all original
   raises `ModelProposalError`. The message names the wrong id, lists the original source ids, and
   says to use `progress.kind='none'` otherwise.
4. Do not touch the check in `stage_inquiry_step`. Do not add a `try` around it and do not widen
   the `except` in `InnerThoughtLoop.handle_batch`.
5. Add no attempt cap. `complete_json` already asks again without a limit. Confirm that by reading
   and say so in the record.
6. Verify:
   - One real-store check in `tests/unit/workflows/inner_thought/`: a step that cites an earlier
     thought as inspected evidence gets `ModelProposalError` from `validate_graph_step`, and the
     store owner still raises `ValueError` when called directly.
   - The same pair of assertions for each F17 gap: a resolution that cites a thought as its
     observation, and an expectation that names a goal id which is not an intention.
   - Inner-thought unit suite, architecture test, pyright.
   - Replay fresh copies of all five current recordings. Expected: exact. If one diverges, report
     the first differing call and stop.
   - Run live Part C twice. Record requests and results. One model-quality `NOT RUN` is a result,
     not a failure. A `ValueError` from `stage_inquiry_step` is a failure: stop.

### Pass 2 — D2 and D3: one owner for the expectation check, each observation counted once (90 minutes)

1. `ExpectationRecord.counted_observation_ids: list[int]`, default empty. The description says:
   observed records that already moved this expectation; empty on a record older than B07 means no
   check history was recorded.
2. `OpenExpectation` gains `origin` (`"main_cognition"` or `"private_loop"`) and
   `counted_observation_ids`.
3. `_open_expectations` offers every pending expectation in the accepted reality scope. Check how
   private graph expectations set `scope` (`graph_outcomes.py`) and do not offer an expectation from
   an imagined or foreign frame.
4. Observed stimuli only. The private path already has the rule: the evidence roles allowed in
   `graph_outcomes.stage_graph_outcome`. Move that set to one named constant and use it in both
   places. A stimulus with another role does not enter the main check.
5. For each expectation, the stimuli to rate are the observed stimuli of the cycle that are not in
   its counted list. An expectation with none is not checked. Group expectations by that stimulus
   list and make one `check` call for each group. In the normal case there is one group, so the call
   count does not change.
6. `apply_expectation_outcomes` appends the rated stimulus ids to `counted_observation_ids`,
   including when the status stays `pending`.
7. `stage_expectations` sets `goal_ids` to the intention ids that govern the action (F8).
8. Update the two docstrings that still name A19G (F5) to state the new rule.
9. Verify with real-store checks, no model, in `tests/unit/workflows/main_cognition/`:
   - the same observation presented twice moves fulfillment once, and the second pass makes no call;
   - a `thought.created` stimulus that cites the observation does not move the expectation;
   - a private-origin pending expectation is offered to main and resolved once; the private loop's
     later resolution of the same record is refused by the head guard and ends as the stale result
     that Pass 0 step 4 described;
   - an older record without the new field loads and reads as "no history";
   - completing a goal twice gives `completed` then `missing`, with one successor (F9).
   - Then: affected unit suites, architecture test, pyright. Add
     `workflows/main_cognition/expectations.py` to `pyrightconfig.json` only if it is strict-clean
     without casts; otherwise report the count and leave the config.

### Pass 3 — D4: the interval review (120 minutes)

New module `workflows/main_cognition/interval_review.py`. Its conceptual parent is the main
cognition host; start its docstring with a pointer to that overview. Check every new name against
`docs/concepts/glossary.md` and `docs/concepts/naming.md` first.

Proposed contracts. These are a starting point: use the repository's real types where one exists.

```python
class GoverningIntention(BaseModel):
    intention_id: int   # the exact historical version the action recorded
    text: str           # the goal as it read then
    later_state: Literal["current", "changed", "completed", "deferred", "refused"]
    current_id: int | None   # active head of the same goal now, or None

class IntervalMove(BaseModel):
    action_id: int
    origin: Literal["main", "interaction", "task_reply"]
    purpose: str
    status: Literal["unknown", "selected", "committed", "delivered", "failed"]  # RecentMove's words
    utterance_id: int | None
    governing: list[GoverningIntention]
    expectation_ids: list[int]   # ExpectationRecord.created_from_action_id == action_id

class GuidanceSuspension(BaseModel):
    receipt_id: int
    disposition: GuidanceDisposition   # the existing enum, never a string copy
    reason: str
    source_ids: list[int]

class IntervalReview(BaseModel):
    after_record_id: int      # the frontier this review started from
    through_record_id: int    # the highest record id this review read
    moves: list[IntervalMove]
    suspensions: list[GuidanceSuspension]

def build_interval_review(view: Snapshot, *, after_record_id: int, ...) -> IntervalReview: ...
def render_interval_review(review: IntervalReview, ...) -> str: ...
```

Rules:

1. Frontier. `CycleState.interval_reviewed_through: int | None = None`. `None` means no accepted
   cycle recorded a review. The first cycle after the change then reviews nothing and sets the
   frontier to the highest record id in its request view. It must not review the whole history.
   Record ids only grow, so an action committed while a cycle calculates is read by the next cycle,
   and no action is read twice. A cycle that is not accepted leaves the frontier unchanged.
2. Family head. Find the current head of a goal by walking `supersedes_id` back from each head
   `Intention`. Do not add a store API for it.
3. Never show an action under a goal version that did not exist when the action was chosen.
4. `CycleRequest.interval_review: IntervalReview` carries the projection. Appraise and goals render
   it through `render_interval_review`. No record dictionary reaches a prompt.
5. Prompt text, in `appraise.yaml` and `goals.yaml`, as story prose in the style of those files:
   what the character did since it last took stock, with the goal that held then. Add the three
   rules of milestone section 16 in plain words: an asked question is not an answer; a sent request
   is not a finished task; something good after an action does not prove the action caused it.
6. The token budget rule of `AGENTS.md` section 12.2 applies. The review is mandatory evidence. If
   it does not fit, fail visibly.
7. Verify with real-store checks, no model:
   - an action chosen under goal version 1 still shows version 1's text after the goal changed;
   - an accepted cycle moves the frontier, and the next review does not repeat the action;
   - an unaccepted cycle does not move it;
   - a fast-path suspension appears once with its reason;
   - an old checkpoint state (`None`) starts at the current head and reviews nothing;
   - default clean restart keeps the frontier.
   - Then: main-cognition and public-speech unit suites that already pass, architecture test,
     pyright with the new module added to `pyrightconfig.json`.

### Pass 4 — live gate (60 minutes plus run time)

New `tests/live/test_interval_appraisal.py`. Model it on `tests/live/test_perspective_integration.py`:
real `build_app`, real store, real OpenRouter chat and embeddings, one application per attempt, an
evidence folder under `.runtime/beta-b07/integration/`, and `result.json`, `not-run.json`, or
`failure.json`. Never force an action, clear a wait, or edit a record to reach a pass.

- **Part A, SC-13.** Spamuel holds a goal and an open expectation. A real Jesse message fulfils it.
  Pass: one accepted main turn completes the goal with the message as evidence and moves the
  expectation. Then a second main turn runs without new observed evidence (a wake, or a private
  finding about the same message). Pass: no second completion, no second expectation movement, no
  second expectation effect on the appraisal.
- **Part B, attribution.** An action is delivered under one goal. The goal then changes. Pass: the
  next accepted main turn's request shows the action under the earlier goal text.
- **Part C, SC-04.** A promise, then a wish that conflicts with it. Pass: the promise is still
  active, or it was changed with a recorded reason. Fail: it is gone with no record.

Three attempts at most for each part. This is a sample size for behaviour, not a retry cap on a
failure (failure rule, point 5). Inside each attempt every model call and every endpoint failure
still retries without limit. A part the model does not reach is `NOT RUN` with the kept decision
records and the rate. Then build one owner review package from a passed run with the review script
pattern of `dev/work_items/beta_b06_review.py`, register it, and give the direct URL.

### Pass 5 — run readiness (90 minutes; the owner starts the runs, not you)

1. **Short comparison first.** One exact plan: one day, two scenes, the preserved Spamuel history,
   the sleep situation of SC-01 and SC-18 (Jesse says good night, timers fire, Jesse returns).
   Run it through the real harness on a free port, bound to `0.0.0.0`, OpenRouter only, completion
   cache off. If it is still running after 45 minutes, pause it with the harness's own control and
   report the progress; that limit is on your time, not a timeout in the product. Report: Jesse messages, Spamuel messages,
   unsolicited messages during the announced absence, main cycles, and requests. Compare with the
   baseline (4 scenes, 16 Jesse messages, 115 Spamuel messages). This is the first evidence that
   the beta slices work together in one process.
2. Confirm from that run's telemetry that these beta modules ran: early triage, ordinary
   interaction, entity context, private inquiry, and the interval review. State any that never ran.
3. **Run 1 (D8).** Create the run from the month template under a new run id. Leave it `READY`,
   zero attempts. Copy the kept launcher and monitor (F15) into `.runtime/beta-final/`, with a free
   port (not 8080, not 8095). The launcher must write the exit code and any exception to a file
   when the process ends, because the last month run left nothing. Do not add an automatic restart:
   a process that dies is our bug to read, not an external failure to loop on.
4. If Pass 0 step 5 found a cap or timeout that would end a long run on an external failure, change
   it now to an endless retry with a delay and visible status. One commit for each. Report the
   rest.
5. **Run 2 (D9): Jesse with Spamuel 1 and Spamuel alpha.**
   - Two participants in one template, ids `spamuel-1` and `spamuel-alpha`, both forked from the
     same preserved Spamuel home. The harness forks a home without changing the source.
   - Do not rewrite either history and do not rename the persona inside a store. Each participant
     is Spamuel to itself. Jesse gives each its name in the first scene, in her own words, and the
     name then lives in each one's real memory. Write Jesse's private intent for that scene so that
     she treats both as equally real and never calls either one a copy.
   - No text that a participant can read may say copy, clone, duplicate, or second instance. That
     covers scene instructions that reach the actor's messages, connection names, and run titles.
   - **Assumption, not yet confirmed by the owner:** the two Spamuels can also speak to each other.
     That is two directed connections, shown to each receiver as `Spamuel 1` and `Spamuel alpha`.
     If the owner wants two separate conversations, remove the connections; nothing else changes.
   - Planning and the actor profile use the first participant only (F13). Supply an exact plan
     whose scenes name their target. Before you write it, read `simulation/engine.py` and state
     what Jesse's actor sees when two participants answer. If the harness cannot give Jesse both
     conversations, report that and stop; do not build a new harness feature without the owner.
   - Run a two-scene check of this template first (one scene for each target). Report whether both
     participants answered and whether a message crossed between them.
   - Memory: two participant processes and the controller. Use the Pass 0 measurement.
   - Leave the real run `READY`, zero attempts.
6. **Run 3 (D10).** Write `dev/work_items/beta-final-minecraft-prerequisites.md`: what the harness
   lacks (F14), what a second body needs, and which real server and accounts are required. No code.
7. Write the three start commands, the dashboard URLs, and the memory each run needs into
   `dev/work_items/beta-b07.md`.

### Pass 6 — optional, only with time left, in this order

- **D7, selection of mental work (60 minutes).** Read `_cognition_workflow_options` and `route.yaml`.
  Read the kept decision records of the six failed attempts first and state why the model chose
  otherwise. At most one prompt change. Measure with `test_part_b_main_cognition_selects_theory_of_mind`
  and `test_part_c_main_cognition_selects_deliberate_mental_work`, three attempts each.
- **D5, resume (90 minutes).** `resume` in `ALLOWED_CHANGES`; `apply_goal_changes` reactivates the
  deferred intention of that key as a successor with a `choice_history` entry; the goals prompt
  shows goals that were put aside. Real-store checks, then one live sample.
- **Early-triage rejected answer (45 minutes).** Every message costs one extra request: the first
  answer misses properties at `entity_extraction.claims[0]`. The B06 repair-message fix may now show
  the true cause. Fix it only if the cause is one prompt or schema mismatch. Otherwise report.

### Pass 7 — documentation and close-out (45 minutes)

1. Read the project-documentation skill before you edit a manual page. If it is missing, stop and
   tell the owner.
2. Manual pages describe only what Pass 4 and Pass 5 observed, with status labels:
   `docs/subsystems/cognition-state.md` (frontier, interval review), `decisions-and-routing.md`
   (what appraisal and goals now read), `private-loop.md` (expectation ownership), and the glossary
   for each new term.
3. `uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict`, then inspect
   the changed pages and the new API reference page in `site/`.
4. Finish `dev/work_items/beta-b07.md`. Add one dated note to card 7.

## 5. Guard rails

- Do not add a second expectation store, a second intention store, or a new memory store.
- Do not change `GwtTurnRecord`, the recovery-root rule, `stage_decision`, or the hard-stop policy.
- Do not implement gamma G01 items, semantic horizons, activation conditions, or avoidance
  constraints.
- Do not relax a validator, add a default, or lower a threshold to make a model answer pass. Keep
  the failed sample and report it.
- Do not start a month run, a three-party run, or a Minecraft run. Do not start a local model.
- Do not run the complete unit or contract suites.
- Do not replay an original recording in place. Copy the snapshot folder first.
- Do not stop, restart, or reconfigure Basalt or any process the owner runs.
- Bind every server to `0.0.0.0`.
- Set `PYTHONHASHSEED=0` in the shell before `python -m tests.replay record`. Without it the harness
  restarts in a child process, and that child ran out of memory on 2026-11-15.

## 6. Stop and ask the owner when

- a pass needs a decision that section 1 does not hold;
- you would need a timeout, an attempt cap, a default, or a catch-and-continue to get past a failure;
- a finding in section 3 is wrong in a way that changes the design;
- the frontier or the new record field cannot load an existing checkpoint without a migration;
- a live gate fails for a reason that is not clearly one model-quality sample;
- the same symptom needs a third patch;
- the machine cannot hold the processes a final run needs.

## 7. Commands

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/inner_thought -t . -b
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/focus_management -t . -b
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/entity_context -t . -b
.venv/Scripts/python.exe -m unittest tests.unit.workflows.inner_thought.test_expectations -b
.venv/Scripts/python.exe -m unittest tests.unit.workflows.main_cognition.test_decisions -b
.venv/Scripts/python.exe -m unittest tests.unit.workflows.public_speech.test_situational_guidance -b
.venv/Scripts/python.exe -m unittest tests.unit.test_architecture -b
npx --no-install pyright --project pyrightconfig.json

python -m tests.replay list
python -m tests.replay replay --snapshot .runtime/replay/<copy of the snapshot folder>

python -m unittest tests.live.test_perspective_integration.LivePerspectiveIntegrationTests.test_part_c_noticing_reaches_main_cognition
python -m unittest tests.live.test_perspective_integration.LivePerspectiveIntegrationTests.test_part_b_main_cognition_selects_theory_of_mind
python -m unittest tests.live.test_private_integration.LivePrivateIntegrationTests.test_part_c_main_cognition_selects_deliberate_mental_work
python -m unittest tests.live.test_interval_appraisal

python -m simulation.templates validate <template.json>
python -m simulation.templates create <template.json> --config config.yaml --run-id <id>
uv run --locked python -m simulation --config config.yaml --openrouter-only --port <free port>
uv run --locked python -m simulation.review register-package <file> --id <id> --title "<title>"
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

Check that a unit module is green at `ce3780d` before you count it as affected evidence. Several
main-cognition and public-speech modules are red for known reasons (B06 handoff, "Known defects").

## 8. What "finished" means, and what you may claim

- B07 engineering is finished when Passes 1 to 5 and 7 are done and reported, with every live part
  marked passed, `NOT RUN`, or failed from real evidence.
- The beta **milestone** is not finished by that. Milestone section 24 needs reviewable evidence
  that the mechanisms cooperate. That evidence is the final runs and the owner's review.
- So card 7 stays `doing`. Do not move it. The owner moves it after the runs.
- Report with the six parts of `AGENTS.md` section 19. Give gate cost as request counts and say
  that the cost is not in a manifest, unless you add usage receipts.

## 9. After B07

The owner starts the final runs. Gamma follows: `dev/work_items/gamma.md`, card 8.
