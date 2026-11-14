# Beta handoff — finish B05

Rewritten on 2026-11-14 after an audit and four owner decisions. This is the only handoff file.
It holds the current state and the remaining B05 work. Earlier versions are in git history (`7a93f59` and before).

Your task: finish B05 in four passes, in order. Each pass ends with one local commit.
Read `AGENTS.md` first. Its rules are mandatory. The ones that matter most here:
real stores and real endpoints only, no mocks, typed contracts, fail loud on a broken internal contract,
and no `.get(field, default)` on a required internal field.

## Where things are

- Branch `beta`. The tree is clean at the commit that holds this file. Nothing is pushed. Do not push.
- B01 to B04 are complete within their targeted scope. B05 is in close-out. B06 and B07 have not started.
- The beta card (7) stays `doing` until the owner closes it.
- Package version is `1.0.2`.

The records are the authority, not this file:

- `dev/work_items/beta-b05.md`: accepted design, every pass, limits, and verification. Its last section, "Audit and close-out decisions (2026-11-14)", holds the owner decisions.
- `dev/kanban/board.json`, card 7: one dated note per pass.
- `docs/subsystems/private-loop.md`, `docs/subsystems/deliberate-mental-work.md`, `docs/subsystems/reflection.md`, `docs/subsystems/introspection.md`: how the current system works.
- `docs/workflows/scenario-replay.md`: how to record and replay a scenario.
- `docs/continuous-character-milestone.md`: the target. Sections 8 to 11, 15, and 18.2 govern B05.

## What B05 is

The private mind proposes up to three questions (nomination). Attention selects at most one.
An inquiry then runs connected steps: a generative step proposes branches, a discriminatory step compares them.
It stops with a finding, an information gap, a wait, a pause, or an abandonment. A finding goes to main cognition for appraisal.
A selected mental action ("think about X") uses the same private slot. Ordinary input does not cancel private work. Critical input displaces it.

All of that is implemented and has run against real models in three small recorded scenarios.
The inquiry graph stores a step as an `InnerMonologue` feature with `metadata["loop"] == "inquiry"`.
The older linear path stores `metadata["loop"] == "inner"`.

## What the audit found

Probe and output: `.runtime/beta-b05/audit-20261114/probe.py` and `probe.txt` (ignored by git, on this machine). Real store, real `PrivateInquiryOwner`, no model.

Several readers still understand only the linear thought.
The companion therefore cannot read its own graph thinking well enough to notice that it dwells on one concern.
That is milestone section 10 and scenario SC-07, and it is the main gap between the current system and realistic introspection.

| # | Finding | Evidence |
| --- | --- | --- |
| F1 | A graph step enters `recent_thoughts` with `operation=""` and no cited sources. `PrivateThoughtLoop._recent_private_thoughts` reads it through `LegacyThoughtMetadata`, whose defaults hide the mismatch. `triggers._repeated_source` counts `source_ids`, so the `repeated_concern` trigger cannot see graph thinking. | Probe: `legacy view of graph op -> operation='' evidence_ids=()`. |
| F2 | `PrivateThoughtLoop._private_thought_ids` holds only `loop == "inner"` records. A main frame whose only stimuli were graph findings is not marked `self_feedback`, so `fresh_main_frame` can fire on the companion's own thought. | By reading the code. Not reproduced. |
| F3 | `workflows/introspection/snapshot.py` marks the `private_thought` capability as observed only from `loop == "inner"` records. | By reading the code. Not reproduced. |
| F4 | One graph step is two episode rows: the inner frame turn and the operation. `reflection.experiments.context` takes `recent(limit=6)` and `private_material` renders 3. The frame row has the later tick, so it sorts first. In the real replayed prompt the three rows were one action, the frame of step 15, and step 15. The same 130-word thought appeared three times. | `.runtime/replay/deliberate-reflection-20261114T110440Z/sample-01/divergence-01-006-replayed.txt`; probe: `thought text occurrences in private_material: 3`. |
| F5 | `_build_inspection` answers the introspection questions for `recent[0]`. For a graph step that is the frame row, which has no evidence links. The companion asks what information it had for its newest thought and is told none was retained. | By reading `workflows/inner_thought/host.py::_build_inspection` with the probe's `recent(limit=6)` order. |
| F6 | The episode metadata `inquiry_outcome`, `reason`, `information_gap`, `progress`, and `relations` never reaches a prompt. `workflows.reflection.schemas.EpisodeArchitecture` has `extra="ignore"` and no such fields. Only `reality` and `alternatives` survive. | Probe: all five `rendered contains ...: False`. |
| F7 | `EpisodeIndex._from_inquiry_operation` reads required writer fields with `total=False` TypedDicts and `.get(..., "")` defaults. Only `outcome` fails loud. | `pm/introspection/episodes.py`. |
| F8 | The self-parent filter in `_from_inquiry_operation` is dead code with a false comment. `stage_inquiry_step` sets `parents = sorted(targets) if targets else [prior.origin_record_id]`. The origin is the nomination or the action, never the operation. | Probe: `lineage.parent_record_ids: [4] operation id: 6`. |
| F9 | `docs/subsystems/introspection.md` says "the alternatives it settled". A generative step opens branches; it settles nothing. | That page. |
| F10 | A diverging replay overwrote `replay-01.json` in `deliberate-entry-20261114T110610Z` and `deliberate-reflection-20261114T110440Z`. Their `calls.jsonl` is intact. Pass 3 records them again. | File times in the two sample folders. |
| F11 | `pm/introspection/episodes.py` is not in `pyrightconfig.json`. The configured strict run does not cover it. | `pyrightconfig.json` include list. |
| F12 | No existing live gate drives a private finding into a main cycle, a Critical displacement of private work, or a main cycle that selects a mental action. `tests/live/test_private_inquiry.py` runs no main cycle. `tests/live/test_critical_preemption.py` covers main calculation and a public draft. The replay harness cannot record a main cycle. | Module docstrings of both tests; `docs/workflows/scenario-replay.md`. |

## Owner decisions (answered 2026-11-14)

No pass waits for the owner.

- **D1: what counts as a repeated concern in the graph.**
  A graph step counts toward repetition only when its accepted `progress.kind` is `"none"`. Linear thoughts keep the existing rule.
  Reason: every step of one inquiry cites the same origin sources by design, so "same source in 3 recent thoughts" would fire inside every normal 3-step inquiry. The rule reuses the stall signal that the graph boundary already validates.
- **D2: the double row.**
  No record-schema change. `EpisodeIndex` keeps both rows for owner tools. The two companion-facing projections drop an inner frame row when the list holds the private episode it outputs. The link is explicit (`turn.knoxel_ids` holds the operation id), so no causal fact is invented.
- **D3: recording budget.**
  No cap for this close-out. The owner said: "no im going all out". Record what the passes need. Do not stop to ask about recording cost. Report the provider cost of every sample from its manifest.
- **D4: B05 closes after these passes.**
  These are not B05 any more. They are gamma G01 in `dev/work_items/gamma.md`. Do not start them:
  a graph step that commits a `ReflectionDecision`; `capability_doubt` in the graph; decision 3B (a step proposes the next recall query); a sample that needs the extended reflection citation grant.

## Pass 1 — graph thoughts reach the private loop's own readers

Fixes F1, F2, F3. Uses D1. No model.

1. `workflows/inner_thought/host.py::_recent_private_thoughts`: branch on the record's `loop`.
   - `"inner"`: keep the current `LegacyThoughtMetadata` read.
   - `"inquiry"`: validate `workflows.inner_thought.inquiry.InquiryNodeMetadata`. Take the operation from `record.lineage.operation`, the sources from `step.evidence_ids`, and the step's `outcome`, `reason`, `information_gap`, and `progress.kind`.
   - any other value: raise.
2. `workflows/inner_thought/request_views.py::RecentThoughtView`: add typed fields with descriptions for what step 1 reads, plus the inquiry episode id. A linear thought has no inquiry. Represent that absence honestly, not with an invented value.
3. `workflows/inner_thought/triggers.py::_repeated_source`: implement D1. Use a plain loop with named intermediates.
4. `host.py::_private_thought_ids`: include `loop == "inquiry"` operations.
5. `workflows/introspection/snapshot.py`: count a graph operation as an observed private thought.

Checks, with a real store and a real `PrivateInquiryOwner`. Copy the setup of `tests/unit/pm/introspection/test_episodes.py::InquiryGraphEpisodeTests`:

- a graph step appears in `recent_thoughts` with its real operation and sources;
- three accepted steps with `progress.kind == "none"` on one source make `evaluate_triggers` return `repeated_concern`;
- three steps that claim accepted progress do not;
- a main frame that claims only a graph finding is `self_feedback`.

Run the inner-thought suite, the focus-management suite, `tests.unit.test_architecture`, and strict Pyright (see Commands).
The request JSON changes, so all three recordings will diverge on replay. That is expected. Do not record yet.

## Pass 2 — the companion reads each step once, with its reasons

Fixes F4 to F9. Uses D2. No model.

1. `pm/introspection/episodes.py::_from_inquiry_operation`: replace the four `total=False` TypedDicts with one small read-side Pydantic model in the same module, with required fields and `extra="ignore"`. `pm` must not import `workflows`. A missing required field must raise.
2. Same function: delete the self-parent filter and its comment. Add to `test_an_executed_graph_operation_is_an_episode`: the first step's `links["cause"]` equals the nomination record id.
3. D2: add one function in `pm/introspection/episodes.py` that returns recent episodes without the inner frame rows whose outputs hold a listed private episode. Use it in `workflows/reflection/experiments.py::context` and in `host.py::_build_inspection`. Leave `EpisodeIndex.recent` and `EpisodeIndex.all` unchanged.
4. `workflows/reflection/schemas.py::EpisodeArchitecture`: add typed fields for `inquiry_outcome`, `reason`, `information_gap`, `progress`, and `relations`. Read `pm/context_render/architectural.json` and make sure empty values on other episode kinds add no noise to a prompt.
5. Add `pm/introspection/episodes.py` to `pyrightconfig.json` only if you can bring it to zero strict errors without `Any`, `cast(Any, ...)`, or a bare `# type: ignore`. Otherwise report the remaining count and leave the config unchanged.
6. Docs: correct F9 in `docs/subsystems/introspection.md`; update the limits sentence in `docs/subsystems/private-loop.md`. Use the `project-documentation` skill.

Checks. Move the expectations of `.runtime/beta-b05/audit-20261114/probe.py` into `test_episodes.py`:

- the step's `reason` and `information_gap` appear in `private_material`;
- the frame row of that step does not appear;
- `newest_episode_answers` names the step's evidence.

Run the Pass 1 suites, `tests/unit/workflows/reflection`, `tests.unit.pm.introspection.test_episodes`, and the strict MkDocs build. Inspect the two changed pages in `site/`.

## Pass 3 — recorded evidence and owner review

Uses D3. OpenRouter only. Never start a local model.

1. Record `private-continuity`, `deliberate-entry`, and `deliberate-reflection` again. Copy each new snapshot folder and confirm the copy replays exactly.
2. Open one calls file of `deliberate-reflection` and read the reflection material the model received. Confirm with your own eyes that the thought appears once as an episode and that its reason is present. Quote the lines in the work-item record.
3. Add the scenario `private-dwelling` in `tests/replay/scenarios.py`, with its situation in `tests/replay/situations.py`. It has the shape of milestone 18.2, SC-07, and SC-11.
   - Synthetic input: Jesse answers a long Spamuel message with one short ambiguous line. Nothing else arrives.
   - The starting state has a committed cycle with a non-neutral mental state. `InnerRequest.mental_state` comes from the previous cycle, and `private_concern` currently builds a store with no cycle. Every recording so far used the neutral default, so no real run has shown felt state moving or a graph `QualeRecord`.
   - Run private cycles until the episode stops or the configured limits stop it, then two more opportunities.
   - The typed outcome records what happened: the nominations, the selection, each step's operation, outcome, and progress kind, the stop reason, whether `repeated_concern` fired, whether a later nomination concerns the companion's own thinking, and any `QualeRecord`.
   - Record at least three samples: `python -m tests.replay record --scenario private-dwelling --samples 3`.
   - Do not assert a model answer. An engineering failure is, for example, three stalled steps on one source with no `repeated_concern`. A model that declines to nominate a self-observation is a recorded quality result. Keep it and report it.
4. Run `python -m unittest tests.live.test_private_inquiry`. Keep its evidence folder.
5. Build the owner review. The owner has not yet seen B05 behaviour in the simulation browser.
   - Read `docs/subsystems/record-packages.md` and `dev/work_items/alpha_a19f_review.py` first.
   - From the `situation.pickle` of the most informative `private-dwelling` sample, capture with `pm.introspection.record_packages`: Jesse's message, the nominations, the selected one, every executed step and its branches, the inner turns, and any `QualeRecord`.
   - Register it: `uv run --locked python -m simulation.review register-package <file> --id b05-dwelling-<date> --title "B05: dwelling on a short reply"`.
   - Give the owner the direct review URL. Never invent a rating.
6. Add `private-dwelling` to the scenario table in `docs/workflows/scenario-replay.md`. Rebuild the strict MkDocs site.

## Pass 4 — one integrated live gate

Fixes F12. This is the hardest pass. Do the three parts in order and commit after each part that works.

Add `tests/live/test_private_integration.py`. Assemble the real application the way `tests/live/test_real_app_path.py` and `tests/live/test_critical_preemption.py` do: real `build_app`, real bus, real store, OpenRouter chat and embeddings. Write evidence under a fresh folder in `.runtime/beta-b05/integration/`.

- **Part A — a private finding reaches main cognition.** Let a private inquiry reach a finding. Then observe a main turn that claims the finding record as a stimulus. Assert on records only: the main `GwtTurnRecord.stimulus_ids` holds the finding id, and `EpisodeIndex.cross_loop_links()` returns the pair. Record what main appraisal did with it.
- **Part B — Critical displacement of private work.** While a private inquiry is active, send a message that real early triage classifies as Critical. Observe the displaced receipt that names the Critical epoch. After main accepts the reconsideration, give the private loop another opportunity and record whether the displaced question is offered and what the model does with it. If triage does not classify the message as Critical, that is a recorded result. Do not force the category.
- **Part C — main selects deliberate mental work.** Give input that makes a mental action a reasonable choice. If main selects one, follow it: the deliberate episode, the delivered finding under the action's correlation, and the main turn that appraises it. If main does not select a mental action in three attempts, report `NOT RUN — main did not select a mental action` with the three decision records. Do not commit the action yourself and call it main selection.

Report each part as passed, failed with its cause, or `NOT RUN` with its reason.

## Closing B05

When the four passes are done:

- update the status line of `dev/work_items/beta-b05.md` and add the final dated section;
- add the final note to card 7. Do not move the card: beta continues with B06;
- report to the owner in the order `AGENTS.md` section 19 requires, with the review URL;
- stop. The owner closes B05 after reviewing the record package. Do not start B06 or gamma.

## Commits and records

End each pass with one local commit. Use a scoped Conventional Commit subject, for example
`fix(inner-thought): show graph steps to the private loop's own readers`.
After each pass, add a dated section to `dev/work_items/beta-b05.md` and one dated note to card 7 in `dev/kanban/board.json`.
State what ran, what was observed, and what is `NOT RUN`. Edit the board with `dev.kanban.server.read_board` and `write_board` so its format stays valid.

## Do not

- Do not add operations, needs, a ToM mode, or a new store. B06 owns ToM and main reconsideration of private thought.
- Do not change `GwtTurnRecord`, the recovery-root rule, or `stage_decision`.
- Do not fix the known red suites below in a B05 commit.
- Do not replay an original recording when the result can diverge. A replay overwrites the `replay-*.json` files of its snapshot. Copy the snapshot folder first.
- Do not relax a validator, add a default, or lower a threshold to make a model answer pass. Keep the failed sample and report it.
- Do not run the complete unit or contract suites.

## Stop and ask the owner when

- a pass needs a contract change this file does not name;
- the D1 rule fires inside an ordinary inquiry that makes progress, or never fires in any `private-dwelling` sample that stalls;
- a live gate fails for a reason that is not clearly one model-quality sample;
- the same symptom needs a third patch.

## Known defects outside B05

- `tests/unit/pm/introspection`: three failures that reproduce at `4018528`: `test_a_cycle_package_holds_the_state_the_turn_and_every_named_record`, `test_a_changed_finding_supersedes_and_keeps_history`, and `test_narrative_revision_keeps_the_original_episode_reference`.
- Every recorded sample shows one rejected early-triage answer: missing properties at `entity_extraction.claims[0]`. It costs one extra request per message.
- `tests/unit/pm/memory/reconsolidation`: one failure and one error.
- `test_focus` (3), main-cognition discovery fixture errors, `public_speech` discovery (1 error, 1 failure), and the main-cognition phase contracts are red, as the B05 work item records.
- `tests.unit.workflows.cognitive_control.test_wait` and `tests.unit.workflows.test_companion_wiring` retry remote embeddings until an endpoint answers.
- `tests/contract/inner_thought` has a stale inspection fixture (`test_inspection_evidence_reaches_the_prompt`).
- `test_cognition_wiring` has 11 errors at `ab9c7f3`.

## Standing rules for this milestone

- OpenRouter only. Never start a local LLM.
- Known provider-reported spend so far is about $0.44. Not included: earlier incomplete B03 usage, two interrupted B05 requests, and the six recordings of 2026-11-13, which have no cost receipt.
- Use short targeted real checks. Do not run a month simulation.
- The final Jesse–Spamuel, Jesse–Spamuel 1–Spamuel alpha, and Spamuel–Spamuel–Minecraft scenarios come only after integrated beta completion.

## Commands

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/inner_thought -t . -b
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/focus_management -t . -b
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/reflection -t . -b
.venv/Scripts/python.exe -m unittest tests.unit.pm.introspection.test_episodes tests.unit.test_architecture -b
npx --no-install pyright --project pyrightconfig.json
python -m tests.replay list
python -m tests.replay record --scenario <name> --samples <count>
python -m tests.replay replay --snapshot .runtime/replay/<copy of the snapshot folder>
python -m unittest tests.live.test_private_inquiry
python -m unittest tests.live.test_displaced_offer_salience
uv run --locked python -m simulation.review register-package <file> --id <id> --title "<title>"
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

A recording made before a prompt change stops at the first changed request. Record it again.
The recordings current before Pass 1 are `private-continuity-20261114T104837Z`, `deliberate-entry-20261114T110610Z`, and `deliberate-reflection-20261114T110440Z`.
Receipts of earlier passes are under `.runtime/beta-b05/`.

## After B05

B06 (deliberate theory of mind and metacognition) and B07 (interval appraisal and sustained acceptance) follow in `dev/work_items/beta.md`.
Gamma starts after beta: `dev/work_items/gamma.md`.
