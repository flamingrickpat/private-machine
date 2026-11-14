# B05 close-out handoff (2026-11-14 audit)

The owner accepted this plan on 2026-11-14 and asked to commit it.
It supersedes the "Remaining B05 work" section of `beta-handoff-20261112.md`. That section still lists the episode-index change as open.
The authority for the decisions is `dev/work_items/beta-b05.md`, section "Audit and close-out decisions (2026-11-14)".

Read `AGENTS.md` first. Its rules are mandatory. The ones that matter most here:
real stores and real endpoints only, no mocks, typed contracts, fail loud on internal contract breaks, no `.get(field, default)` on a required internal field.

## State of the tree

- Branch `beta`. The episode-index half is committed as `3c2640b`. Nothing is pushed. Start from a clean tree.
- That half is sound and additive. Audit reran `tests.unit.pm.introspection.test_episodes`: 13 passed.
  `tests/unit/pm/introspection`: 42 ran, 3 failed. The same 3 fail at `4018528` in a clean worktree (36 ran, 3 failed), so they predate the change.
- Known provider spend is about $0.44. One recorded sample costs $0.01 to $0.02.

## What the audit found

Probe and output: `.runtime/beta-b05/audit-20261114/probe.py`, `probe.txt`. Real store, real `PrivateInquiryOwner`, no model.

The private mind now thinks in the inquiry graph (`loop: "inquiry"`). Several readers still understand only the old linear thought (`loop: "inner"`).
The result: the companion cannot see its own graph thinking properly, so it cannot notice that it is dwelling on something. That is milestone section 10 and SC-07.

| # | Finding | Evidence |
| --- | --- | --- |
| F1 | A graph step enters `recent_thoughts` with `operation=""` and no cited sources. `PrivateThoughtLoop._recent_private_thoughts` reads it through `LegacyThoughtMetadata`, whose defaults hide the mismatch. `triggers._repeated_source` counts `source_ids`, so the `repeated_concern` trigger cannot see graph thinking. | Probe: `legacy view of graph op -> operation='' evidence_ids=()`. |
| F2 | `PrivateThoughtLoop._private_thought_ids` holds only `loop == "inner"` records. A main frame whose only stimuli were graph findings is not marked `self_feedback`, so `fresh_main_frame` can fire on the companion's own thought. | By reading the code. Not reproduced. |
| F3 | `workflows/introspection/snapshot.py` marks the `private_thought` capability as observed only from `loop == "inner"` records. | By reading the code. Not reproduced. |
| F4 | One graph step is two episode rows (inner frame turn + operation). `reflection.experiments.context` takes `recent(limit=6)` and `private_material` renders 3. The frame row has the later tick, so it wins. In the real replayed prompt the three rendered rows were: one action, the frame of step 15, and step 15. The same 130-word thought appeared three times. | `.runtime/replay/deliberate-reflection-20261114T110440Z/sample-01/divergence-01-006-replayed.txt`; probe: `thought text occurrences in private_material: 3`. |
| F5 | `_build_inspection` answers the introspection questions for `recent[0]`. For a graph step that is the frame row, which has no evidence links. The companion asks "what information did I have?" about its newest thought and is told none was retained. | By reading `workflows/inner_thought/host.py::_build_inspection` with the probe's `recent(limit=6)` order. |
| F6 | The new episode metadata (`inquiry_outcome`, `reason`, `information_gap`, `progress`, `relations`) never reaches a prompt. `workflows.reflection.schemas.EpisodeArchitecture` has `extra="ignore"` and no such fields. Only `reality` and `alternatives` survive. | Probe: all five `rendered contains ...: False`. |
| F7 | `_from_inquiry_operation` reads required writer fields with `total=False` TypedDicts and `.get(..., "")` defaults. Only `outcome` fails loud. | `pm/introspection/episodes.py`. |
| F8 | The self-parent filter in `_from_inquiry_operation` is dead code with a false comment. `stage_inquiry_step` sets `parents = sorted(targets) if targets else [prior.origin_record_id]`; the origin is the nomination or action, never the operation. | Probe: `lineage.parent_record_ids: [4] operation id: 6`. |
| F9 | `docs/subsystems/introspection.md` says "the alternatives it settled". A generative step creates open branches; it settles nothing. | Diff of that page. |
| F10 | The current recordings' `replay-01.json` files were overwritten by a diverged replay. `calls.jsonl` is intact. Pass 3 re-records them anyway. | File times in the two sample folders. |
| F11 | `pm/introspection/episodes.py` is not in `pyrightconfig.json`. "Configured strict Pyright: 0 errors" does not cover it. The 92-error figure was not rerun by the audit. | `pyrightconfig.json` include list. |

## Owner decisions (answered 2026-11-14)

All four are answered. No pass waits for the owner.

- **D1 (Pass 1): what counts as a repeated concern in the graph. Accepted.**
  A graph step counts toward repetition only when its accepted `progress.kind` is `"none"`. Linear thoughts keep the old rule.
  Reason: every step of one inquiry cites the same origin sources by design, so "same source in 3 recent thoughts" would fire inside every normal 3-step inquiry. The rule reuses the stall signal the graph boundary already validates.
- **D2 (Pass 2): the double row. Accepted.**
  No record-schema change. `EpisodeIndex` keeps both rows for owner tools. The two companion-facing projections drop an inner frame row when the list holds the private episode it outputs. The link is explicit (`turn.knoxel_ids` holds the operation id), so no causal fact is invented.
- **D3 (Pass 3): recording budget. No close-out cap.**
  The owner declined a $2 cap: "no im going all out". Record what the passes need. Do not stop to ask about recording cost. Report the provider cost of every sample from its manifest.
  `dev/work_items/beta.md` still names a $25 interim limit; expected close-out spend is far below it.
- **D4: close B05 after Pass 3. Accepted.**
  These are no longer B05. They are gamma G01, in `dev/work_items/gamma.md`:
  a graph step that commits a `ReflectionDecision`; `capability_doubt` in the graph; decision 3B (a step proposes the next recall query); a sample that needs the extended reflection citation grant.

## Pass 0 — done

The episode-index half is committed as `3c2640b`. Card 7 has its note.

## Pass 1 — graph thoughts reach the private loop's own readers

Uses D1.

1. `workflows/inner_thought/host.py::_recent_private_thoughts`: branch on the record's `loop`.
   - `"inner"`: keep the current `LegacyThoughtMetadata` read.
   - `"inquiry"`: validate `workflows.inner_thought.inquiry.InquiryNodeMetadata`. Take the operation from `record.lineage.operation`, the sources from `step.evidence_ids`, and the step's `outcome`, `reason`, `information_gap`, and `progress.kind`.
   - any other value: raise.
2. `workflows/inner_thought/request_views.py::RecentThoughtView`: add typed, described fields for what step 1 reads, plus the inquiry episode id. A linear thought has no inquiry; represent that absence honestly, not with an invented value.
3. `workflows/inner_thought/triggers.py::_repeated_source`: implement the D1 rule. Keep it a plain loop with named intermediates.
4. `host.py::_private_thought_ids`: include `loop == "inquiry"` operations (F2).
5. `workflows/introspection/snapshot.py`: count a graph operation as an observed private thought (F3).

Checks, no model, real store and real `PrivateInquiryOwner` (copy the setup in `tests/unit/pm/introspection/test_episodes.py::InquiryGraphEpisodeTests`):
- a graph step appears in `recent_thoughts` with its real operation and sources;
- three accepted steps with `progress.kind == "none"` on one source make `evaluate_triggers` return `repeated_concern`; three steps that claim progress do not;
- a main frame that claims only a graph finding is `self_feedback`.

Run: `tests/unit/workflows/inner_thought`, `tests/unit/workflows/focus_management`, `tests.unit.test_architecture`, and `npx --no-install pyright --project pyrightconfig.json`.
Expect all three recordings to diverge on replay, because the request JSON changed. Replay copies, not the originals. Do not re-record yet.

## Pass 2 — the companion reads each step once, with its reasons

Uses D2.

1. `pm/introspection/episodes.py::_from_inquiry_operation` (F7): replace the four `total=False` TypedDicts with one small read-side Pydantic model in the same module (required fields, `extra="ignore"`). `pm` still must not import `workflows`. A missing required field must raise.
2. Same function (F8): delete the self-parent filter and its comment. Add to `test_an_executed_graph_operation_is_an_episode`: the first step's `links["cause"]` equals the nomination record id.
3. D2 fold (F4, F5): one function in `pm/introspection/episodes.py` that returns recent episodes without the inner frame rows whose outputs hold a listed private episode. Use it in `workflows/reflection/experiments.py::context` and in `host.py::_build_inspection`. Leave `EpisodeIndex.recent` and `all` unchanged.
4. `workflows/reflection/schemas.py::EpisodeArchitecture` (F6): add typed fields for `inquiry_outcome`, `reason`, `information_gap`, `progress`, and `relations`. Check `pm/context_render/architectural.json` so empty values on other episode kinds do not add noise.
5. Docs: fix F9 in `docs/subsystems/introspection.md`; update the limits sentence in `docs/subsystems/private-loop.md`; add a dated section to `dev/work_items/beta-b05.md`. Use the `project-documentation` skill.

Checks: extend `.runtime/beta-b05/audit-20261114/probe.py` expectations into `test_episodes.py`:
the step's `reason` and `information_gap` appear in `private_material`; the frame row does not; `newest_episode_answers` names the step's evidence.
Run the Pass 1 suites, `tests/unit/workflows/reflection`, and the strict MkDocs build.

## Pass 3 — real evidence and owner review, then close B05

Uses D3. OpenRouter only. Never start a local model.

1. Re-record `private-continuity`, `deliberate-entry`, `deliberate-reflection` (`python -m tests.replay record --scenario <name>`). Confirm each replays exactly from a copy.
2. Add one scenario, `private-dwelling`, in `tests/replay/scenarios.py` with its situation in `tests/replay/situations.py`. It is the shape of milestone 18.2, SC-07, and SC-11:
   - synthetic input: Jesse answers a long Spamuel message with one short ambiguous line, and nothing else arrives;
   - the starting `CycleState` has a non-neutral mental state. Every recording so far used the neutral default, so no real run has shown felt state moving or a graph `QualeRecord`;
   - run private cycles until the episode stops or the configured limits stop it.
   Record at least three samples. The model is stochastic, and one lucky sample proves little.
   The typed outcome records what happened: nominations, the selection, each step's operation, outcome and progress kind, the stop reason, whether `repeated_concern` fired, whether a later nomination concerns the companion's own thinking, and any `QualeRecord`.
   Do not assert a model answer. An engineering failure is, for example, three stalled steps on one source with no `repeated_concern`. A model that declines to nominate a self-observation is a recorded quality result, not a test to weaken.
3. Run the existing integrated live gate `tests/live/test_private_inquiry.py` and, for Critical displacement of private work, the relevant case in `tests/live/test_critical_preemption.py`. Read both files first and report what each actually exercises. The replay harness cannot record a main cycle, so main appraisal of a finding can only be shown here.
4. Capture a record package from the `private-dwelling` result with `pm.introspection.record_packages`, register it with `python -m simulation.review register-package`, and give the owner the direct review URL. The owner has not yet seen B05 behaviour in the simulation browser.
5. Report provider cost per sample from the manifests. Update `dev/work_items/beta-b05.md`, card 7, and `docs/workflows/scenario-replay.md` (scenario table).

Anything not run is `NOT RUN` with the reason. Do not turn it into a pass.

## Commits and records

End each pass with one local commit. Use a scoped Conventional Commit subject, for example
`fix(inner-thought): show graph steps to the private loop's own readers`. Do not push.
After each pass, add a dated section to `dev/work_items/beta-b05.md` and one dated note to card 7 in `dev/kanban/board.json`.
State what ran, what was observed, and what is `NOT RUN`.

## Do not

- Do not add operations, needs, a ToM mode, or a new store. B06 owns ToM and main reconsideration of private thought.
- Do not start any gamma G01 part.
- Do not change `GwtTurnRecord`, the recovery-root rule, or `stage_decision`.
- Do not fix the three pre-existing introspection failures or the other known red suites in a B05 commit.
- Do not replay an original recording when the result can diverge. Copy the sample folder first.
- Do not relax a validator or add a default to make a model answer pass. Keep the failed sample and report it.

## Stop and ask the owner when

- a pass needs a contract change this file does not name;
- the D1 rule fires on an ordinary inquiry or never fires in `private-dwelling`;
- a live gate fails for a reason that is not clearly a model-quality sample.
