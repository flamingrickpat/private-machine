# Beta handoff — B06: deliberate theory of mind and metacognition

Written 2026-11-14 after B05 close-out. **B06 has no accepted design.** This handoff starts with an
audit and an architecture proposal. Do not write product code before the owner accepts that proposal.

> **Update 2026-11-14:** Pass 0 (audit and architecture proposal) is written in
> `dev/work_items/beta-b06-plan.md`, together with the ordered implementation steps. That file replaces
> `beta-b06-proposal.md` and the pass sketch below. It is **proposed**: do not implement before the owner's
> answer is recorded in its section 2. This handoff stays the authority for guard rails, known red suites,
> and standing rules.

Your task: produce the B06 architecture proposal, stop for the owner, then implement the accepted
passes in order. Read `AGENTS.md` first. Its rules are mandatory. The ones that matter most here:
real stores and real endpoints only, no mocks, typed contracts, fail loud on a broken internal
contract, and no `.get(field, default)` on a required internal field.

## Where things are

- Branch `beta`. B05 close-out is committed through `574a923`. The tree is clean. Nothing is pushed.
  Do not push.
- B05 is complete pending the owner's rating of the review package. B06 has not started. B07 follows
  B06. Gamma starts after beta.
- The beta card (7) stays `doing`.
- Package version is `1.0.2`.

The records are the authority, not this file:

- `dev/work_items/beta.md`: the beta plan. B06 is the row "ToM and metacognition | B04, B05 |
  Action-selected perspective inquiry and reconsideration of private thought".
- `dev/work_items/beta-b05.md`: the accepted B05 design, the four close-out passes, and what B05
  deliberately left to B06 and gamma.
- `dev/work_items/gamma.md`: G01 is the B05 reflection work deferred to gamma. It is **not** B06.
- `docs/continuous-character-milestone.md`:
  - section 14, deliberate theory of mind;
  - section 15.3, self-confirming thought loops;
  - section 18.2, spontaneous suspicion and metacognition;
  - SC-07 (noticing dwelling) and SC-09 (a ToM inquiry estimates a morning routine);
  - sections 16 and 17 for the boundary with B07 and character fidelity.
- `docs/subsystems/entity-context.md`, `deliberate-mental-work.md`, `private-loop.md`,
  `reflection.md`, `introspection.md`.
- `dev/work_items/beta-handoff-20261112.md`: the B05 handoff. Its guard rails and known defects
  still apply.

## What B06 is

Two connected capabilities, not two systems:

1. **Deliberate theory of mind.** A first-class, action-selectable inquiry workflow that builds and
   revises a bounded model of another entity. It reuses the existing generative/discriminatory graph,
   evidence, hypotheses, lineage, and reflection machinery. It is not a mandatory stage of every
   cycle. It can return a provisional model revision, competing explanations, a conditional
   prediction, a better question, a proposed way to obtain evidence, or an explicit unresolved
   result. Model revisions pass through the entity-model owner. Consequential findings return to
   main cognition for appraisal. Public questions or contact remain selected actions.

2. **Metacognition.** The character's own thought activity becomes appraisable. It may pause,
   redirect, or investigate its preoccupation (SC-07). This is the B06 home for "reconsideration of
   private thought". It must not create infinite reflection recursion, and it must not treat
   rumination as corroboration.

B06 must generalize beyond humans. A person may be unavailable because of work; a Minecraft agent
may be unavailable because it is escaping danger. Do not force human motives onto every entity.

## What exists today (audit, by reading the code)

These are starting points, not proof that the target works. Verify each at the current revision.

| # | Finding | Evidence |
| --- | --- | --- |
| F1 | There is no ToM workflow, action type, or inquiry context. `ActionType` has Reply, Ignore, ToolCall, ToolCallAndReply, InitiateUserConversation, Loopback, Sleep, and Idle. The only ToM token is the unused producer label `"tom"` in `EntityProducer`. | `pm/records/enums.py::ActionType`; `workflows/entity_context/schemas.py::EntityProducer`. |
| F2 | The entity-model owner already has the right storage boundary. `EntityContextOwner.stage_update` accepts typed proposals with `producer`, `parent_receipt_id`, claims, retractions, challenges, and correction evidence. Claims carry `layer` including `"perspective"` and `"open_question"`, `epistemic_kind` including `"hypothetical"`, `perspective_entity_id`, confidence, and interaction ability estimates. | `workflows/entity_context/owner.py::stage_update`; `entity_context/schemas.py::EntityContextProposal`. |
| F3 | Mental action selection exists but is generic. Main cognition dispatches a `focus_management` action with `ActionArguments(instruction, operation)`. `MentalPlan` carries `focus`, `goal_changes`, `evidence_ids`, `summary`, and `disposition`. There is no target entity, perspective, or ToM question in the action contract. | `workflows/cognitive_control/schemas.py::MentalPlan`; `docs/subsystems/deliberate-mental-work.md`. |
| F4 | The inquiry graph is generic and can host ToM. `InquiryStep` has `operation`, `evidence_ids`, `assumptions`, `reality`, `judgments`, `relations`, `alternatives`, `progress`, `outcome`, `reason`, `information_gap`, and `next_operation`. It has no target entity or attributed-perspective field. | `workflows/inner_thought/schemas.py::InquiryStep`. |
| F5 | Main cognition already appraises private findings. `thought.created` is an accepted stimulus; `MainCogLoop._private_only` marks a batch of only private results; B05's integrated gate Part A observed a private finding claimed by a main turn and an explicit cross-loop pair. | `workflows/main_cognition/host.py`; `tests/live/test_private_integration.py`; `.runtime/beta-b05/integration/20261114T192729Z-99d18e27-part-a/part-a.json`. |
| F6 | There is no "reconsideration of private thought" path beyond that appraisal. Main can accept a private finding, but nothing revises a prior appraisal of a private thought, and the private loop cannot ask main to reconsider one. | By reading `main_cognition/host.py` and `inner_thought/host.py`. Not reproduced. |
| F7 | The metacognition foundation is the B05 repeated-concern rule. `_repeated_source` counts a graph step only when its accepted `progress.kind == "none"` and the same source recurs three times. `rumination_notice` is a delivered operation. The `private-dwelling` scenario recorded three samples; two stalled and fired `repeated_concern`, one made progress and did not. | `workflows/inner_thought/triggers.py::_repeated_source`; `tests/replay/scenarios.py::run_private_dwelling`; `.runtime/replay/private-dwelling-20261114T181741Z/`. |
| F8 | The stop is a hard switch, not a bias. The model proposes `outcome = "pause"`; the graph accepts it. The deterministic bound (`stalled_steps`, `episode_steps`, `episode_seconds`) can also pause the episode, and the model cannot continue past it. In `private-dwelling` sample 3 the model's `pause` and the stall bound coincided, so the model's wish changed the reason text but not whether the episode stopped. | `workflows/inner_thought/inquiry.py::stage_inquiry_step`; sample 3 receipt history. |
| F9 | The owner said on 2026-11-14 that the current hard-stop behaviour "is just fine". Do not turn the stop into a graded bias in B06 unless the owner explicitly changes that preference. | This conversation, 2026-11-14. |
| F10 | No live gate or recorded scenario covers SC-09 (a ToM inquiry estimates a morning routine) or a metacognitive appraisal of the character's own thought activity beyond noticing dwelling. | By reading `tests/live/` and `tests/replay/scenarios.py`. |
| F11 | B05 deferred four reflective-inquiry items to gamma G01: a graph step that commits a `ReflectionDecision`, `capability_doubt` in the graph, decision 3B (a step proposes the next recall query), and a sample for the extended reflection citation grant. Do not implement them in B06. | `dev/work_items/gamma.md`. |

## Open owner decisions

These must be settled in the proposal. Do not choose silently.

- **D1: the ToM action shape.** A new `ActionType`/workflow name, or a `focus_management` action with a
  ToM operation and a target entity? The milestone says `ActionType.ToM` is a conceptual routing
  label, not a required enum spelling.
- **D2: where a ToM model revision is accepted.** The existing `EntityContextOwner.stage_update` with
  `producer="tom"` is the obvious boundary. Confirm it, or name the mismatch and the migration.
- **D3: what "reconsideration of private thought" means.** Main re-appraises a private finding, the
  private loop reconsiders its own prior step, or both? Name the owner of the revision and the
  record that carries it.
- **D4: the metacognition control policy.** Keep the hard `pause` switch (owner's current
  preference), or make the stop wish a graded bias. If the latter, it is a contract change and needs
  its own design.
- **D5: the ToM evidence grant.** Which records may a ToM step cite: the target entity's claims, the
  observer's own history, both, or a scoped subset? The entity context already separates subject,
  observer, perspective, and scope.
- **D6: the first recorded ToM scenario.** SC-09's morning routine is the milestone example. Confirm
  it, or name a different bounded question.

## Pass 0 — audit and architecture proposal

No product code. Produce a proposal in `dev/work_items/beta-b06-proposal.md` with the nine items
`AGENTS.md` section 5 requires:

1. The current module graph and observed execution path for action selection, the private slot, the
   inquiry graph, and entity-context acceptance.
2. Concrete current problems with filenames and symbols.
3. Target responsibilities and folder structure.
4. Exact existing and proposed public types or signatures.
5. Data ownership, persistence, side effects, and error boundaries.
6. How existing behaviour is preserved or intentionally changed.
7. Live prerequisites and a realistic verification plan.
8. An ordered list of small implementation steps.
9. Risks, open decisions, and required owner choices.

Run a read-only audit against the current revision. Do not trust this file's findings without
checking them. Include the owner decisions above with a recommended answer and its consequences.

**Stop after the proposal.** Do not implement until the owner accepts it. The implementation passes
below are a sketch, not accepted scope.

## Implementation passes (sketch, to be replaced by the accepted proposal)

- **Pass 1 — ToM action and inquiry context.** An action-selected perspective inquiry with a target
  entity, a bounded question, and a scoped evidence grant. It reuses the existing graph and the
  existing private slot. It sends no message.
- **Pass 2 — entity-model revision.** A ToM finding becomes a typed `EntityContextProposal` through
  `EntityContextOwner`, with `producer="tom"`, explicit perspective, confidence, and correction or
  challenge targets. Consequential findings still return to main for appraisal.
- **Pass 3 — metacognition.** The character's own thought activity becomes appraisable. Keep the
  hard `pause` switch unless D4 changes it. Cover SC-07 beyond noticing dwelling.
- **Pass 4 — recorded evidence and integrated gate.** A recorded ToM scenario (SC-09 shape) and a
  metacognition scenario, plus one integrated live gate that shows a ToM finding reaching the entity
  model and main appraisal, and a private thought being reconsidered.

## Guard rails

- Do not add a second inquiry system, a second entity store, or a new memory store. Reuse
  `workflows.inner_thought` and `workflows.entity_context`.
- Do not let ToM send a message, contact a person, or change an entity model without the owner's
  acceptance boundary.
- Do not force human motives onto a Minecraft agent or any non-human entity.
- Do not change the B05 hard-stop policy without D4.
- Do not implement gamma G01 items.
- Do not change `GwtTurnRecord`, the recovery-root rule, or `stage_decision`.
- Do not fix the known red suites in a B06 commit.
- Do not run the complete unit or contract suites.
- Do not replay an original recording in place; copy the snapshot folder first.
- Do not relax a validator, add a default, or lower a threshold to make a model answer pass. Keep the
  failed sample and report it.

## Stop and ask the owner when

- the proposal needs a contract change this file does not name;
- a ToM action would contact the person or mutate an entity model without the owner's boundary;
- a metacognition policy change would alter the B05 hard-stop behaviour;
- a live gate fails for a reason that is not clearly one model-quality sample;
- the same symptom needs a third patch.

## Known defects outside B06

Carry these forward; do not fix them in a B06 commit.

- `tests/unit/pm/introspection`: three failures that reproduce at `4018528`:
  `test_a_cycle_package_holds_the_state_the_turn_and_every_named_record`,
  `test_a_changed_finding_supersedes_and_keeps_history`, and
  `test_narrative_revision_keeps_the_original_episode_reference`.
- Every recorded sample shows one rejected early-triage answer: missing properties at
  `entity_extraction.claims[0]`. It costs one extra request per message.
- `tests/unit/pm/memory/reconsolidation`: one failure and one error.
- `test_focus` (3), main-cognition discovery fixture errors, `public_speech` discovery (1 error,
  1 failure), and the main-cognition phase contracts are red, as the B05 work item records.
- `tests.unit.workflows.cognitive_control.test_wait` and
  `tests.unit.workflows.test_companion_wiring` retry remote embeddings until an endpoint answers.
- `tests/contract/inner_thought` has a stale inspection fixture
  (`test_inspection_evidence_reaches_the_prompt`).
- `test_cognition_wiring` has 11 errors at `ab9c7f3`.
- `tests.live.test_private_inquiry` failed 3/3 on 2026-11-14 with
  `deepseek/deepseek-v4.1-flash-1017`: two "did not exercise generation", one "no accepted outcome
  finding". This is a model-quality result, not a licence to change the gate.
- `tests.live.test_private_integration` Part B observed a Critical displacement once, then reported
  `NOT RUN` twice because main committed an inference wait. Part C reported `NOT RUN` because main
  selected no mental action in three attempts.

## Standing rules for this milestone

- OpenRouter only. Never start a local LLM.
- Use short targeted real checks. Do not run a month simulation.
- The final Jesse–Spamuel, Jesse–Spamuel 1–Spamuel alpha, and Spamuel–Spamuel–Minecraft scenarios
  come only after integrated beta completion (B07). Do not start them in B06.

## Commands

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/inner_thought -t . -b
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/focus_management -t . -b
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/entity_context -t . -b
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/reflection -t . -b
.venv/Scripts/python.exe -m unittest tests.unit.pm.introspection.test_episodes tests.unit.test_architecture -b
npx --no-install pyright --project pyrightconfig.json
python -m tests.replay list
python -m tests.replay record --scenario <name> --samples <count>
python -m tests.replay replay --snapshot .runtime/replay/<copy of the snapshot folder>
python -m unittest tests.live.test_private_inquiry
python -m unittest tests.live.test_private_integration
uv run --locked python -m simulation.review register-package <file> --id <id> --title "<title>"
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

## After B06

B07 (interval appraisal and sustained acceptance) follows in `dev/work_items/beta.md`. Gamma starts
after beta: `dev/work_items/gamma.md`.
