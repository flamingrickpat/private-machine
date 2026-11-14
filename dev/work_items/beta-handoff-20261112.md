# Beta handoff — current state

The owner asked to commit this file on 2026-11-14. It describes the state at `4018528`.
Its "Remaining B05 work" section is superseded by [the B05 close-out handoff](beta-b05-closeout-handoff.md).
It was rewritten on 2026-11-14. It holds the current state only.
The earlier text is kept at `.runtime/beta-b05/reflection-evidence/beta-handoff.before-rewrite.md`
and `.runtime/beta-b05/displacement-20261114/beta-handoff-20261112.before-rewrite.md`.

## Where things are

Branch `beta`, head `4018528`, nothing uncommitted except this file. Nothing is pushed. Package version is `1.0.2`.
Three passes landed on 2026-11-14: graph felt-state emission (`df69d59`), deliberate mental work in the private slot
(`eaef6a9`), and scoped reflection and inspection evidence (`4018528`).
B01 to B04 are complete within their targeted scope. B05 is partial. The beta card (7) stays `doing`.

The records are the authority, not this file:

- `dev/work_items/beta-b05.md`: accepted design, every pass, deviations, limits, and verification.
- `dev/kanban/board.json`, card 7: one dated note per pass.
- `docs/subsystems/private-loop.md`: how the private loop works now.
- `docs/subsystems/deliberate-mental-work.md`: how a selected mental action runs now.
- `docs/subsystems/reflection.md`: what a reflection step reads now, and what it cannot change.
- `docs/workflows/scenario-replay.md`: how to record and replay a scenario.

## Done in B05

- Typed nominations, separate attention, the guarded inquiry graph, parked re-admission, and deliberate entry at the owner boundary.
- Ordinary input no longer cancels private work. Critical acceptance owns cancellation.
- Critical displacement, and salience-based selection among several displaced questions.
- Graph felt-state emission: each accepted operation retains one sample of the five channels, and the derived inner turn emits a `QualeRecord` per notably moved channel.
- Deliberate mental work in the private slot. `FocusManagement` is a plain bus module: it admits the selected action, parks an active spontaneous inquiry with a receipt naming the action, commits the action lifecycle, and delivers the finding under the action's correlation. The pm-coder agent session is removed. The integration that ends deliberate work is asked for focus advice.
- Two defects fixed there: a private commit held the store lock while input awaited classification (deadlock), and a private opportunity cited an expectation from an unaccepted derived commit.
- Scoped reflection and inspection evidence. A step whose operation is `reflection_review` reads the recorded experiments, retained episodes, and their sources; `capability_inspection` and `episode_inspection` read the inspector's projection. Both are built under the observer's accepted reality and frame, an out-of-scope experiment or episode is omitted with a stated limit, the evidence travels beside the request so other steps' prompts are unchanged, and a reflection step extends its citation grant with those records.
- Defect fixed there: a retrieved inquiry receipt carried its accumulated computation time into `source_fingerprint`, so a recording that cited one could never replay. The elapsed duration is now treated as process bookkeeping.

## Remaining B05 work, in order

1. Finish remaining item 1. Two halves remain, and the first needs an owner decision:
   - Run the reflection child for a graph `reflection_review` step and commit its `ReflectionDecision`. Every graph commit is an append-only recovery root, and a decision replaces an existing experiment record, so it needs a derived, non-recovery-root commit plus a decision about what to retain on the operation for repair. The three extra model calls per reflection step are part of that choice.
   - Let the episode index read graph operations. `EpisodeIndex` still lists only `loop == "inner"` monologues, so a graph operation (`loop: "inquiry"`) is not an episode row, and the inspection evidence cannot show the graph's own trace.
2. Run the integrated live gates. They must include a real Critical displacement: no model has seen a displaced offer or `earlier_artifacts` yet. They should leave two displaced questions waiting, because the offer selection has run against the real embedding endpoint only outside the private loop. They should move the mental state between two private operations: no real run has emitted a graph `QualeRecord` yet. For deliberate work they need a main cycle that selects a mental action and appraises its finding and focus advice, the default first operation `memory_search`, an accepted wait, and Critical displacement of a deliberate episode.

Accepted separate pass, not part of the list above: let an inquiry step propose the query of the next `memory_search` step (decision 3B). It changes the step contract of every inquiry, so the recordings need re-recording.

## Known defects outside B05

- Every recorded sample shows one rejected early-triage answer: missing properties at `entity_extraction.claims[0]`. One extra request per message.
- `tests/unit/pm/memory/reconsolidation`: one failure and one error, reproduced with the `dependencies.py` of `eaef6a9`.
- `test_focus` (3), main-cognition discovery fixture errors, `public_speech` discovery (1 error, 1 failure), and the main-cognition phase contracts remain red as recorded in the B05 work item.
- `tests.unit.workflows.cognitive_control.test_wait` and `tests.unit.workflows.test_companion_wiring` retry remote embeddings until an endpoint answers.
- `tests/contract/inner_thought` still has the stale inspection fixture (`test_inspection_evidence_reaches_the_prompt`).
- `test_cognition_wiring` has 11 errors at `ab9c7f3`.

## Standing rules for this milestone

- OpenRouter only. Never start a local LLM. The shared interim limit is $25; the final scenarios have separate authorization.
- Known provider-reported spend: B03 + B04 + B05 gates $0.2629, plus $0.0173 for two recordings of 2026-11-14, plus $0.0094 for the felt-state re-recording, plus $0.0751 for six runs of `deliberate-entry`, plus $0.080125 for six samples of 2026-11-14 evening (three current recordings plus three superseded samples). Not included: earlier incomplete B03 usage, two interrupted B05 requests, and the six recordings of 2026-11-13 (82 requests), which have no cost receipt.
- Use short targeted real checks between changes. Do not run a month after each subtask.
- The final Jesse–Spamuel, Jesse–Spamuel1–Spamuel alpha, and Spamuel–Spamuel–Minecraft scenarios come only after integrated beta completion.

## Commands

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/inner_thought -t . -b
.venv/Scripts/python.exe -m unittest discover -s tests/unit/workflows/focus_management -t . -b
npx --no-install pyright --project pyrightconfig.json
python -m tests.replay replay --snapshot .runtime/replay/private-continuity-20261114T104837Z
python -m tests.replay replay --snapshot .runtime/replay/deliberate-entry-20261114T110610Z
python -m tests.replay replay --snapshot .runtime/replay/deliberate-reflection-20261114T110440Z
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

A recording made before a prompt change stops at the first changed request. Re-record it; that costs provider money.
A recording made before a `source_fingerprint` rule change also stops; `deliberate-entry-20261114T104553Z`,
`deliberate-reflection-20261114T104401Z`, and `deliberate-reflection-20261114T105637Z` are superseded samples from that period.
A replay overwrites the `replay-*.json` files of its snapshot. Replay a copy when the result may diverge.
The current recordings are `private-continuity-20261114T104837Z`, `deliberate-entry-20261114T110610Z`, and
`deliberate-reflection-20261114T110440Z`; all three replay exactly.
Receipts of the last five passes: `.runtime/beta-b05/displacement-20261114/`, `.runtime/beta-b05/offer-salience/`,
`.runtime/beta-b05/felt-state/`, `.runtime/beta-b05/deliberate-entry/`, and `.runtime/beta-b05/reflection-evidence/`.
The offer selection has a real-endpoint check: `python -m unittest tests.live.test_displaced_offer_salience` (four embedding requests, no chat completion).
