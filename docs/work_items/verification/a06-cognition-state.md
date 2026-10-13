# A06 verification

## Summary

A06 evidence has three parts: deterministic contract tests, one live model
gate with the real seed and a real checkpoint, and the unaffected foundation
suite. The live gate changed real affect and created real active goals. Both
survived a clean restart.

Parent: [Accepted cognition state](../../subsystems/cognition-state.md).
Gate: `dev/work_items/alpha_a06_state_gate.py`.

## Deterministic contract tests

`tests/unit/workflows/main_cognition/test_cognition_state.py` covers the state path without a model.

| Check | What it proves |
| --- | --- |
| Appraisal commit | One appraisal Feature and a changed latent state land |
| Report carry | The parent report carries appraisal, goals, and origins |
| Goal add and complete | An active intention appears, then closes |
| Refuse and defer | Both keep distinct durable statuses |
| Unknown change | An unsupported kind changes no state |
| Old intention | Text identity resolves a goal without a `goal_key` |
| Queued prerequisite | A queued action does not release dependent work |
| Duplicate start | A repeated start identity does not start a task again |
| Task evidence | A workflow event keeps progress and evidence |
| Thought origin | A thought Feature is `SourceKind.Self` |
| Lifecycle | Uncertain gap, clean gap, and no duplicate boot event |
| Checkpoint restart | An active intention survives a reload |
| Stale batch | A `GhostConflict` prevents a partial affect or goal update |

Run:

```powershell
.venv/Scripts/python.exe -m unittest tests.unit.workflows.main_cognition.test_cognition_state -v
```

Result: 17 passed.

## Live model gate

The gate used the local chat model, the real Spamuel seed, and a checkpoint
under `.runtime/alpha/20261003T233753Z-a06/`.
It submitted one message about a finished treehouse and a requested stone
house by the river.
It then recorded a clean shutdown and restarted from the same checkpoint.

Observed accepted state:

- Boot A reported an uncertain end and a gap of 3026.6 hours.
- Two real appraisal Features committed. Joy, pride, and connection changed.
- Three real active intentions committed.
- One intention was `Remember the river stone house plan`.
- The clean shutdown event committed.
- Boot B reported a clean stop and a gap of 9.6 seconds.
- All three active goals survived the restart.
- The restart reply discussed the river plan.

The gate asserts shutdown presence, the clean gap, and the surviving river
goal.
It reads real report and model output only.
It uses no fixed model response.

Run:

```powershell
uv run --locked python dev/work_items/alpha_a06_state_gate.py `
  --config config.alpha.example.yaml --out .runtime/alpha/<stamp>-a06
```

Result: `ok: true`.

## Foundation and package checks

- Root foundation suite: 285 tests passed in 488.286 seconds.
- pm-workflows package: 199 passed in 58.13 seconds.
- The seed `test_data/personas/spamuel/canon.yaml` is unchanged.
- The gate uses a separate checkpoint.

## Failure and recovery checks

- A clean stop writes one `clean_shutdown` event.
- A repeated boot identity reuses the accepted event.
- An abrupt exit writes no clean event. The next boot reports uncertainty.
- A stale batch raises `GhostConflict` and applies no partial change.
- A failed prerequisite rejects a dependent action.
- A missing prerequisite queues it.

## Remaining limits

The live gate ran one goal and emotion scenario.
It did not test long-term mood decay or behavior change; A07 owns that.
It did not test queued-action requeue after a receipt; A12 owns that.
The restart test used one process pair, not repeated crash recovery.
The manual immersion rating remains with the operator.

## Go deeper

Read [Accepted cognition state](../../subsystems/cognition-state.md) for the mechanism.
Read [Cognition](../../subsystems/cognition.md) for the stage order.
Read [Ghost and bus](../../ghost.md) for the commit boundary.
