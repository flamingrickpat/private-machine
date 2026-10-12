# A07 verification

## Summary

A07 evidence has three parts: deterministic contract tests, one live model
sequence with the real seed and a real checkpoint, and the unaffected
foundation suite. The live sequence drove disappointment, a subject change, a
return, and a repair offer. It recorded the internal cause beside the public
behavior.

Parent: [Emotional dynamics](../../subsystems/emotional-dynamics.md).
Gate: `dev/work_items/alpha_a07_affect_gate.py`.

## Deterministic contract tests

`tests/test_a07_affect_dynamics.py` covers the affect path without a model.

| Check | What it proves |
| --- | --- |
| Decay | Mood returns toward baseline and stays in bounds |
| Mixed state | Two emotions coexist and accumulation stays bounded |
| Persona contrast | Two trait profiles differ on the same facts |
| Gain bounds | Every personality gain stays in `[0.5, 1.5]` |
| Danger | A threat forces the reactive regulation mode |
| Expectation | A negative expectation moves mood with no observed fact |
| Pipeline trace | The report links event, affect, and regulation |
| Stage affect | Later stages receive the accepted affect block |
| Committed state | The appraisal Feature carries the trace and the behavior |
| Need delta | An explicit need delta applies at full weight |
| Restart | Affect survives a checkpoint restart |
| Lifecycle | A startup event is appraised, not treated as neutral |

Run:

```powershell
.venv/Scripts/python.exe -m unittest tests.test_a07_affect_dynamics -v
```

Result: 12 passed.

## Live model sequence

The gate used the local chat model, the real Spamuel seed, and a checkpoint
under `.runtime/alpha/20261004T023706Z-a07/`. It submitted four turns:

1. a disappointment;
2. an unrelated subject change;
3. a return to the disappointment;
4. a repair offer.

Observed accepted state (sadness, valence, regulation):

| Turn | Sadness | Valence | Regulation | Public reply |
| --- | --- | --- | --- | --- |
| Disappointment | 0.806 | -0.860 | reflective | yes |
| Subject change | 0.360 | -0.209 | blended | yes |
| Return | 0.572 | -0.310 | reflective | yes |
| Repair | 0.312 | 0.190 | blended | yes |

The run wrote five real appraisal Features. Each carried an `affect_trace`
with the elapsed seconds, the personality gains, the derived and explicit
emotions, the expectation error, and the behavior block.
The recorded recovery was a sadness drop of 0.494 and a valence rise of 1.050.
The gate recorded the recovery and did not require it.

The boot reported an uncertain stop and a gap of about 3027.5 hours.
The seed had about 800 cycles, so the cycle count reached 812.

The cheerful fixture ran the first disappointment turn as a separate control.
It reached sadness 0.713 and valence -0.816 with the reactive regulation mode,
which differs from the reflective Spamuel reaction. The Story evaluator rejected
that one reply, so the control produced no public reply. The gate records the
reply and does not require it, because public voice is A09's concern.

## Foundation and package checks

- Root foundation suite: recorded in `foundation-final.log`.
- The seed `test_data/personas/spamuel/canon.yaml` is unchanged.
- The gate uses a separate checkpoint.

## Failure and recovery checks

- A model failure raises a stage error and leaves the previous state.
- A stale batch raises `GhostConflict` and applies no partial change.
- An abrupt exit leaves no clean shutdown event; the next boot reports it.

## Remaining limits

The behavior mapping maps affect to regulation and named pressures. It does not
yet prove that every pressure changes the final action.
The `state_personality` field and the universal `state_expression` channel do
not persist in the latent vector.
Long-term weeks-scale mood drift is not tested.
The cheerful control produced no public reply. The Story evaluator rejected the
candidate. A09 owns public voice.
The human immersion rating remains with the operator.

## Go deeper

Read [Emotional dynamics](../../subsystems/emotional-dynamics.md) for the mechanism.
Read [Accepted cognition state](../../subsystems/cognition-state.md) for the commit boundary.
Read [Cognition](../../subsystems/cognition.md) for the stage order.