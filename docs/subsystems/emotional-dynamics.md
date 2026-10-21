# Emotional dynamics

## Summary

Emotional dynamics turn accepted appraisals and elapsed time into a bounded mental state.
The state changes attention, codelet scheduling, regulation, and the feelings that prompts read.
Decision D27 sets the model: the mental state comes from `compute_state_delta` and an EMA baseline over feature deltas.
The old generation (`ghost_codelets.py`, `_compute_mental_state`) is the reference design.

Parent: [Cognition](cognition.md).
Entry: [`workflows.main_cognition.phases.appraise.update_affect`][].
Math: [`pm.records.mental_state.compute_mental_state`][].
Committer: [`workflows.main_cognition.host.MainCogLoop.publish_output`][].

## Place in the system

The appraise phase of the cycle runs after triage.
The model answers with appraisal axes, optional emotion and need deltas, and an expectation error.
When the expectation check of the same phase rated open expectations, their summed valence impact replaces the expectation error, and their fear impact adds to the fear delta ([decisions, routing, and expectations](decisions-and-routing.md)).
[`workflows.main_cognition.phases.appraise.update_affect`][] then computes the affect of the cycle.
It sets `CycleState.mental_state` and returns an
[`AppraisalAffect`][workflows.main_cognition.schemas.AppraisalAffect].
Later phases read the feelings in words and the regulation mode.
At commit, [`workflows.main_cognition.cycle_commits.apply_appraisal`][] writes the appraisal `Feature`.
The Feature carries the appraisal state and the delta.
The next cycle reads that delta from the store as part of its history.

## Input → change → output

| Input | Change | Output |
| --- | --- | --- |
| Causal feature history | EMA readout at the cycle time | Current readout (base of the appraisal) |
| Appraisal axes | `compute_state_delta` derives core affect, emotions, relationship, and expression | Appraisal state |
| Explicit emotion and need deltas | Add to the derived values, inside the field bounds | Appraisal state |
| Expectation error | Moves valence, and joy or sadness | Appraisal state |
| Appraisal state minus current readout | New sample in the history | Delta on the appraisal Feature |
| History with the new sample | EMA readout up to the cycle time | `CycleState.mental_state` |
| `PersonaConfig.personality_profile` | Stable traits of the state | Personality-dependent expression and attention bias |

## The affect step

[`workflows.main_cognition.phases.appraise.update_affect`][] runs these steps:

1. Compute the half-life.
   [`pm.records.mental_state.reference_half_life_seconds`][] returns
   `MENTAL_STATE_REFERENCE_MINUTES * 60` seconds.
   The old `ghost_codelets.cognitive_cycle` scaled this time frame by `|1 - ego_strength|`.
   That scaling is off, because nothing changes ego strength yet.
2. Compute the current readout.
   [`pm.records.mental_state.compute_mental_state`][] reads the EMA state of the history at the cycle time.
   [`pm.records.mental_state.compute_relationship_state`][] adds the relationship toward the partner.
3. [`workflows.main_cognition.affect.appraise_state`][] copies the current readout and sets the appraisal axes.
   [`pm.records.mental_state.compute_state_delta`][] derives core affect, emotions, expression, and the relationship toward the partner.
   The explicit deltas and the expectation error then add to the derived values.
   The delta is the appraisal state minus that copy: the step from the current readout to the appraised state.
4. Add the delta as a new sample at the cycle time.
   The EMA readout of the extended history becomes the cycle mental state.
5. [`pm.records.mental_state.derive_expression`][] sets the expression readouts from that state.

The partner is the entity of the primary stimulus group.
A self group or a workflow group has no partner, so a world event cannot change a relationship.

## The EMA readout

[`pm.records.mental_state.ema_baselined_normalize`][] computes the readout for each axis:

- The EMA sums core affect, emotions, needs, and the relationship (`EMA_STATE_MODELS`). The other axes read their field defaults.
- `level` is the running sum of the deltas.
- `ema` follows `level` with the half-life. Before each sample it moves by `alpha = 1 - 2 ** (-dt / half_life)`.
- `dev = level - ema` after each sample is one deviation sample.
- A last catch-up step moves `ema` from the newest sample to the cycle time.
  So a feeling fades while no new feature arrives.
- `scale = max(std(dev), 1.4826 * MAD(dev), MENTAL_STATE_SIGMA_FLOOR)`.
- `z = tanh(MENTAL_STATE_TANH_GAIN * dev_now / (2 * scale))`.
- The readout is the field default at `z = 0`. It moves toward the upper bound for `z > 0` and toward the lower bound for `z < 0`.

[`pm.records.mental_state.features_to_history`][] makes the history.
Only causal features count, because only they entered the story.
A feature without a delta also counts, because its deviation sample changes the scale.
[`pm.records.mental_state.compute_relationship_state`][] reads only the samples of one entity.
A sample concerns an entity when the Feature names it in `entity_id` or in the relationship slot of its appraisal vector.

## Alpha failsafe clamps (temporary)

!!! danger "ALPHA RUN ONLY. Fix properly later."
    These clamps are temporary failsafes for the first owner run with the real persona.
    They hide the symptom, not the cause.
    In the A19F review run, one stimulus moved core valence from +0.78 to -1.0 in one tick.
    The real fix is a tuned affect design: sigma floor, gain, half-life, and appraisal calibration.
    After that fix, delete the `ALPHA_*` block in `pm/constants.py` and every use of it.

- `ALPHA_MAX_STIMULUS_DELTA` (1/3): one record moves one axis by a third at most.
  The appraise phase stores the clamped delta.
  The EMA reader also clamps old stored deltas and codelet percept deltas ([`pm.records.mental_state.alpha_clamp_delta`][]).
- `ALPHA_MAX_STATE_STEP` (1/3): each axis of core affect, emotions (boredom too), and needs moves a third at most from the state of the last cycle ([`pm.records.mental_state.alpha_limit_state_step`][]).
  The relationship is limited only when the partner stays the same.
  The limit acts on the readout only, so a lasting change still arrives over some cycles.
- `MENTAL_STATE_SIGMA_FLOOR` is 0.35 instead of 0.25 for the alpha run. One delta of 0.5 now reads as about 0.61 instead of 0.76.

## Departures from the old code

- The delta is the step from the current readout to the appraised state, as in the old `cognitive_cycle`. The raw level is the running sum of all deltas and has no bound. That is intended: only the deviation of the level from its EMA baseline is read out. An earlier port derived the delta against the raw level. Its sign flips came from the old scale floor of 0.001, not from this rule. With that floor, one small delta pinned the readout at a bound.
- The EMA sums only the parts that an appraisal changes. Old features carry deltas on appraisal axes and cognition from the replaced A07 projection. Summed, they put ego strength at 1.0 and so the half-life at 1 second. In the old generation these axes never had deltas, so they read their baseline.
- The old EMA moved after the new delta. After a gap of several half-lives a new event then read as no change. The port moves the EMA first, so a new event always shows.
- The old readout put every 0..1 axis at 0.5 without deviation. The port uses the field default, so a calm emotion reads 0 and a need reads 0.5.
- The old global state kept only the features of entity 2, which was written into the code. The port uses all causal features for the global state and filters by entity only for the relationship.
- The ego-strength time frame is off. The current readout and the cycle state both use the 24-hour half-life.
- The old code read no time after the newest feature. The port adds the catch-up step to the cycle time, so the time between cycles has an effect.
- The relationship entity id is an identifier, not an axis. The delta keeps 0 in its slot, so ids never add up.
- No appraisal axis derives resentment, jealousy, or boredom. `compute_state_delta` keeps their current values, so only an explicit delta changes them.

## Regulation, behavior, and attention

[`workflows.main_cognition.affect.regulation_mode`][] returns one mode:

- `reactive`: the appraised bodily threat or aversive pressure is high, or fear or anger in the accepted state is high. Game danger preempts slow regulation.
- `reflective`: sadness, or the pull to go quiet (`disengage_silence`), is high.
- `blended`: no strong pressure.

The thresholds are in `pm/constants.py`.
The threat axes come from the appraisal state, because the EMA does not sum appraisal axes.

[`workflows.main_cognition.affect.describe_mental_state`][] writes the accepted state as short story text.
It names the strongest emotions, the mood, the relationship toward the partner, the strongest urge to act (from `StateExpressionExt`), where attention drifts (from [`pm.records.mental_state.compute_attention_bias`][]), and the pressing needs.
It uses words, not numbers.
The text goes into the triage-to-decide prompts, the appraisal Feature metadata (`affect_summary`), and the action request.
The public speech context computes the same text from the newest accepted cycle state.

The codelets phase reads the typed state directly.
[`family_state_match`][workflows.main_cognition.codelet_registry.family_state_match] matches it to the codelet families, and [`percept_affect`][workflows.main_cognition.affect.percept_affect] mini-appraises each codelet percept.
A percept does not change the mental state. See [codelets](codelets.md#mini-appraisal).

## State and side effects

The phase computes in memory and writes no store.
[`workflows.main_cognition.host.MainCogLoop.publish_output`][] stages the appraisal Feature and the cycle state in one batch.
A failure before the commit leaves the accepted state unchanged.
The appraisal Feature is causal. The next cycle therefore reads its delta in the history.

## Traps

`AppraisalResult` rejects unknown emotion, need, and axis names, and axis values outside their bounds.
The repair loop asks the model to fix them.
A scripted test answer with an invalid name never becomes valid.
The host then retries the whole cycle forever, so the test hangs.

The scale is the spread of the past deviations. With a short or quiet history the scale is small, so one change reads close to the axis bound.
A change of two scales reads as about 0.9 of the range.

A neutral appraisal derives no emotion, so it sets the raw emotion level to 0.
Resentment, jealousy, boredom, and the needs keep their level until an explicit delta changes them.
Their readout still fades with time, because the EMA baseline catches up.

A level that stays the same for several half-lives reads as the default.
The readout shows change against the recent mood, not the absolute level.

`FullMentalState.to_list` holds no personality and no universal expression channel.
Copy a `FullMentalState` as a model, not through its vector.

## Go deeper

Read [A07 verification](../work_items/verification/a07-emotional-dynamics.md) for the older A07 checks.
Read [accepted cognition state](cognition-state.md) for the commit boundary.
Read [`workflows.main_cognition.affect`][] and [`pm.records.mental_state`][] for the exact functions.
Read the [A19F plan](../work_items/alpha-plan/work-items/a19f.md) for the cycle that uses this state.
