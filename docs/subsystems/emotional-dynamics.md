# Emotional dynamics

## Summary

Emotional dynamics turn accepted events and elapsed time into a bounded mental
state. The state changes attention, goals, regulation, and expressive context.
Two trait profiles can react differently to the same facts.

Parent: [Cognition](cognition.md).
Entry: [`workflows.main_cognition.affect_dynamics.project_affect`][].
Committer: [`workflows.main_cognition.host.MainCogLoop.publish_output`][].

## Place in the system

[`workflows.main_cognition.pipeline.run_cycle`][] runs the eight stages.
After the appraisal stage, the parent computes one affect projection.
The projection feeds the later stages and the parent report.
[`workflows.main_cognition.host.MainCogLoop.publish_output`][] commits the same projected vector.
[`workflows.main_cognition.stage_commits.apply_appraisal`][] writes the
appraisal Feature with the state, the delta, and the causal trace.

## Input → change → output

| Input | Change | Output |
| --- | --- | --- |
| Previous latent vector | Elapsed-time decay | State near baseline |
| Accepted appraisal axes | Core affect and discrete emotions | New bounded state |
| Explicit emotion and need deltas | Field values | Named field changes |
| Expectation error | Valence, joy or sadness, certainty | Comparison effect |
| Personality profile | Gain and decay factors | Different reaction per persona |

## The affect step

[`workflows.main_cognition.affect_dynamics.project_affect`][] applies these inputs in
order:

1. Load the previous state. An empty vector starts from an empty state.
2. Apply exponential decay toward the model baseline. Emotions decay fastest,
   core affect next, and needs slowest. Neuroticism slows the decay of negative
   emotions. Self-control steadies the mood.
3. Apply the canonical appraisal axes. The existing
   [`pm.records.mental_state.compute_state_delta`][] maps the axes
   to core affect and discrete emotions. The step blends the derived value into
   the current state with one EMA gain.
4. Apply explicit emotion and need deltas at full weight. An explicit delta is
   a direct instruction from the appraisal.
5. Apply the expectation error. A negative error lowers valence and raises
   sadness. A positive error raises valence and joy. The magnitude lowers
   certainty.
6. Clamp every field to its declared bounds.

The `emotion_delta` and `need_delta` names use the canonical `StateEmotions`
and `StateNeeds` fields. The `appraisal` names use the `AppraisalGeneral` and
`AppraisalSocial` fields. An unknown name has no effect.

## Personality

[`workflows.main_cognition.affect_dynamics.personality_gains`][] returns a bounded gain
for each named field. Every gain stays in `[0.5, 1.5]`.

- Extraversion, openness, and the approach drive raise positive emotion gains.
- Neuroticism and the inhibition system raise negative emotion gains.
- Self-control and agreeableness lower anger.
- Agreeableness and expressive warmth raise care.
- Affiliation raises connection, and attachment anxiety raises closeness need.

The gains scale the derived and decay paths. They never create an event.
A trait changes the size of a reaction, not the facts.

## Behavior mapping

Accepted affect maps to a regulation mode and named behavior pressures in
[`workflows.main_cognition.affect_dynamics._behavior_mapping`][].

- `reactive`: danger is present. The appraisal `bodily_threat`, the aversive
  pressure, fear, or anger crosses a threshold. Game danger preempts slow
  regulation.
- `reflective`: sadness or withdrawal is high. The companion ruminates or
  considers the situation.
- `blended`: no strong pressure.

The behavior block also carries `attention_focus`, `persistence`,
`exploration`, `affiliation`, and `withdrawal`. Later stages read the block
from the request `affect` field.

## Causal trace

The appraisal Feature metadata holds `affect_trace`. The trace records:

- the elapsed seconds and the two half-lives;
- the personality gains;
- the derived emotions and core values;
- the explicit emotions that changed;
- the expectation error and its effect;
- the regulation mode and the behavior block.

The committed cycle stores the projected vector in
`mental_state_latent_vec`. The appraisal Feature stores the same vector in
`mental_state_appraisal_vec`. The `mental_state_delta_vec` holds the step from
the previous accepted state. A reader can follow one event from the stimulus,
through the appraisal Feature, to the affect trace, to the regulation mode, and
to the action content.

## State and side effects

The step is pure. It reads no Ghost and writes no file.
[`workflows.main_cognition.host.MainCogLoop.publish_output`][] stages the appraisal Feature and the
cycle state in one Ghost batch. A failure before the commit leaves accepted
state unchanged. A stale revision raises `GhostConflict`.

The parent report carries the projected vector. The module reuses that vector
instead of recomputing the step. The stage input and the accepted state cannot
diverge.

## Traps

The `FullMentalState.to_list` round trip does not persist the stable
`state_personality` field or the `state_expression` channel. The dynamics read
the personality from the request, not from the latent vector.

An explicit `emotion_delta` or `need_delta` applies at full weight. Personality
does not scale a value that the model already named.

A model failure raises `StageOutputError` after the retries. It never becomes a
neutral appraisal. A failed turn leaves the previous accepted state.

The decay uses the previous cycle timestamp minus the current time. A long
offline gap therefore decays the mood toward baseline before the first new
appraisal.

## Go deeper

Read [A07 verification](../work_items/verification/a07-emotional-dynamics.md) for the checks and
the live receipts.
Read [accepted cognition state](cognition-state.md) for the commit boundary.
Read [`workflows.main_cognition.affect_dynamics`][] for the exact functions.
Read [Cognition](cognition.md) for the stage order.