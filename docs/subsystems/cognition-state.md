# Accepted cognition state

## Summary

Main cognition proposes affect, goals, and actions.
A06 makes the accepted path carry those proposals into Ghost state.
An appraisal changes the latent mental state.
A goal change creates or moves an `Intention` through its lifecycle.
Actions keep a stable identity and a separate success prerequisite.

Parent: [Cognition](cognition.md).
Entry: [`workflows.main_cognition.pipeline.run_cycle`][].
Committer: [`modules.MainCogLoop.publish_output`][].

## Place in the system

[`modules.MainCogLoop`][] builds one request from a bus batch.
The seven child stages return typed reports.
[`workflows.main_cognition.pipeline.run_cycle`][] collects them into one
[`workflows.main_cognition.schemas.CognitiveReport`][].
The report now carries the appraisal, the goal changes, and the stimulus
origins.
The module applies those proposals inside one Ghost write batch.

## Input → change → output

| Input | Change | Output |
| --- | --- | --- |
| Appraisal report | New latent state and one appraisal Feature | Feature in accepted memory |
| Goal change list | New or changed `Intention` knoxel | Durable goal lifecycle |
| Action selection | Stable action IDs and prerequisite records | `action.request` or `action.receipt` |
| Startup gap | One system lifecycle Feature | First-cycle stimulus |

## Appraisal to affect

[`workflows.main_cognition.pipeline.run_cycle`][] computes one affect
projection after the appraisal stage.
[`pm_core.model.affect_dynamics.project_affect`][] applies elapsed-time decay,
personality gains, canonical appraisal axes, explicit emotion and need deltas,
and an expectation error.
[`workflows.main_cognition.state_changes.apply_appraisal`][] stages the result
as one `Feature` with `feature_subtype=Appraisal`.
The Feature carries the new total state in `mental_state_appraisal_vec`.
It carries the difference in `mental_state_delta_vec`.
Its metadata carries the causal `affect_trace` and the regulation mode.
The module stores the same new state in the cycle
`mental_state_latent_vec`.
The appraisal Feature is self-origin (`SourceKind.Self`).

Read [emotional dynamics](emotional-dynamics.md) for the step order, the
personality gains, and the behavior mapping.
The model names axes, for example `goal_congruence` or `connection`.
The code maps each name onto the existing appraisal, emotion, or need field.
An unknown name has no effect.
The math uses bounded values, so state stays in range.

## Goal lifecycle

[`workflows.main_cognition.state_changes.apply_goal_changes`][] accepts these
change kinds:

- `add` starts an active goal.
- `complete` closes a satisfied goal.
- `reprioritize` changes the salience of a goal.
- `defer` pauses a goal without closing it.
- `refuse` declines a requested goal and keeps the reason.
- `retain` leaves an existing goal unchanged.

Each intention has a stable `goal_key`.
The key comes from the `goal_key` field or from normalized goal text.
The lookup uses identity, not a slice of the text.
A repeated `add` with the same identity does not create a second intention.
A missing `complete` never invents a completion.
A `defer` or `refuse` of a new request is durable evidence.
Active intentions alone steer work.
The request builder reads only active goals.

## Stable actions and prerequisites

[`workflows.main_cognition.pipeline.stable_action_id`][] hashes the workflow,
action type, content, and arguments.
The same intent receives the same ID across cycles.
Two dependency kinds stay apart:

- `depends_on` only orders dispatch inside one selection.
- `requires_completed` needs a `completed` receipt.

[`workflows.main_cognition.dispatch.resolve_same_cycle`][] releases ordering
dependencies.
[`workflows.main_cognition.dispatch.resolve_prerequisites`][] gates success
dependencies.
A missing prerequisite leaves the action `queued`.
A failed prerequisite rejects it.
A queued action never starts dependent work.
[`modules.MainCogLoop`][] suppresses a repeated non-reply start with the same
durable identity.

## Lifecycle evidence

[`pm.app.lifecycle_records.record_startup`][] writes one `SystemMessage` Feature.
The Feature records the boot identity, the last known activity, and the gap.
When a clean shutdown exists, the gap text says so.
When no clean shutdown exists, the gap is uncertain.
[`pm.app.lifecycle_records.record_shutdown`][] writes the clean shutdown event.
[`modules.MainCogLoop.on_terminate`][] calls it on a normal stop.
An abrupt exit never reaches that hook.
The next startup then reports an uncertain end interval.
A repeated boot identity or shutdown identity reuses the accepted event.

At construction, [`main.build_app`][] records the boot event.
For real cognition it also puts a `system.lifecycle` envelope on the bus.
[`modules.MainCogLoop`][] accepts that topic in `_stimulus_from`.
The first cycle then appraises the offline interval.
The system never attributes an unknown stop cause to the user.

## State and side effects

The module opens one batch with `new_tick=True`.
It stages the appraisal Feature, the goal intentions, the attention frame, and
each action in that batch.
It commits once.
A failure before the commit leaves accepted state unchanged.
A stale revision raises `GhostConflict`, so no partial update lands.
The checkpoint write precedes action publication.

[`modules.MainCogLoop`][] reads active goals from accepted memory.
It writes intentions and Features through the Ghost batch.
It sends `action.request`, `action.receipt`, and `gwt.updated` messages.
It sends no model call in the commit path.

## Traps

The pipeline maps unknown emotion or need names to no change.
A goal without a stable key resolves by normalized text only.
A queued prerequisite is not retried inside this item.
A12 owns publication recovery and requeue.
The current model still returns `DEFAULT_WORKFLOWS` when no configuration
catalogue is supplied.
Only a normal module stop records a clean shutdown.
A hard process kill leaves the interval uncertain by design.

## Go deeper

Read [emotional dynamics](emotional-dynamics.md) for the bounded affect step.
Read [A06 verification](../work_items/verification/a06-cognition-state.md) for the checks and
the receipts.
Read [Cognition](cognition.md) for the stage order.
Read [Ghost and bus](../ghost.md) for the commit boundary.
Read [`workflows.main_cognition.state_changes`][] for the exact helpers.
