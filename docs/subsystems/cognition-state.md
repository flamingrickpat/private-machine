# Accepted cognition state

## Summary

Main cognition proposes affect, goals, and actions.
A06 makes the accepted path carry those proposals into Ghost state.
An appraisal adds one delta to the history that sets the mental state.
A goal change creates or moves an `Intention` through its lifecycle.
Actions keep a stable identity and a separate success prerequisite.

Parent: [Cognition](cognition.md).
Entry: [`workflows.main_cognition.steps.run`][].
Committer: [`workflows.main_cognition.host.MainCogLoop.publish_output`][].

## Place in the system

[`workflows.main_cognition.host.MainCogLoop`][] builds one request from a bus batch.
Each phase of the cycle adds its typed result to one shared
[`workflows.main_cognition.cycle_work.CycleWork`][].
[`workflows.main_cognition.steps.run`][] validates it at the entry of each phase.
The final work carries the appraisal, the appraisal affect, the goal changes, the
dispatches, and the changed cycle state.
The module applies those proposals inside one Ghost write batch.

## Input → change → output

| Input | Change | Output |
| --- | --- | --- |
| Appraisal report | EMA mental state and one appraisal Feature | Feature in accepted memory |
| Goal change list | New or changed `Intention` knoxel | Durable goal lifecycle |
| Decisions and routes | Stable action IDs, expectations, and prerequisite records | `action.request` or `action.receipt` |
| Startup gap | One system lifecycle Feature | First-cycle stimulus, unless an accepted wait suppresses routine cognition |
| `Sleep` | Wait and cognitive result commit together | Routine inference pauses; outside evidence or timeout ends the pause |

## Appraisal to affect

The appraise phase computes the affect of the cycle (D27).
[`workflows.main_cognition.phases.appraise.update_affect`][] reads the causal feature history of the request.
It derives the appraisal state and its delta with [`workflows.main_cognition.affect.appraise_state`][].
Then it sets the cycle `mental_state` to the EMA readout over the history and the new delta.
[`workflows.main_cognition.stage_commits.apply_appraisal`][] stages one `Feature` with `feature_subtype=Appraisal`.
The Feature carries the appraisal state in `mental_state_appraisal_vec`.
It carries the delta in `mental_state_delta_vec`.
Its metadata carries the appraisal axes, the explicit deltas, the relationship entity id, the feeling in words, and the regulation mode.
The Feature is causal and self-origin (`SourceKind.Self`).
The next cycle reads its delta from the store.

Read [emotional dynamics](emotional-dynamics.md) for the formula, the regulation mode, and the departures from the old code.
The model names axes, for example `goal_congruence`, and deltas, for example `connection`.
`AppraisalResult` accepts only the field names of the appraisal, emotion, and need models.
An unknown name or an out-of-bounds axis is invalid model output, and the repair loop asks the model to fix it.
The math clamps every value to its field bounds, so the state stays in range.

## Goal lifecycle

[`workflows.main_cognition.stage_commits.apply_goal_changes`][] accepts these
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

[`workflows.main_cognition.phases.route.stable_action_id`][] hashes the workflow,
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
[`workflows.main_cognition.host.MainCogLoop`][] suppresses a repeated non-reply start with the same
durable identity.

## Lifecycle evidence

[`pm.app.lifecycle_records.record_startup`][] writes one `SystemMessage` Feature.
The Feature records the boot identity, the last known activity, and the gap.
When a clean shutdown exists, the gap text says so.
When no clean shutdown exists, the gap is uncertain.
[`pm.app.lifecycle_records.record_shutdown`][] writes the clean shutdown event.
The worker-join callback in [`workflows.companion.build_app`][] calls it after all owned workers stop.
[`pm.host.application.Application.join`][] waits for that callback.
An application error or abrupt exit leaves no clean shutdown marker.
The next startup then reports an uncertain end interval.
A repeated boot identity or shutdown identity reuses the accepted event.

At construction, [`workflows.companion.build_app`][] records the boot event.
For real cognition it also puts a `system.lifecycle` envelope on the bus.
[`workflows.main_cognition.host.MainCogLoop`][] accepts that topic in `_stimulus_from`.
The first cycle then appraises the offline interval.
The system never attributes an unknown stop cause to the user.

## State and side effects

The module opens one batch with `new_tick=True`.
It stages the appraisal Feature, the goal intentions, the codelet percepts, the
structures, the attention frame, and each action in that batch.
The GWT turn record of that batch lists the shard changes of the cycle
(see [structures, attention, and the workspace](workspace.md)).
It commits once.
A failure before the commit leaves accepted state unchanged.
A stale revision raises `GhostConflict`, so no partial update lands.
The checkpoint write precedes action publication.
The [wait controller](inference-wait.md) can accept a pause in that same commit.
Its generation fence prevents a late sleep decision from hiding newer input.

[`workflows.main_cognition.host.MainCogLoop`][] reads active goals from accepted memory.
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
Only a successful application join records a clean shutdown.
A hard process kill leaves the interval uncertain by design.

## Go deeper

Read [emotional dynamics](emotional-dynamics.md) for the bounded affect step.
Read [A06 verification](../work_items/verification/a06-cognition-state.md) for the checks and
the receipts.
Read [Cognition](cognition.md) for the stage order.
Read [Ghost and bus](../ghost.md) for the commit boundary.
Read [`workflows.main_cognition.stage_commits`][] for the exact helpers.
