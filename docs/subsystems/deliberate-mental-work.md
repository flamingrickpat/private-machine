# Deliberate mental work

## Summary

The `focus_management` workflow performs private recall through a real pm-coder session.
It proposes goals and temporary focus advice.
Main cognition appraises the result before any goal or focus change.
The workflow cannot change emotions, publish speech, or command the body.

Parent: [cognition](cognition.md).
Related: [private episodes](private-loop.md), [waits](inference-wait.md), and [A18C evidence](../work_items/verification/a18c-control.md).

## Sources and acceptance

[`workflows.focus_management.host.FocusManagement`][] binds the workflow to the shared memory store and model configuration.
It reads the typed action request of main cognition. The task is `arguments.instruction`, else `arguments.instructions`, else the decision text.
The request also carries the shards of the choosing cycle. The agent prompt adds their story text ([`MentalWorkRequest`][workflows.focus_management.schemas.MentalWorkRequest]), so the agent knows what was on the companion's mind.
Its memory tools read a fixed grant.
The goal tool records inspected goal revisions.
Final validation reads that inspected set without adding a new inspection.
The manifest must grant `mental_inspected` to the Python role.
Source conflicts reject the proposal before acceptance.

The accepted result is a private thought with source revisions and lineage.
Main appraisal receives it through `thought.created`.
The goals phase can accept, change, or decline proposed goals.
The attention phase can accept or reject focus advice after appraisal and the codelets.
Accepted advice becomes CSM endorsements with the endorser `deliberate_focus` ([structures, attention, and the workspace](workspace.md)).
`FOCUS_ACTIVATION_WEIGHT`, `FOCUS_MAX_SHARE`, and the configured bias cap bound its influence.
Reserved outside stimuli and urgent concerns retain protection.

## Failures and restart

The agent deadline defaults to 600 seconds through `ControlConfig.mental_timeout_seconds`.
A deadline or expected provider error produces a durable failed result with no plan.
Cancellation discards uncommitted work after agent cleanup.
Programming and validation errors still propagate.

The private result and the action's admission receipt commit together.
A duplicate action cannot repeat a committed session.
[`workflows.focus_management.host.FocusManagement.resume`][] replays an unappraised result after restart.
Accepted main cycle source ids acknowledge the result.
A one-second timer retries publication after a wait or discarded main invocation.
These timer checks perform no inference.
The diagnostic `action.receipt` requests no public reply.

## Verification boundary

The development gate is `dev/work_items/alpha_a18c_mental_gate.py`.
It uses the full Spamuel canon and actual configured services.
It supplies a task at the action boundary, so it cannot establish autonomous action selection.
Original evidence, agent traces, source hashes, and failures remain under a fresh `.runtime` directory.
Human review of recall quality remains separate from successful workflow execution.
