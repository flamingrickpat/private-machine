# Managed work and consolidation

## Summary

The supervisor owns managed workflow receipts and worker lifetimes.
Updates take effect at explicit instruction boundaries.
Progress and terminal artifacts enter accepted memory through consolidation.
The steward stays dormant until explicit activation.

Parent: [system map](../concepts/architecture.md).
API: [`pm.runs.supervisor.ManagedRunSupervisor`][].

## Input, change, and output

Cognition or Assistant tools request a start, status, update, answer, or cancellation.
The supervisor accepts the request and stores a receipt.
A worker runs the configured manifest.
Events return through the bus to cognition.
An explicit [singleton setting](managed-singletons.md) limits active runs for a workflow or shared resource key.
A busy start returns its owner as workflow evidence. Cognition chooses the next management action.

The configured targets include `local_task`, `local_task_question`, `note_task`, `minecraft`, and `minecraft_project`.
`local_task` is a control worker and cannot execute arbitrary note instructions.
`note_task` uses the real pm-coder driver and verifies exact UTF-8 file bytes.
`minecraft_project` runs [durable subtasks](minecraft-tasks.md) with fresh actors and independent world verification.
Read [public publication and recovery](public-publication.md) for terminal task replies and note evidence.
The console exposes update, answer, and cancellation for existing runs.
It also exposes [explicit body recovery](body-recovery.md) after interruption, cancellation, or failure.
A chat request can ask the companion to start a target through its normal action path.

## State

Receipts distinguish `revision` from `applied_revision`.
An accepted update increments `revision`.
The worker advances `applied_revision` only after it consumes that update.
Updates after the final instruction boundary fail.

Checkpoint restoration changes active receipts to `interrupted`.
It does not resume an external task automatically.
[`workflows.memory_consolidation.artifact_consolidation.Consolidator`][] stores evidence from terminal artifacts.
Repeated consolidation keys do not duplicate accepted memory.
Changed inventories also reuse unchanged artifact memories through their proposal keys.
Read [experience consolidation](memory-consolidation.md) for evidence attribution and concurrency guards.

## Application shutdown

The companion assembly registers worker hooks on [`pm.host.application.Application`][].
Application shutdown stops bus modules and calls [`pm.runs.supervisor.ManagedRunSupervisor.stop`][].
That method fences new worker launches, signals every owned cancellation event, and wakes instruction waiters.
It does not join a worker from the worker's own error callback.

Application join first waits for bus modules.
It then calls [`pm.runs.supervisor.ManagedRunSupervisor.shutdown`][].
That method records cancellation for remaining active or waiting runs and joins every worker.
The caller must keep Minecraft and other external services available until join returns.
This order lets uncertain calls finish their cleanup before the body lease becomes available.
Cancellation cannot reverse a completed world effect.

The assembly records a clean shutdown only after worker cleanup succeeds and the application has no recorded error.
An abrupt process exit or application fault cannot manufacture a clean ending.
Repeated cleanup calls preserve terminal receipts and the original boot identity.

Worker faults keep their traceback even when a cancellation signal exists.
Only an explicit cancellation exception with the matching signal counts as expected cancellation.
A failure to record the worker error adds a diagnostic note and reaches the application.
A terminal consolidation hook failure also reaches the application while the original task outcome remains intact.

## Traps

Minecraft defaults to a [sustained specialist session](minecraft-actor.md).
The explicit W8 boundary check retains one action per run.
The actor has a shared [body lease](body-leases.md) and a native pm-coder history.
Durable ownership and explicit recovery use the delivered A12 implementation.
Projects publish one [gameplay milestone](gameplay-evidence.md#meaningful-progress-publication) per completed actor attempt.
Cancellation is cooperative.
A completed external effect remains complete after cancellation.
MCP transport success alone does not establish tool success.
The Minecraft adapter raises [`TaskServiceError`][pm.runs.errors.TaskServiceError] for a failed connection or transport.
The supervisor retains a failed receipt and lets other work continue.
A late `BodyBusy` also remains a task failure, with its contention detail available to cognition.
A lost response leaves external effects uncertain. The supervisor does not repeat the command.
Cancellation still propagates. Unexpected logic errors still reach the application's error handler.

## Go deeper

Read [workflow runtime](../runtime.md), [one exchange](../workflows/exchange.md), and [`workflows.steward.gate.Steward`][].


For service ownership, task isolation, and cleanup, read the [Minecraft run guide](../workflows/minecraft.md) and [evidence guide](../workflows/minecraft-evidence.md).
