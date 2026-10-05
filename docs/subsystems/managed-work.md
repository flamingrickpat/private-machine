# Managed work and consolidation

## Summary

The supervisor owns managed workflow receipts and worker lifetimes.
Updates take effect at explicit instruction boundaries.
Progress and terminal artifacts enter accepted memory through consolidation.
The steward stays dormant until explicit activation.

Parent: [system map](../concepts/architecture.md).
API: [`runtime.managed_runs.ManagedRunSupervisor`][].

## Input, change, and output

Cognition or Assistant tools request a start, status, update, answer, or cancellation.
The supervisor accepts the request and stores a receipt.
A worker runs the configured manifest.
Events return through the bus to cognition.

The configured targets are `local_task`, `local_task_question`, `note_task`, and `minecraft`.
`local_task` is a control worker and cannot execute arbitrary note instructions.
`note_task` uses the real pm-coder driver and verifies exact UTF-8 file bytes.
Read [public publication and recovery](public-publication.md) for terminal task replies and note evidence.
The console exposes update, answer, and cancellation for existing runs.
A chat request can ask the companion to start a target through its normal action path.

## State

Receipts distinguish `revision` from `applied_revision`.
An accepted update increments `revision`.
The worker advances `applied_revision` only after it consumes that update.
Updates after the final instruction boundary fail.

Checkpoint restoration changes active receipts to `interrupted`.
It does not resume an external task automatically.
[`runtime.consolidation.Consolidator`][] stores evidence from terminal artifacts.
Repeated consolidation keys do not duplicate accepted memory.
Changed inventories also reuse unchanged artifact memories through their proposal keys.
Read [experience consolidation](memory-consolidation.md) for evidence attribution and concurrency guards.

## Traps

Minecraft defaults to a [sustained specialist session](minecraft-actor.md).
The explicit W8 boundary check retains one action per run.
The actor has a process-local body lease and a native pm-coder history.
Durable ownership across separate processes belongs to A12.
Cancellation is cooperative.
A completed external effect remains complete after cancellation.
MCP transport success alone does not establish tool success.

## Go deeper

Read [workflow runtime](../runtime.md), [one exchange](../workflows/exchange.md), and [`runtime.steward.Steward`][].


For service ownership, task isolation, and cleanup, read the [Minecraft run guide](../workflows/minecraft.md) and [evidence guide](../workflows/minecraft-evidence.md).
