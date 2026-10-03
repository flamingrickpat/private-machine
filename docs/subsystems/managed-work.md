# Managed work and consolidation

## Summary

The supervisor owns managed workflow receipts and worker lifetimes.
Updates take effect at explicit instruction boundaries.
Terminal artifacts enter accepted memory through consolidation.
The steward stays dormant until explicit activation.

Parent: [system map](../concepts/architecture.md).
API: [`runtime.managed_runs.ManagedRunSupervisor`][].

## Input, change, and output

Cognition or Assistant tools request a start, status, update, answer, or cancellation.
The supervisor accepts the request and stores a receipt.
A worker runs the configured manifest.
Events return through the bus to cognition.

The configured targets are `local_task`, `local_task_question`, and `minecraft`.
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

## Traps

Minecraft performs one action per run.
It has no sustained survival policy.
Cancellation is cooperative.
A completed external effect remains complete after cancellation.
MCP transport success alone does not establish tool success.

## Go deeper

Read [workflow runtime](../runtime.md), [one exchange](../workflows/exchange.md), and [`runtime.steward.Steward`][].


For service ownership, task isolation, and cleanup, read the [Minecraft run guide](../workflows/minecraft.md) and [evidence guide](../workflows/minecraft-evidence.md).
