# Minecraft actor controls and ownership

## Summary

One session owns the avatar across planning, action calls, verification, and cleanup.
Updates apply before the next dispatch.
Cancellation prevents later dispatches but cannot undo an action already sent.
Original calls and uncertain outcomes remain in the actor journal.

Parent: [sustained actor](minecraft-actor.md).
Evidence: [A10 verification](minecraft-actor-verification.md).

## Lease boundary

[`runtime.body_lease.BodyLeases`][] owns a process-local table keyed by the configured MCP endpoint.
All application instances in that process use the shared `BODY_LEASES` registry.
A request's `body_id` is a descriptive label and cannot obtain another lease.
The registry normalizes a trailing slash.
A second session receives `BodyBusy` before it sends a world call.
The original owner retains the lease until external cleanup finishes.

The W8 production boundary check also uses this lease.
The public Assistant exposes only `minecraft_observe`, which reads numeric body state.
It grants no direct body mutations.
Actor and Assistant calls share a per-command lock beneath the session lease.
The direct read waits for the current command and then makes its single observation call.
It does not take the actor's mutation lease or resend a busy observation.
Any later direct mutation must acquire the same lease.
Horizon scans rotate the camera and therefore require ownership.
Tool names outside the explicit read set also require ownership.

Process restart and multiple separate PM processes need durable leases and epochs.
A12 owns that work.
An alternate MCP endpoint for the same avatar is not identified by the current process-local key.
Use one configured endpoint per avatar.

## Updates between actions

[`runtime.managed_runs.ManagedRunSupervisor.update`][] durably stores the new instruction and increments `revision`.
Duplicate request IDs retain the original update.
[`runtime.managed_runs.ManagedRunSupervisor.instructions_at_boundary`][] advances `applied_revision` on the worker boundary.

Before each call, the gateway reads the current boundary and checks cancellation.
If the revision changed, the gateway returns `instruction_changed` without sending the planned call.
Its response includes the new instruction and revision.
The actor must choose again from this evidence.
The journal retains the rejected call and changed instruction.
This ordering prevents an already selected action from silently ignoring the update.
An unchanged boundary returns its current revision without another checkpoint commit.
This avoids rewriting the entire companion history before every tool call.

Updates remain open throughout the native session.
The host closes the final boundary before its independent observation and report.
If a final instruction changed after the actor's last boundary, the completion candidate becomes `yield`.
A later update receives an explicit final-boundary rejection.

## Cancellation and uncertain calls

[`runtime.managed_runs.ManagedRunSupervisor.terminate`][] sets the code-owned cancellation event.
It records cancellation and preserves task files.
The pm-coder driver cancels supported asynchronous inference and awaits session cleanup.
The gateway rejects later calls before dispatch.

Synchronous external work can continue after an MCP client disconnects.
[`runtime.minecraft_actor.ActorTool.run`][] waits for that call's thread before completing cancellation.
The gateway's final cleanup also waits for the dispatch lock.
Only then can the body lease become available.
Blocks, inventory, position, and time do not roll back.

An external transport exception produces an `uncertain` journal entry.
It records the exact tool, arguments, revision, time, and error.
The gateway sets cancellation and sends no automatic retry.
A subsequent session must obtain fresh world state.
The task cannot publish an accepted success after cancellation.

The scoped MCP registry stops its Uvicorn accept loop before closing its socket.
It closes its temporary listener on success, error, and cancellation.
The external Minecraft and MCP services remain owned by their operator or live harness.

## Direct inventory dialogue

Action selection uses the normal LLM.
A clear inventory question selects `dialogue_assistant` with `arguments.minecraft_read: inventory`.
It does not select the managed `minecraft` workflow or require Laya.
[`modules.AssistantDialogue.prepare_input`][] retains that argument in the typed request.

[`workflows.dialogue_assistant.steps.run`][] performs exactly one `minecraft_observe` call for this route.
It uses `include_image: false`, appends the actual receipt, and generates a public answer with no further tools.
Claim review receives the executed receipt beside the accepted public context.
The observation receipt also retains the actual body owner and its managed-run receipt.
This evidence supports a statement that the wood task remains active.
Public claims cite executed calls through `tool_receipt_ids`, using the actual `tool_call_id`.
Numeric `evidence_ids` remain reserved for supplied Ghost records.
A successful observation can establish its current Minecraft snapshot.
An absent, denied, or failed receipt cannot establish that condition.
This namespace does not replace historical sources or weaken unknown-record checks.
An observation error becomes explicit tool evidence and cannot establish an invented inventory.
The inventory is a snapshot and can change while the actor continues.
The synthesized assistant tool call includes `type: function` and serialized JSON arguments.
A10 captured an actual HTTP 500 when the type field was absent.
[`runtime.model_io.rejected_message_protocol`][] treats that deterministic parser rejection as a permanent protocol error.
Other transient server failures retain cancellation-aware retries.
The failure classifier does not expose response bodies in public error text.
The role manifest grants the direct observation tool explicitly.
Raw tool-call markup cannot pass as a public reply.
The speech check runs before claim review and publication.

## Inspect a failure

1. Read the durable managed receipt and its `latest_event`.
2. Read `scratch/actor-journal.jsonl` in the task directory.
3. Pair each `dispatch` with its `result` or `uncertain` entry.
4. Read rejected instruction calls before interpreting a missing world effect.
5. Read the kernel journal, actor trace, and native pm-coder session artifacts.
6. Read `outbox/report.yaml` and compare the verification with the original observations.
7. If an external outcome is uncertain, obtain fresh state before another mutation.

The journal distinguishes transport completion, game progress, and leaf acceptance.
A successful harvest with zero logs cannot add a completed tree.
A reached movement result cannot establish the return without final coordinates.
