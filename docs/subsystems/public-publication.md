# Public publication and recovery

## Summary

The canonical Ghost owns pending exchanges and committed public utterances.
A public commit resolves its exchange atomically.
Publication follows that commit.
Checkpoint restart can replay an unacknowledged utterance or dispatch a selected terminal task reply.

Parent: [public dialogue](public-dialogue.md).
Related: [operator service](console.md), [managed work](managed-work.md), and [A09 verification](../work_items/verification/a09-public-dialogue.md).
Accepted companion actions and managed progress also use [durable publication recovery](recovery-publication.md).

## Input and durable ownership

[`workflows.public_speech.user_input.UserInputProxy.submit`][] assigns the correlation ID before it stores the user's utterance.
The Feature retains `exchange_correlation`, `exchange_status=pending`, and the console session name.
These fields live in the existing Ghost checkpoint.
There is no separate state authority or broker.

[`workflows.public_speech.publication.commit_public_reply`][] creates the companion utterance and resolves the corresponding user record in one Ghost batch.
The user record retains `public_reply_id` and any recovery outcome.
The companion record retains `reply_correlation`, `publication_status=pending`, and `factual_promotion=false`.
Canonical records are copied through a snapshot before replacement.
A failed commit cannot alter their live metadata.

## Stale generation and publication

The public context retains source revisions, the latest user ID, and the accepted internal-state tick.
Publication rejects a changed source, a newer user stimulus, or a newer accepted internal state.
[`workflows.public_speech.publication.validate_public_context`][] checks these guards before semantic or protocol recovery.
A fresh recovery snapshot cannot erase a conflict with the original frozen sources.
Before commit, the module probes current Minecraft availability and compares tool names with the frozen context.
A changed connection or tool list makes the reply stale.
It also verifies successful task bytes again.
Changed bytes or unreadable evidence invalidate the earlier verification before recovery chooses its public text.
The whole candidate remains outside memory after rejection.
[`workflows.public_speech.public_dialogue.PublicDialogue`][] records the stale receipt and publishes a separate recovery utterance.

Rejected semantic claims and protocol failures retain separate recovery outcomes.
The normal public candidate remains in workflow artifacts.
The recovery receipt retains the failure reason, outcome, and committed reply ID.
A task recovery can report independently verified note bytes even when its conversational candidate fails.
That fallback does not convert the rejected prose into a passing conversation.

Cancellation before commit prevents acceptance.
Cancellation after an external task effect cannot undo that effect.
A durable pending publication remains available after restart.

[`workflows.public_speech.publication.publish_reply`][] publishes only an already committed record.
[`workflows.public_speech.publication.acknowledge_reply`][] marks delivery after the user-input consumer receives that record.
The console emits the correlated `chat.reply` before this acknowledgement.
The event includes `knoxel_id` for consumer deduplication and `recovery_outcome` for failure inspection.
If a crash occurs between delivery and acknowledgement, replay can repeat delivery.
Consumers must deduplicate by the public utterance ID.
The design does not claim exactly-once bus delivery.

## Rejected replies

A reply reaches the rejection path only when the workflow marks it rejected.
For Story, that means its best draft still holds a blocking violation (`false_task_result`) after all attempts.
See [Story speech](story-speech.md#generation-attempts).
The receipt keeps the rejection reason, and the host publishes the explicit recovery text.
A task exchange instead publishes the actual task state from its receipt.

The A09 grounding recovery was removed on 2026-10-13.
It rewrote rejected replies in a clinical "you reported" voice, and its own review often failed as well.
An original protocol failure still validates its source guards and then publishes explicit host failure text.
Changed source guards still produce the stale reply.
No exchange depends on an unlimited model retry.

## Terminal task replies

[`workflows.public_speech.reply_recovery.ReplyRecovery`][] consumes terminal managed-run events.
It also consumes self-management receipts for rejected requests that never started a worker.
`completed`, `failed`, `cancelled`, and `interrupted` are terminal states.
The original accepted action must select `ToolCallAndReply`.
Self-management retains that choice and action ID in the durable run arguments.

If the original exchange remains pending, recovery creates a receipt-based Assistant action.
The new action retains the original action as its cause.
The actual task receipt and report accompany the request.
The public reply still receives the same identity, affect, goals, memory, and hallucination review as ordinary dialogue.
If task evidence cannot be read or verified, recovery publishes an explicit failure result.
It does not leave the exchange pending.

A completed `local_task` proves that the control worker finished.
Its artifacts do not establish arbitrary instruction execution.
The public receipt must preserve that limit.

[`workflows.note_task.steps.run`][] provides actual agent-backed note execution.
It leases the configured normal chat connection through the existing model-session provider.
The installed pm-coder driver writes the requested file in the task outbox.
Its normal automatic compaction remains available.
The host compares the actual file bytes with the requested UTF-8 bytes.
Self-management retains the original user request and its source ID in the run arguments.
The note specification comes from that original request when it exists.
Rewritten action instructions cannot override the requested file contents.
The extractor first determines whether that original message requests a new note operation.
An apology, affection, frustration, or completion event cannot become note content by default.
A rejected extraction stops before the agent starts or writes a file.
The report retains expected and actual hexadecimal bytes, SHA-256, session reference, and verification status.

[`workflows.public_speech.reply_recovery.task_evidence`][] reads the report and verifies note bytes again before public dispatch.
A changed file or extra newline invalidates task verification.
The control worker always lacks that independent note verification.
Expected input, model protocol, timeout, and file errors produce rejected note reports.
Cancellation propagates to the worker, and unexpected logic errors keep their traceback.

## Cognitive protocol recovery

Context selection can only use IDs from the current memory candidate list.
An invalid selection retains its original response in `outbox/selection-rejection.json`.
One scoped model call receives the actual granted IDs and repairs that protocol error.
The host validates the repaired result again.
This procedure does not rewrite public prose or optimize a quality score.

[`workflows.main_cognition.host.MainCogLoop`][] closes the original user exchange after a model or stage protocol failure.
Its receipt retains the error, and its public recovery states that no task result was verified.
Logic errors still reach application shutdown.
Action selection uses the existing appraisal and workspace with hidden reasoning disabled.
A live teasing-choice completion ended before producing usable output with hidden reasoning enabled.
The client rejects length or content-filter termination and reports the finish reason and token usage.
An evidence revision conflict rejects the stale cognitive cycle without shutting down the application.
This includes exchange-marker changes caused by public resolution while a task event cycle reads that original source.
The rejection emits a concrete receipt; an unchanged queued user input can still run in the next cycle.
When a batch contains task events and a new user input, the latest user input owns the cycle's default correlation.
Individual stimuli keep their original correlations for attribution.
Goal management and action selection also receive the exact current stimuli and sixteen recent managed-run receipts.
Completed task goals must close instead of starting again during later conversation.
These receipts describe run state; completion alone cannot establish successful requested work.
An older task event cannot take the cause of the new user's request.
Self-management operations also retain their current action correlation when they request a public result.
Missing original task reply intent produces an explicit failure result instead of a silent pending exchange.

## Restart and diagnostics

The console reconstructs pending sessions from durable exchange records.
Startup activates recovery synchronously after module subscriptions exist, before it queues the boot stimulus.
Pending original inputs and the boot event then enter the same initial mailbox batch.
This ordering prevents startup from executing a remembered pending request without its original correlation.
The `public.resume` topic remains available for explicit recovery requests.
Recovery replays pending public Features and inspects terminal task receipts.
A pending input without a selected managed run is dispatched again through cognition.
Recovery reuses its original user record and correlation.
It does not manufacture another user utterance.
The original source seed remains unchanged.

For a pending exchange, inspect its correlation ID and `exchange_status`.
For a terminal task, inspect `_public_reply_requested`, `_public_action_schema_id`, and `parent_correlation`.
For a rejected reply, inspect `action.receipt.recovery_outcome` and the linked public Feature.
For a note, compare `actual_hex` with the real file bytes.

The in-memory bus still has no durable message queue.
Checkpoint markers provide recovery for the documented public boundary.
Other queued prerequisites remain A12 work.
The second private conscious loop remains A17 work.

Go deeper: [A09 verification](../work_items/verification/a09-public-dialogue.md) and [Ghost state](../ghost.md).
