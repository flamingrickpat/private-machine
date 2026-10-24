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

## Asynchronous generation and publication

The public context freezes the evidence available to one speaking action.
New user messages and later cognitive states do not invalidate that draft.
Delivery order remains the order shown in chat.
Causal metadata records the earlier context separately.
See [asynchronous conversation review](async-conversation-review.md).

[`workflows.public_speech.publication.validate_public_context`][] rejects changed source meaning.
[`workflows.public_speech.public_context.source_fingerprint`][] excludes exchange, publication, and dispatch bookkeeping from that comparison.
A changed fact, attribution, scope, or other source field still causes a conflict.
The caller holds the memory lock across validation and commit.
The write batch guards the current revisions after successful semantic comparison.

Each utterance retains `generation_user_frontier`, `generation_state_tick`, `generation_source_revisions`, and `responding_to_ids`.
These fields describe the generation context. They do not prescribe chat order or require alternating speakers.
One action can address several pending user messages.
The commit resolves those stimulus records together.
Each action retains `public_reply_id`, which prevents repeated action delivery from publishing duplicate speech.
A different action can produce a follow-up with the same correlation.

Before commit, the module checks current Minecraft tools and connection state.
It also verifies successful task bytes again.
Changed or unreadable task evidence stops publication with a visible error.
The rejected candidate stays in workflow artifacts.

Cancellation before commit prevents acceptance.
Cancellation after an external task effect cannot undo that effect.
A durable pending publication remains available after restart.

[`workflows.public_speech.publication.publish_reply`][] publishes only a committed record.
[`workflows.public_speech.publication.acknowledge_reply`][] marks delivery after the user-input consumer receives that record.
If a crash occurs between delivery and acknowledgment, replay can repeat delivery.
Consumers deduplicate by utterance ID.
The bus does not guarantee exactly-once delivery.

## Rejected replies and errors

[`workflows.public_speech.public_dialogue.PublicDialogue.reject_reply`][] records rejection telemetry and raises a runtime error.
It does not create companion speech or resolve a pending user request.
A blocked false task result stays outside public memory.
[Story](story-speech.md#generation-and-correction) and Assistant each revise a rejected draft, so a blocked draft normally never reaches this method.
The two host guards still call it for a draft that arrives unaccepted or unsupported, and a changed public snapshot calls it with status `stale`.
Unexpected Python failures retain their traceback and reach application shutdown.
Model protocol failures also reach the application.
No host string substitutes for an unsuccessful conversational generation.

[`pm.runs.workflow_module.WorkflowModule`][] handles explicit cancellation exceptions only when a stop signal exists.
An unexpected cancellation still raises.
Other errors remain visible during shutdown.
The operator can inspect the failed workflow and pending input before recovery.
A restart preserves committed speech and pending publication.

## Terminal task replies

[`workflows.public_speech.reply_recovery.ReplyRecovery`][] consumes terminal managed-run events.
It also consumes self-management receipts for rejected requests that never started a worker.
`completed`, `failed`, `cancelled`, and `interrupted` are terminal states.
The original accepted action must select `ToolCallAndReply`.
Self-management retains that choice and action ID in the durable run arguments.

Recovery creates a receipt-based Assistant action even after an independent interim reply resolves the original exchange.
The new action retains the original action as its cause.
The actual task receipt and report accompany the request.
The public reply still receives the same identity, affect, goals, memory, and hallucination review as ordinary dialogue.
An expected task rejection creates an Assistant action with the rejection receipt.
The model writes its explanation in the companion's voice.
Unreadable or malformed internal evidence raises and leaves the request pending.

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
A workflow can also declare `reply_evidence` in its specification.
[`ReplyRecovery`][workflows.public_speech.reply_recovery.ReplyRecovery] adds its fields to the reply evidence.
For `tool_call`, [`reply_evidence`][workflows.tool_call.report.reply_evidence] reads the report again and checks the bytes of each reported local file.
`task_verified` stays false. See the [tool goal guide](../workflows/tool-call.md).
The control worker always lacks that independent note verification.
Expected input, model protocol, timeout, and file errors produce rejected note reports.
Cancellation propagates to the worker, and unexpected logic errors keep their traceback.

## Cognitive protocol recovery

Context selection can only use IDs from the current memory candidate list.
An invalid selection retains its original response in `outbox/selection-rejection.json`.
One scoped model call receives the actual granted IDs and repairs that protocol error.
The host validates the repaired result again.
This procedure does not rewrite public prose or optimize a quality score.

[`workflows.main_cognition.host.MainCogLoop`][] propagates model, stage protocol, and Python failures to application shutdown.
The original request remains pending. No failure message becomes companion dialogue.
Action selection uses the existing appraisal and workspace with hidden reasoning disabled.
A live teasing-choice completion ended before producing usable output with hidden reasoning enabled.
The client rejects length or content-filter termination and reports the finish reason and token usage.
An evidence revision conflict retries the whole cognitive cycle against a new view.
Each attempt retains its original causes and emits `cycle.conflict` telemetry.
Three consecutive conflicts raise. The runtime cannot silently discard a user request.
Retries reuse materialized workflow events, so one observed event cannot become several remembered events.
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
