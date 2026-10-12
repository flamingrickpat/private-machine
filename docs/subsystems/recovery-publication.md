# Durable intent and progress publication

## Summary

Accepted actions retain their exact dispatch envelope in Ghost before bus publication.
Managed events retain their original identity before publication.
Restart reconciles these records with accepted run and exchange receipts.
World work remains interrupted until an explicit resume.

Parent: [public publication](public-publication.md).
Related: [body recovery](body-recovery.md) and [managed work](managed-work.md).

## Accepted companion intent

[`modules.MainCogLoop`][] accepts the action schema, GWT state, and frozen dispatch payload in one Ghost batch.
The action metadata stores `dispatch_status: pending`, `dispatch_target`, `dispatch_correlation`, and `dispatch_payload`.
The payload retains the selected action ID, content, arguments, stimuli, memory evidence, workspace, affect, and regulation.
Publication follows the durable commit.
The original action schema ID remains the publication identity.

[`workflows.main_cognition.dispatch_recovery.replay_intents`][] reads pending actions after restart.
An existing managed receipt with the same action ID proves admission.
Recovery acknowledges that intent without launching its worker.
A resolved exchange also proves that its public reply already committed.
Otherwise, recovery publishes the frozen payload with the original action and exchange IDs.
It does not run cognition again to invent a replacement intent.

[`workflows.main_cognition.dispatch_recovery.acknowledge_intent`][] marks durable admission after SelfManagement returns its control receipt.
A normal public commit acknowledges its selected intent in the same batch as speech and exchange resolution.
Updates, cancellation, and resumes use stable request IDs for idempotency.
An acknowledged old cancellation cannot cancel a later resumed epoch when its intent replays.
Start already deduplicates by accepted action ID.
Interrupted body receipts remain evidence for terminal replies.
They never cause automatic mutation replay.

## Managed progress

[`pm.runs.supervisor.ManagedRunReceipt`][] stores pending `events` with their message and acknowledgment status.
The event ID contains the logical run ID, worker epoch, and event sequence.
[`pm.runs.supervisor.ManagedRunSupervisor`][] commits the message before it calls the event sink.
An event-sink failure leaves the pending message in the checkpoint and `control.json`.
If the task outcome already committed, a later publication error preserves that outcome.
Its separate `publication_interrupted` event identifies the delivery failure.

[`workflows.public_speech.reply_recovery.ReplyRecovery`][] consumes `workflow.event` and `workflow.question` messages.
It acknowledges their original event IDs through the supervisor.
Startup replays pending progress through `replay_events`.
This replay creates no worker and changes no world state.
Repeated delivery retains the same ID until acknowledgment.

## State authorities and commit boundaries

Ghost remains the accepted source for companion actions, public exchanges, and managed receipts.
Each task also mirrors its control receipt to `control.json` after the Ghost commit.
This mirror lets a worker observe controls from another local process.
The worker also reads changed checkpoint control rows before authorizing a dispatch.
This covers a crash after the Ghost commit and before mirror replacement.
Unchanged checkpoints use cached managed rows and require only a filesystem timestamp check.
`control_version` rejects a stale mirror write.
`control.lock` orders accepted controls against the gateway's durable dispatch marker.
`worker.lock` prevents two processes from writing the same task evidence.

The mirror does not make the full Ghost checkpoint safe for unrelated simultaneous writers.
Use one active application owner per checkpoint.
Separate instances use separate checkpoints and the shared body registry.
The body lease still prevents concurrent mutation across those instances.

## Public reply recovery

The existing public mechanism retains `publication_status: pending` on committed utterances.
Recovery republishes the same utterance ID.
The consumer acknowledges delivery without creating another turn.
Terminal task replies use retained evidence and the original selected public-reply intent.
Pending input without an accepted intent can still re-enter cognition through its original user record.

The bus provides at-least-once recovery.
A crash after delivery and before acknowledgment can repeat an envelope.
Consumers must deduplicate by action ID, event ID, or public utterance ID.
Neither a published event nor a completed workflow establishes an unverified external effect.
Read [A12 verification](../work_items/verification/a12-body-recovery.md) for the actual failure-injection receipts.
