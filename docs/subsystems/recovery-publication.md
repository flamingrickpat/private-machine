# Publication and clean restart

Ghost owns accepted actions, exchanges, and managed receipts in memory.
Completed main cycles write an atomic checkpoint before releasing actions.
Ordinary input, reply, and task commits do not each write the whole Ghost.
A normal stop saves a checkpoint before it waits on active work.
After the work finishes, PM saves the final state.
A crash can therefore lose recent messages. The previous checkpoint remains consistent.

## Restart behavior

Normal loading opens accepted history and discards executable leftovers.
If a checkpoint contains a public reply and its delivery receipt, normal loading keeps the reply.
It also keeps the exchange resolution for user messages present in accepted history.
The original generation evidence remains available in the raw checkpoint.
Private inputs still follow the usual causal admission rules.

Pending user exchanges become resolved with a discarded-on-restart receipt.
Undelivered publications are discarded. Pending actions and parked workers are cancelled.
The application does not redispatch old inputs, replay old replies, or restore a partly executed cognitive cycle.
The operator can send the same input again or choose a different one.

[`pm.memory.restart.discard_unfinished_work`][] applies these operational dispositions without deleting historical messages.
Raw checkpoint inspection preserves the original view for diagnosis.
`--clean DESTINATION` exports a separate accepted checkpoint with these dispositions.
It leaves the source file unchanged.

## Live delivery

Within a running process, action IDs, event IDs, and utterance IDs still identify messages.
The receiving owner uses those identities to avoid duplicate effects.
A draft has no public effect until accepted publication.
A completed worker does not prove that its requested external effect occurred.

Managed tasks retain their reports and mirror controls to `control.json`.
A newer Ghost control version takes precedence over an older mirror after restart.
Discarding a worker does not reverse an external filesystem or world mutation.
New external work must use current observed state rather than assume that interruption undid it.

Parent: [public publication](public-publication.md).
Related: [accepted cognition](cognition-state.md), [managed work](managed-work.md), and [memory ownership](../ghost.md).
