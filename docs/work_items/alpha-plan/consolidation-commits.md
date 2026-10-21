# Atomic consolidation windows

## Summary

One persisted cursor identifies a fully consolidated prefix of accepted input.
A complete window and its cursor commit together.
An interruption discards the unfinished window and permits a fresh attempt.
No persistent stage flags or separate processing ledger are required.
This contract is planned for A19B.

Parent: [memory consolidation](memory-consolidation.md).
Work item: [A19B](work-items/a19b.md).
Related: [managed work](../../subsystems/managed-work.md) and [memory-store transactions](../../ghost.md).

## Input order and cursor meaning

The cursor means that every eligible source in the accepted prefix received a complete consolidation pass.
It does not identify the highest source ID in the last completed topical cluster.
One window includes all eligible channel partitions within its selected input interval.
Overlapping output times do not change this rule.

[`pm.memory.write_batch.WriteBatch.create`][] reserves record IDs before acceptance.
A batch can reserve ID 101, while another batch commits ID 102 first.
The first batch can then commit after a consolidator reads through 102.
A later query for IDs greater than 102 loses that input.
Reserved cycle ticks have the same ordering concern.

Use a cursor over durable commit order.
[`pm.memory.checkpoint.CommitReceipt`][] already identifies records added or updated by each accepted commit.
Formalize that order in the persistence contract instead of relying on record-ID magnitude or incidental iteration.
Do not change the global ID allocator merely for consolidation convenience.
Do not add per-record processing fields or per-channel cursor maps.

Treat initial seed contents as the initial input segment.
A missing cursor grants no prior completion credit.
Existing copied-text clusters do not establish a completed initial pass.
Later accepted source-content changes must enter a future window.
Bookkeeping-only replacements do not create new experience.
Derived consolidation output does not recursively enter the raw-source stream.

## One transaction per complete window

1. Capture the saved cursor and a bounded prefix of later accepted input.
2. Select the source records from a stable snapshot.
3. Partition sources by channel and reality scope.
4. Produce all required topical summaries, temporal updates, extracted knowledge, and graph records.
5. Stage the output and next cursor in one write batch.
6. Guard the cursor, source revisions, and changed ancestor revisions.
7. Commit the batch through the existing memory store.

All model calls and embedding requests occur outside the memory-store lock.
Only one consolidation worker owns progress for an instance.
A revision conflict rejects the whole window and leaves its cursor unchanged.
Other application commits can continue while the worker prepares proposals.
State updates must preserve newer main and private mental-state values.

[`pm.memory.store.MemoryStore.commit`][] validates and writes a candidate checkpoint before installing accepted memory.
[`pm.memory.checkpoint.write_checkpoint`][] syncs a temporary file before atomic replacement.
Extend that transaction to include the consolidation cursor.
Never save output first and advance the cursor in a second transaction.
The cursor remains memory-store bookkeeping, outside the normalized mental-state vector.

Persist graph nodes and edges with the same memory batch.
Treat graph-search indexes and embedding caches as rebuildable derived data.
A failed index refresh does not erase accepted memory or justify duplicating a complete window.
Inspect and repair the index from committed graph records.

## Interruption and retry

| Interruption point | Restart behavior |
| --- | --- |
| Before the window commits | Old cursor and accepted memory remain. Repeat the complete window |
| During checkpoint preparation | Atomic replacement retains either the old checkpoint or the complete new checkpoint |
| After replacement but before acknowledgment | Reopen the new cursor and output. Do not repeat the accepted window |
| During index refresh | Rebuild the derived index from committed records |

Cancellation stops new consolidation calls and requests cancellation of active inference.
The worker checks cancellation before commit.
An unfinished window's proposals remain private and can be discarded.
Already accepted windows remain useful.
Resume starts a fresh worker from durable state.
It does not need an old model session or a saved stage cursor.

Closing an HTTP request does not prove that a provider stopped generation.
Measure the actual serving behavior during a foreground interruption.
Retain uncertain resource-release outcomes in the real gate.
No fixed conversation latency threshold is introduced.

## Minimal state and evidence

Persist one cursor and the ordinary memory records and source links.
Use existing managed-run receipts and diagnostic logs for operation evidence.
Do not add a second durable lifecycle for every cluster or extraction stage.
Keep source membership and graph edges because they carry memory meaning.

Choose windows that permit a complete pass within actual context limits.
An interruption can repeat work in the current window.
That cost is an accepted simplicity tradeoff.
Verify crash recovery, lower-ID late commits, source revisions, and concurrent chat before delivery.
