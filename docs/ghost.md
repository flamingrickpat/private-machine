# Ghost commits and bus delivery

## Summary

Ghost accepts a complete batch of memory changes after its checkpoint save succeeds.
Each bus subscriber receives an independent JSON copy.
The same module `step()` supports direct debugging and threaded operation.

Parent: [project README](index.md).

## Accepted state

The [`pm.memory.store.MemoryStore`][] owns accepted records, indexes, GWT records, revisions, counters, and commit receipts.
The store retains each loaded knoxel instance. Reads and views share those records and their central owner.
A snapshot captures membership and revisions. Its record values remain shared.

A [`pm.memory.write_batch.WriteBatch`][] captures record references and revisions and collects private proposals.
New records reserve integer IDs under the Ghost lock.
A discarded batch can leave ID gaps.
Other batches cannot read its proposals.

`snapshot(visible_ids)` returns shared records within the supplied set.
An empty set exposes no records. `None` selects all records for trusted host code.
A narrower view also excludes hidden revisions and GWT records that reference hidden memory.
The virtual Bash target uses this snapshot for each invocation.

## Commit order

The service performs these steps under one commit lock:

1. Reject cancellation, invalid references, or stale replacement revisions.
2. Collect typed state with the existing records and the staged changes.
3. Write and flush a temporary sibling of the checkpoint.
4. Replace the previous checkpoint with the temporary file.
5. Install only the new or replaced records and update their indexes.

The module publishes its bus output after the commit returns.
No model or network call runs under this lock.

An unrelated append does not cause a conflict.
Two replacements of the same revision produce one winner and one `GhostConflict`.
A failed save preserves accepted memory and the previous checkpoint.
The error reaches the direct caller or the application error handler.

Unchanged records retain object identity across every commit, including receipt-only commits.
No commit serializes records into dictionaries and then reconstructs the store.

A repeated commit ID returns its original receipt without duplicate memories.
Callers use the receipt's IDs, not IDs from a repeated attempt.
Cancellation before checkpoint replacement rejects the batch.
Cancellation after replacement does not undo an accepted commit.

## Direct debugger example

```python
from pathlib import Path
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType

ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
              checkpoint=Path(".runtime/example-ghost.pickle"))
batch = ghost.begin_batch("example-turn", new_tick=True)
proposal = batch.create(Feature(feature_type=FeatureType.Dialogue, content="Hello, Spamuel."))
assert ghost.get_knoxel_by_id(proposal.id) is None
assert batch.view().get(proposal.id).content == "Hello, Spamuel."
receipt = ghost.commit(batch)  # breakpoint: validation, disk replacement, installation
restarted = MemoryStore.from_checkpoint(ghost.checkpoint)
assert restarted.get_knoxel_by_id(receipt.added_ids[0]).content == "Hello, Spamuel."
```

`replace(record)` stages a replacement of a visible record.
For an isolated edit, call `record.copy_for_edit()` before mutation.
This method copies that record with pickle and retains its central owner.
Ordinary getters never copy record values. Direct mutations are immediately visible through shared references.
It retains the record's ID, class, scope, and creation tick.
The service validates typed memory references before acceptance.
It does not reinterpret IDs inside free-form metadata.

The old `add_knoxel()` and `replace_knoxel()` mutation helpers now raise errors.
Use `store_knoxel()` for one trusted host input.
Use a batch for related records, replacements, GWT state, and managed-run receipts.

## Persistence and restart

The CLI uses `<runtime-root>/ghost.pickle` by default.
`--checkpoint PATH` selects another checkpoint.
`--ephemeral` disables accepted-state persistence.
The `build_app()` function remains in-memory unless its caller supplies a checkpoint or a persistent Ghost.

An existing checkpoint takes precedence over the seed.
A missing checkpoint starts from the seed and saves on the first accepted change.
Loading a corrupt checkpoint raises an error. It never silently starts a new memory.
A bare store constructor makes no model requests.
Real application startup calls `initialize_embeddings()` before workflow execution.
It embeds every missing record, builds FAISS, and saves the completed vectors once.
New record commits also embed missing vectors before the store accepts those records.

The default format is a pickle of the typed [`pm.memory.checkpoint.Checkpoint`][].
[`pm.memory.checkpoint.PickleCheckpointSerializer`][] preserves subclasses, all record fields, and every embedding.
The record pickle excludes its runtime owner and lock. Loading restores the same central owner on every record.
Load pickle files only from trusted character homes.
[`pm.memory.store.MemoryStore.load`][] accepts pickle, JSON, and legacy YAML seeds.
[`pm.memory.checkpoint.JsonCheckpointSerializer`][] remains available for explicit JSON export and legacy imports.
The application prefers `ghost.pickle` and accepts an older `ghost.json` when no pickle exists.
Future saves use pickle. YAML remains an import format.
Read [inference settings](subsystems/inference-settings.md#checkpoint-serialization) for the persistence boundary.
Mental-state arrays remain unchanged.
The source YAML remains a separate, unchanged seed.
Fresh YAML imports retain source hashes and explicit background provenance.
Read [experienced memory](subsystems/experienced-memory.md) for attribution, retrieval, and interval queries.

Restart restores accepted memory, not threads or external effects.
Previously active run receipts become `interrupted` in the restored Ghost.
The next commit persists this classification.
The managed-run supervisor uses these receipts for real workflow execution.

## Observed performance

The comparison uses the same 11,202-record Spamuel seed and installed interpreter.
The baseline is repository revision `5fda8044cc412045985f3a9846b82b2f55c50ce0`.
Both comparison checkpoints contain the seed's empty embedding fields.

| Operation | Baseline | Shared records and pickle |
| --- | --- | --- |
| Snapshot | 515 ms | 2.35 ms |
| 1,000 record getters | 260 ms | 0.48 ms |
| Full-state commit | 2.05 s | 0.54 s |
| Checkpoint reload | 2.60 s | 0.42 s |

The local `embeddinggemma-300m` pass embedded all 11,202 records in about 161 seconds.
The resulting pickle holds every 768-dimensional vector and occupies about 113 MB.
Loading this complete file took about 1.86 seconds. Initialization took another 21 ms and made zero embedding requests.
A fresh semantic query took about 179 ms, including one local embedding request.
These measurements establish storage and retrieval mechanics. They do not establish companion reply quality.
The machine receipts remain under `.runtime/performance-20261021/`.
The earlier OpenRouter HTTP 401 receipt remains there as failed development evidence.

## Bus behavior

[`pm.bus.bus.BusSystem`][] serializes each envelope before delivery.
Each subscriber receives a separate deserialized envelope, including its nested payload.
Predicates and delivery callbacks run outside the subscription lock.
The callbacks select or enqueue data. They do not run a workflow.

Envelopes retain message, correlation, causation, source, and target IDs.
Targeted data reaches only the named module through the default interest function.
Untargeted data reaches all interested modules.
Unsupported live objects and callback errors raise errors.

## Limits

This is one Ghost owner in one process, not a multi-process database.
Only that owner can write its checkpoint.
Each commit serializes the full state. No incremental journal or database migration is necessary for this version.
Atomic replacement protects the previous file during a failed save. It does not promise recovery from every hardware or power failure.

The bus has no durable replay or exactly-once delivery.
A crash after the checkpoint save can leave an accepted result without a delivered event.
Repeated publication can repeat an event, but it must reference the accepted receipt.

The batch API is trusted host code, not a role permission system.
Inherited indexes and host attributes are internal mutable state, not a Python sandbox.
Workflow tools receive scoped payloads, write grants, and child-run boundaries.
The in-process read view retains typed references to the central records.

The terminal uses real cognition. The direct fixture tests also cover dummy modules.
These tests establish state and delivery behavior, not dialogue quality.

## Tests

For direct memory and application tests, run:

```powershell
uv run --locked python -m unittest tests.unit.pm.bus.test_bus_isolation tests.unit.pm.memory.test_store_commits tests.unit.pm.host.test_durable_application -v -b
```

These tests start no module threads. The full-seed test uses the real Spamuel seed.
The direct application tests run the real Python workflow kernel and virtual Bash environment.

For actual concurrent commits, run:

```powershell
uv run --locked python -m unittest tests.unit.pm.memory.test_store_concurrency -v -b
```

The concurrent tests use two real threads, barriers, and actual checkpoint files.
The failed-save test uses a real invalid filesystem target, not a mocked writer.

## Related guides

Read the [system map](concepts/architecture.md) and [development loop](workflows/development.md).
