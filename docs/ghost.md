# Ghost commits and bus delivery

## Summary

Ghost accepts a complete batch of memory changes after its checkpoint save succeeds.
Each bus subscriber receives an independent JSON copy.
The same module `step()` supports direct debugging and threaded operation.

Parent: [project README](index.md).

## Accepted state

The [`pm.memory.store.MemoryStore`][] owns accepted records, indexes, GWT records, revisions, counters, and commit receipts.
The existing knoxel classes remain unchanged.
The service copies the seed instead of retaining its mutable objects.

A [`pm.memory.write_batch.WriteBatch`][] captures a stable read view and collects private proposals.
New records reserve integer IDs under the Ghost lock.
A discarded batch can leave ID gaps.
Other batches cannot read its proposals.

`snapshot(visible_ids)` returns detached records within the supplied set.
An empty set exposes no records. `None` selects all records for trusted host code.
A narrower view also excludes hidden revisions and GWT records that reference hidden memory.
The virtual Bash target uses this snapshot for each invocation.

## Commit order

The service performs these steps under one commit lock:

1. Reject cancellation, invalid references, or stale replacement revisions.
2. Construct and validate the complete candidate state.
3. Write and flush a temporary sibling of the checkpoint.
4. Replace the previous checkpoint with the temporary file.
5. Install the candidate as accepted memory.

The module publishes its bus output after the commit returns.
No model or network call runs under this lock.

An unrelated append does not cause a conflict.
Two replacements of the same revision produce one winner and one `GhostConflict`.
A failed save preserves accepted memory and the previous checkpoint.
The error reaches the direct caller or the application error handler.

A receipt-only commit can reuse the already validated record maps.
This path requires unchanged records, an unchanged tick, no new workspace or cycle, and empty record embeddings.
The commit prepares detached receipt tables before checkpoint I/O.
It installs those tables only after the same atomic checkpoint replacement succeeds.
It preserves record revisions and the canonical state bytes.

A repeated commit ID returns its original receipt without duplicate memories.
Callers use the receipt's IDs, not IDs from a repeated attempt.
Cancellation before checkpoint replacement rejects the batch.
Cancellation after replacement does not undo an accepted commit.

## Direct debugger example

```python
from pathlib import Path
from pm.memory.store import Ghost
from pm.records.cycle_state import GhostConfig
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType

ghost = Ghost(GhostConfig(user_name="Jesse", companion_name="Spamuel"),
              checkpoint=Path(".runtime/example-ghost.json"))
batch = ghost.begin_batch("example-turn", new_tick=True)
proposal = batch.create(Feature(feature_type=FeatureType.Dialogue, content="Hello, Spamuel."))
assert ghost.get_knoxel_by_id(proposal.id) is None
assert batch.view().get(proposal.id).content == "Hello, Spamuel."
receipt = ghost.commit(batch)  # breakpoint: validation, disk replacement, installation
restarted = Ghost.from_checkpoint(ghost.checkpoint)
assert restarted.get_knoxel_by_id(receipt.added_ids[0]).content == "Hello, Spamuel."
```

`replace(record)` stages a replacement of a visible record.
It retains the record's ID, class, scope, and creation tick.
The service validates typed memory references before acceptance.
It does not reinterpret IDs inside free-form metadata.

The old `add_knoxel()` and `replace_knoxel()` mutation helpers now raise errors.
Use `store_knoxel()` for one trusted host input.
Use a batch for related records, replacements, GWT state, and managed-run receipts.

## Persistence and restart

The CLI uses `<runtime-root>/ghost.json` by default.
`--checkpoint PATH` selects another checkpoint.
`--ephemeral` disables accepted-state persistence.
The `build_app()` function remains in-memory unless its caller supplies a checkpoint or a persistent Ghost.

An existing checkpoint takes precedence over the seed.
A missing checkpoint starts from the seed and saves on the first accepted change.
Loading a corrupt checkpoint raises an error. It never silently starts a new memory.
Construction starts no threads and makes no model requests.

The checkpoint format is JSON with schema `pm.ghost-snapshot.v1`.
[`pm.memory.checkpoint.CheckpointSerializer`][] separates encoding from atomic file replacement.
The default [`pm.memory.checkpoint.JsonCheckpointSerializer`][] retains that schema and writes one encoded buffer.
Read [inference settings and checkpoint serialization](subsystems/inference-settings.md#checkpoint-serialization) for the boundary and measured encoding cost.
It wraps the existing Ghost serialization with GWT records, counters, revisions, commit receipts, and managed-run receipts.
Semantic `embedding` fields remain empty. Their vectors belong in the shared embedding cache.
Mental-state arrays remain unchanged.
The source YAML remains a separate, unchanged seed.
Fresh YAML imports retain source hashes and explicit background provenance.
Read [experienced memory](subsystems/experienced-memory.md) for attribution, retrieval, and interval queries.

Restart restores accepted memory, not threads or external effects.
Previously active run receipts become `interrupted` in the restored Ghost.
The next commit persists this classification.
The managed-run supervisor uses these receipts for real workflow execution.

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
Workflow tools receive scoped copies, write grants, and child-run boundaries.

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
