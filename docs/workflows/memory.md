# Experiment with memory

## Summary

The knoxel editor shows serialized accepted records.
Edits use the same Ghost batch validation as runtime writes.
Revision checks reject stale replacements.
A copied record can create a new memory with a new ID.

Parent: [console procedure](console.md).
Contract: [Ghost commits](../ghost.md).
Live cognition: [experienced memory](../subsystems/experienced-memory.md).

## Procedure

1. Deploy a separate instance for the experiment.
2. Start the instance.
   Startup reads its accepted knoxels through Ghost and fills the memory list.
3. Select **Read memory**.
4. Open **Knoxels**.
5. Search for a record by ID, type, or content.
6. Select the record.
7. Edit its JSON content.
8. Select **Commit memory**.

For a new record, select **New from selected record** before the commit.
The worker assigns the new ID and creation tick.
The serialized `__type__` selects the knoxel class.
The `data` object contains that class's fields.

## State changes

[`pm.memory.write_batch.WriteBatch.replace`][] preserves the class, scope, and creation tick.
[`pm.memory.write_batch.WriteBatch.create`][] reserves a fresh ID.
[`pm.memory.store.MemoryStore.commit`][] validates references and writes the checkpoint before acceptance.
The console then refreshes its snapshot.

## Traps

The editor exposes complete records, including reference fields.
An invalid class, field value, or reference rejects the batch.
A stale revision rejects a replacement.
Reload the record before another edit.

New records can retain references from their template.
Review those fields before acceptance.
The editor does not delete records or repair references automatically.

## Compare results

1. Send the same test input before and after the change.
2. Inspect the context and attention reports.
3. Record which memory IDs enter the accepted frame.
4. Compare the reply and timing evidence.

Read [next experiments](../work_items/experiments.md) for broader evaluation criteria.

## Vector search

1. Select **Vector similarity**.
2. Enter a query in the search field.
3. Select **Search vectors**.

The result includes cosine similarity and index coverage.
Search uses the configured model, stored record vectors, and the in-memory FAISS index.
Missing vectors do not silently become text matches.

Select **Embed missing records and search** to prepare the accepted-text index for the selected type and subtype.
This operation sends external embedding requests for missing records.
The progress counter shows the indexed and total record counts.
Select **Cancel search** to interrupt preparation without stopping Ghost.
Completed vectors remain on their records and are saved in the complete pickle checkpoint.

[`pm.memory.store.MemoryStore.search_memory`][] owns scoped search.
[`pm.memory.store.MemoryStore.dialogue_history`][] owns accepted public dialogue inspection.
Other trusted callers can reuse these APIs with their permitted record IDs.

The knoxel editor hides embedding and state-vector fields. Existing vectors remain intact when you edit other fields.
Use the editable **Knoxel type** and **Subtype** lists before a text or vector search.
Ghost applies these filters before vector ranking and optional indexing.
