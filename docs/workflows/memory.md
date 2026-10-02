# Experiment with memory

## Summary

The knoxel editor shows serialized accepted records.
Edits use the same Ghost batch validation as runtime writes.
Revision checks reject stale replacements.
A copied record can create a new memory with a new ID.

Parent: [console procedure](console.md).
Contract: [Ghost commits](../ghost.md).

## Procedure

1. Deploy a separate instance for the experiment.
2. Start the instance.
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

[`pm_core.ghost.batch.GhostWriteBatch.replace`][] preserves the class, scope, and creation tick.
[`pm_core.ghost.batch.GhostWriteBatch.create`][] reserves a fresh ID.
[`pm_core.ghost.service.Ghost.commit`][] validates references and writes the checkpoint before acceptance.
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

Read [next experiments](../decisions/experiments.md) for broader evaluation criteria.

