# Runtime record packages

## Summary

PM can export a named collection of serialized records from one frozen snapshot.
The review desk formats that collection for reading and keeps human ratings separate.
Records retain their timestamps, provenance, appraisal fields, codelet metadata, and other serialized fields.
Embedding fields are omitted from the export and hidden when the desk imports external packages.

Parent: [conversation review](async-conversation-review.md).
Related: [private loop](private-loop.md) and [reflection](reflection.md).

## Capture a collection

The runtime API is [`pm.introspection.record_packages.capture_records`][].
It accepts a `Snapshot`, ordered record IDs, a title, and a display hint.
Use `related_ids` for supporting sources that should accompany the collection without becoming its primary members.
The function removes duplicate IDs and preserves caller order.
Missing records raise `KeyError`. Unsupported display hints raise `ValueError`.
It performs no inference, memory commit, or file write.

```python
from pathlib import Path
from pm.introspection.record_packages import capture_records, capture_gwt, dump_package

snapshot = app.ghost.snapshot()
chat = capture_records(
    snapshot,
    delivered_message_ids,
    title="Chat log from Jesse's perspective",
    view="chat",
    perspective="Jesse",
)
dump_package(chat, Path(".runtime/debug/chat.json"))
dump_package(capture_gwt(snapshot, tick_id), Path(".runtime/debug/gwt.json"))
```

The caller selects records visible to the intended reader.
The perspective field labels that selection. It does not establish that a message reached the reader.
For public chat, use actual delivery records and their observed order.
The `chat` display hint rejects records whose type is not dialogue.
It never copies private thoughts into a public view automatically.

[`pm.introspection.record_packages.capture_gwt`][] captures the selected turn and its exact ordered member IDs.
The package retains the complete turn record: the summary, timestamp, slot, claimed stimuli, shard changes, codelet trace, experience, and decision record.
Capture at the tick boundary to preserve that tick's current record revisions.
Exporting an older turn later uses the snapshot's current record payloads.
It does not reconstruct prior versions of mutable records.

[`pm.introspection.record_packages.capture_cycle`][] captures one accepted cycle state with the `cycle` display hint.
Pass the snapshot and the cycle state of the same tick, for example `app.ghost.states[-1]`.
The package field `cycle` holds the state: the CSM items with activation and endorsers, the workspace shards with items and tails, the codelet registry state, the stimulus groups, the decisions, and the dispatches.
`gwt` holds the turn record of the tick.
The members are every record that the state or the turn names ([`cycle_record_ids`][pm.introspection.record_packages.cycle_record_ids]): stimuli, shard items and tails, CSM items, structures, codelet percepts, actions, experiences, and the decision record.
Step 11 of A19F builds the review package from these captures.

[`pm.introspection.record_packages.dump_package`][] writes the detached package as UTF-8 JSON.
The caller owns the destination and exposes it for review after the write returns.
Filesystem errors propagate. A capture does not change accepted runtime state.

## Package contract

The versioned schema is `pm.record-package.v1`.

| Field | Meaning |
| --- | --- |
| `title`, `view`, `perspective` | Collection name, display hint, and intended reader |
| `captured_at` | UTC capture time |
| `record_ids` | Ordered primary members |
| `records` | Existing `__type__` and `data` serialization for members and related sources |
| `revisions` | Captured accepted revision of each included record |
| `gwt` | Complete selected GWT turn, or `null` |
| `cycle` | The captured cycle state. Only the `cycle` display hint has it. |

Display hints are `records`, `chat`, `gwt`, and `cycle`.
All serialized fields remain available except recursively named `embedding` and `embeddings` fields.
Appraisal and mental-state vectors remain available because they are not embeddings.
The package contains no model judgment or human rating.

## Read and rate packages

```powershell
uv run --locked python dev/review/server.py register-package `
  .runtime/debug/chat.json .runtime/debug/gwt.json `
  --id YOUR-REVIEW-ID --work-item A19
uv run --locked python dev/review/server.py serve
```

The importer creates a review item for each file.
This permits separate captures to retain different versions of the same record.
A combined item can also contain multiple packages from one consistent snapshot.
Conflicting copies in a combined item raise an error instead of choosing one silently.

The desk opens chat as speech in caller order.
The **Record packages** view selects named collections and displays GWT summaries beside their member content.
Hover over a record heading or **Record metadata** to see timestamps, source, revision, appraisal, and codelet fields.
Expand **Record metadata** to read all captured fields.
The desk formats text as text. Captured content cannot execute HTML or scripts.
Uncaptured linked records stay disabled and identify their missing capture.

The formatter is [`dev.review.record_format.present_packages`][].
It imports no runtime code, reads no checkpoint, and runs no inference.
The server reads the immutable bundle, formats its packages, and saves ratings in separate feedback files.
The A19 exporter now uses this package contract for the chat, private activity, and each retained GWT tick.
Its original delivery order and accepted record text remain unchanged.
