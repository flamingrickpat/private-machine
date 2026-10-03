# Input versions, replay, and stale evidence

## Summary

Capture records exact values and checks the source version around each read.
Selection checks the listing again before file capture.
Named outputs bind exact producer attempts and item scope.
Receipts support replay while keeping unknown producer metadata explicit.
Stale data rejects the attempt without an automatic retry.

Parent: [scoped workflow inputs](index.md).
Related: [bindings](bindings.md) and [verification](verification.md).

## Candidate and listing evidence

[`pm_workflows.inputs.FileMetadata`][] describes one relative file.
[`pm_workflows.inputs.FolderListing`][] describes a sorted direct-child candidate set.

| Field | Meaning |
| --- | --- |
| `candidate_id` | SHA-256 of the relative path, stable across versions inside one environment |
| `path` | Relative POSIX path inside the working environment |
| `size_bytes` | Exact byte count for host files, UTF-8 text byte count for text-only environments |
| `revision` | Opaque environment version |
| `content_sha256` | SHA-256 of file bytes or exact UTF-8 text bytes |
| `modified_time` | Actual environment time, or null |
| `producer` | Available environment producer metadata, or null |

The listing revision hashes the ordered candidate metadata.
The listing also retains `data_selection` and its digest.
Different environments can have the same path ID. Scope and version evidence distinguish their captures.
The model receives known IDs and cannot invent a file outside the candidate set.

[`pm_workflows.working_target.WorkingEnvironment.file_metadata`][] provides a compatible digest fallback.
It reads text and leaves times and producer metadata unknown.
Older duck-typed environments use the same fallback.
[`pm_workflows.working_target.HostInputEnvironment`][] adds stat identity, nanosecond modification/change values, and exact-byte digests.
PM's folder adapter delegates to that implementation.
Ghost metadata remains independent of the host clock.
Captured producer mappings use independent copies. Later environment changes cannot rewrite an earlier listing.

## Capture ordering

1. Normalize the relative path and apply the declared folder boundary.
2. Apply the role's readable and denied paths.
3. Read metadata and compare any expected listed version.
4. Reject an oversized file.
5. Read exact text through the environment.
6. Read metadata again and compare both versions with the captured digest.
7. Apply structured selection and the declared type assertion.
8. Retain separate consumer and evidence copies.

Host boundary and grant checks include resolved targets.
A symlink or Windows junction cannot escape the environment or the declared folder through an exact file binding.
Virtual reads retain the BashMachine user's existing permissions.
The kernel does not convert a virtual path into a host path.

[`pm_workflows.inputs.InputResolver.capture_selected`][] accepts only an ID from its captured listing.
It creates a fresh listing and compares its complete version before the selected read.
A changed candidate set, changed selected file, or change during capture produces `stale_input`.
The caller must start a fresh selection attempt instead of retrying the old ID blindly.

## Producer and replay evidence

`pm.input-capture.v1` retains the source mode, task/run/item scope, binding, projection pointer, values, and digests.
File receipts also retain original text and metadata.
Selected-file receipts add the listing and selected ID.
Dynamic output paths retain the exact named-output dependencies.
Request receipts retain the selected field and original request digest.

Producer metadata comes only from the environment.
Known task, run, or item fields must match the current scope.
Missing producer metadata remains `producer metadata unavailable`.
Fresh file evidence establishes captured content, rather than an unrecorded producer attempt.
Unchanged content captured for a prior loop or foreach item rejects current-item reuse.

The journal is the durable source for accepted projections and input evidence.
Its active recovery cursor excludes abandoned records.
Replay consumes captured values instead of another file read.
The binding, source, and value digests expose changes to a recorded request.
The producer revision describes its candidate, rather than a later checkpoint commit.

## Limits and failures

Default capture limits are 256 candidates and 1,048,576 bytes per value.
Native metadata and digest fallbacks can read content before enforcing these limits.
These values therefore bound captured data, rather than filesystem I/O or model tokens.
Model consumers must enforce their serving context separately.
Capture never silently clips text.

| Cause code | Diagnostic and recovery |
| --- | --- |
| `missing_file`, `empty_listing` | Inspect the environment and producer path. Obtain current evidence before another attempt |
| `inaccessible_path` | Inspect runtime grants, path spelling, and declared folder bounds |
| `stale_input` | Inspect both source versions. Obtain a fresh listing and capture |
| `missing_output`, `stale_output` | Inspect the latest producer attempt and its task/run/item identity |
| `missing_field`, `incorrect_type` | Inspect the exact source value, pointer, and expected type |
| `malformed_value` | Inspect JSON compatibility, text encoding, metadata fields, and unresolved variables |
| `unknown_candidate` | Use only an ID from the captured listing |
| `oversized_input` | Use an explicit narrower source or an evidence-preserving processing stage |

Invalid role inputs retain partial earlier captures and take `on_invalid` before model dispatch.
A04 owns conversion of these codes into classification INVALID and its scoped agent fallback.
Cancellation propagates between operations. It writes no accepted output for the cancelled dispatch.
Existing external effects and trusted Python/shell behavior remain outside capture's rollback control.

Digest-only environments cannot detect replacement with identical bytes.
Multiple metadata reads do not form a filesystem transaction.
A later consumer must classify the captured value, rather than reopen the source and assume equality.
Absent producer metadata cannot establish the original task or attempt by itself.
These limits remain explicit in receipts and the [verification results](verification.md).
