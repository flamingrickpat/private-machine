# Classification recovery and context

## Summary

INVALID retains a concrete cause and routes to one scoped pm-coder attempt.
The handler receives the same question and classes, plus eligible snapshots and failure evidence.
Direct calls reject oversized context without clipping.
Agent compaction retains the original request and summarizes accumulated execution.

Parent: [classification](classification.md).
Evidence: [A04 verification](../../work_items/verification/a04-classification.md).
Policy: [model selection](../../work_items/alpha-plan/model-policy.md).

## Limits and direct calls

The primary node can declare `classification_limits`.
[`pm_workflows.protocol.ClassificationConfig`][] owns these values.

| Field | Default | Unit and effect |
| --- | --- | --- |
| `max_input_tokens` | 16,384 | Conservative prompt token bound |
| `max_output_tokens` | 512 | Reserved completion tokens |
| `max_capture_bytes` | 1,048,576 | Captured source and value limit |
| `max_candidates` | 256 | Direct-child listing limit |
| `fallback_max_bytes` | 16,777,216 | Total eligible recovery snapshot bytes |
| `timeout_seconds` | 120 | HTTP I/O timeout; recovery session deadline |

Positive finite limits are mandatory. Integer limits reject booleans and fractions.
The engine counts UTF-8 content bytes plus 256 framing tokens as a conservative bound for byte-based chat tokenizers.
It records this method as `utf8_bytes_plus_256`, rather than claiming an exact tokenizer count.
The bound must fit both `max_input_tokens` and the resolved serving context after the output reserve.
Provider usage records actual accepted prompt tokens.
Other tokenizer families require an appropriate bound before adoption.

An oversized prompt produces INVALID before HTTP dispatch.
The engine never clips a listing, source, question, class contract, or answer.
It sends one OpenAI-compatible JSON completion request, with temperature zero and thinking disabled.
An empty local API key omits the authorization header.
Only `finish_reason: stop` permits answer validation.
The direct HTTP timeout applies to each network I/O operation.
The recovery deadline bounds the complete agent session.

PM's direct [`pm.llm.activation.ChatClient.complete`][] remains a single-request API.
It does not inherit this caller's context construction or pm-coder's history loop.
The [chat context guide](../llm-context.md) explains that distinction.

## Scoped recovery

The handler binds the latest INVALID receipt in the same task, run, and item.
Its origin contains the producer phase, attempt, and receipt digest.
A changed question, class contract, input binding, or route configuration rejects recovery.
A repeated handler invocation for the same origin produces `fallback_exhausted`.
The static topology also rejects a recursive unresolved route.

Recovery applies the coding role's readable paths and deny rules during capture.
It checks listed and exact file versions again before snapshot creation.
A stale or denied source cannot become a class through recovery.
An earlier capture remains the evidence source for an oversized direct prompt.
A capture byte-limit failure can recover the same file version within the larger recovery limit.

[`pm_workflows.classification.ClassificationEnvironment`][] creates a separate BashMachine containing only eligible captured values.
Existing snapshot files have read-only ACLs.
All agent file and shell tools bind to that machine and its `classification` user.
The agent receives no application tools, MCP services, original environment, complete Ghost projection, or host-file access.
It can create virtual scratch files; those files cannot change the original source or working target.
Structured values appear as canonical JSON snapshots.

The handler receives the selection instruction, selected answer, invalid reason, question, classes, value digests, and metadata.
It copies descriptor identifiers exactly and quotes actual source text.
Its role skill supplies the use-specific inspection procedure.
Missing data still permits an agent to explain INVALID; missing files do not permit invented evidence.
The engine validates the final object once with the same answer validator.
An invalid agent answer routes to the declared unresolved destination.

## Compaction and cancellation

Recovery requires `pm-coder` and an explicit `runtime.agent_model`.
The adapter preserves the installed coder's automatic context recovery.
The coder summarizes generated execution, keeps the original request exact, stores a checkpoint, and resumes.
An oversized initial instruction with no generated execution cannot use this mechanism.
The engine therefore bounds the recovery instruction before dispatch.

pm-coder's `live_test=True` diagnostic mode raises provider errors before its recovery loop.
The forced-overflow acceptance case uses `live_test=False` and actual model inference for summaries.
It preserves the server's context rejection, summary requests, checkpoint, instruction, and final quotes.
No pm-coder implementation or serving-context change occurs.

Cancellation closes direct HTTP work or cancels the bound agent task and awaits its cleanup.
The workflow driver now enforces its bound-session deadline.
Cancelled classification publishes no accepted label or named output.
A deadline or provider failure produces `agent_failure` with retained diagnostics.
The agent's internal provider retries remain inside that one bounded session.

## State, retries, and failures

Classification reads evidence and appends journal entries.
It neither settles a worktree nor restores external effects.
Resume treats classification as validation and retains the candidate files.
Ordinary role retry, checkpoint, and exhaustion policies retain their existing behavior.
External workflows therefore keep actual effects across classification boundaries.

| Cause | Recovery requirement |
| --- | --- |
| `missing_file`, `empty_listing`, `missing_field` | Obtain data or explain the unresolved condition |
| `stale_input`, `stale_output`, `inaccessible_path` | Obtain new granted evidence before a new classification |
| `oversized_input` | Inspect within the recovery snapshot and agent context limits |
| `uncertain_selection`, `uncertain_classification` | Establish one eligible source and one supported class |
| `unknown_candidate`, `unknown_label`, `invalid_response`, `unsupported_evidence` | Return the exact shared contract from actual evidence |
| `endpoint_unavailable`, `missing_connection`, `incomplete_response` | Diagnose the selected endpoint; use the separately configured handler |
| `missing_agent`, `agent_failure`, `fallback_exhausted` | Inspect retained diagnostics and take the unresolved route |

Unavailable timestamps, absent producer metadata, digest-only replacement limits, and nontransactional reads remain explicit A03 limitations.
