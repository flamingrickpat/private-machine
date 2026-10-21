# Live prompt logs

## Summary

PM saves each chat or embedding request before it contacts the endpoint.
The default directory is the model folder under `~/.pm/private-machine/cache`.
Embedding logs use its `embeddings` subdirectory, followed by the model folder.
Each call has a readable Markdown file and a temporary exact JSON payload.
The response adds token usage, cost, timing, and any error.

These logs let the operator inspect current work and give feedback.
They do not change the request or the companion's accepted state.

Parent: [Python chat and agent context](llm-context.md).
Call boundary: [`pm.llm.http_client.JsonClient`][].
Entry points: [`start_prompt_log`][pm.llm.prompt_log.start_prompt_log] and [`finish_prompt_log`][pm.llm.prompt_log.finish_prompt_log].
Related: [operator console](console.md), [Story speech](story-speech.md), and [model connections](../models.md).

## Files and timing

Each actual HTTP attempt receives a UUID call ID.
The logger writes the request before dispatch, including on a retry.
The owning thread identifies the runtime module or stage.

| File | Content | Written when |
| --- | --- | --- |
| `<call-id>.md` | Readable metadata, messages, tool calls, response, and reasoning text when supplied | Before dispatch, then rewritten after the response |
| `<call-id>.json` | Exact request payload, endpoint, model, route, thread, and request count | Before dispatch. Removed after a response or error |
| `usage-<pid>.jsonl` | One start event and one outcome event per call | At each boundary |

Markdown wraps long lines at 120 characters.
The JSON payload preserves whitespace, image content, tool schemas, and all generation fields exactly.
The JSON file is available while the call is pending.
Completed calls retain one Markdown file and their usage events, without separate response files or mirrored cache transcripts.
Embedding receipts keep usage and omit large response vectors.

The provider's `usage` object stays intact.
It can include prompt tokens, completion tokens, reasoning tokens, cached tokens, and cost.
A missing value stays unknown.
Tokens per second equal completion tokens divided by elapsed request time, including network time.
Counts arrive after the response. PM does not stream token counts during generation.

The console's worker log shows request starts, completion status, and elapsed time.
Usage events retain call IDs and provider token counts beside the Markdown files.
Its existing `model.start` and `model.end` events retain workflow and exchange attribution for direct chat calls.

## Native agent logs

Managed agents use pm-coder's existing logger.
PM does not duplicate those requests through its direct-chat logger.
pm-coder saves the exact outbound request after message and tool conversion, before dispatch.
Its session files include ordinary agent calls and compaction calls.

All native workflow roles, note-task sessions, and Minecraft specialist sessions use the `agents` subdirectory of the operator's prompt directory.
Note-task and specialist reports retain `agent_log_root` and `agent_session` so the operator can locate the exact session.
Native workflow roles retain session IDs and usage in their kernel traces under the run artifacts.
Read `turn_*_agent_*.json` and `turn_*_summarizer_*.json` for the actual requests.
The driver's trace and pm-coder's `runs.jsonl` retain reported session usage.
Native agent usage is available at the completed session boundary.

Request files use `agents/<agent_session>` below the prompt directory.
The session ID comes from the task report or the kernel's agent result.
The run receipt's artifact path locates the kernel traces and task outputs.
The [managed-work guide](managed-work.md) explains the run receipt.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PM_PROMPT_LOG_DIR` | Model folder under `~/.pm/private-machine/cache` | Directory for the prompt files. |
| `PM_PROMPT_LOG` | on | Disable PM's direct logger with `0`, `false`, `off`, or `no`. |

`PM_PROMPT_LOG_DIR` also selects the root for native agent session logs.
`PM_PROMPT_LOG=0` does not disable pm-coder's native session logger.
For development on this machine, leave `PM_PROMPT_LOG_DIR` unset so the operator can inspect the default directory.

## Failure behavior

Each process owns its usage file and serializes its threads' appends.
A write error prints a warning. It cannot replace a model result or mask cancellation.
Cancellation retains the prompt and records `CancelledError`.
A process crash can leave a pending prompt without an outcome.
That pending record does not establish whether the provider completed or charged for the request.

HTTP success means that a response arrived.
Workflow parsing and task verification remain separate boundaries.
Read the workflow receipt before you treat a model response as accepted work.

Headers and API credentials do not enter the logs.
Prompt content can contain private dialogue or tool results supplied by the caller.
The files stay in the operator's private directory or an ignored runtime directory.

## Inspect a prompt

1. Sort the Markdown files by modification time.
2. Open the newest file to inspect its prompt and owning thread.
3. Read the outcome to inspect usage and the reply.
4. For a pending call, open the adjacent JSON file to inspect the exact request.

To observe one process's usage events in PowerShell:

```powershell
Get-Content -LiteralPath C:/Users/rick/.pm/private-machine/cache/<model>/usage-<pid>.jsonl -Wait
```

Replace `<model>` with the model folder and `<pid>` with the worker PID from the console.

The unit check `tests/unit/pm/llm/test_prompt_log.py` verifies pending requests, exact payloads, usage, cancellation, embedding receipts, wrapping, and write failures.
It writes into a temporary directory, so it never touches the operator's log.

## Related pages

- [Python chat and agent context](llm-context.md)
- [Story speech and the agentic review](story-speech.md)
