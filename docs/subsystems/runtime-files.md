# Runtime files and recovery

## Summary

The shared cache holds model responses and inference transcripts.
Each agent home holds its accepted memory, unlimited backups, process logs, and runtime artifacts.
Shutdown cancels active requests and joins owned workers.
Windows also removes child processes when their owning runtime exits.

Parent: [direct workflow execution](../runtime.md).
Related: [model connections](../models.md), [character home](character-import.md#home-folder).

## Locations

| Path | Content |
| --- | --- |
| `~/.pm/private-machine/cache/<model>/` | Completion JSON, readable transcripts, and inference receipts. |
| `~/.pm/private-machine/cache/embeddings/<model>/` | Embedding JSON, readable input, and inference receipts. |
| `<home>/ghost.json` | Current accepted memory. Each accepted commit saves it. |
| `<home>/ghost_<tick>_<UTC datetime>_<suffix>.json` | Stable memory before a main cognition attempt. |
| `<home>/logs/runtime_<UTC datetime>_<pid>.log` | Process diagnostics and native crash traces. |
| `<home>/runtime/` | Kernel journals, managed tasks, and workflow artifacts. |

`--home` selects the agent home. No additional folder configuration is necessary.
Without `--home`, the terminal uses its selected runtime root for diagnostics and checkpoints.
Console workers use their instance root as the home.
`PM_CACHE_DIR` changes the single shared cache root for tests or another deployment.
Legacy `PM_PROMPT_LOG_DIR` still redirects attempt logs. Cache transcripts remain beside their JSON files.
`PM_PROMPT_LOG=0` disables inference transcripts. It does not disable process diagnostics.

Old `embedding.cache_dir` values remain read sources. New writes always use the shared cache.
Valid entries move into the shared cache when retrieval reads them. The old files stay unchanged.
The binary cache at `~/.pm/embeddings` remains read-only.
Startup adds missing Markdown files to existing JSON entries under the shared cache.

## Read an inference

[`pm.llm.http_client.JsonClient`][] writes the request before external I/O.
Its transcript shows each system, user, assistant, and tool turn under a Markdown heading.
Existing line breaks remain. Added line breaks limit each line to 120 characters.
Complete JSON content uses two-space indentation. Long JSON strings wrap for display only.
The JSON cache retains its original bytes.

The completed transcript includes provider token counts, finish reason, elapsed seconds, speed, and cache-hit status.
Provider usage details and timing fields stay in the receipt.
Speed means output tokens divided by total request seconds, including queue and prompt processing inside the provider.
This is an average rate. It does not measure decoding speed alone.
The current transport uses non-streaming responses, so client TTFT remains unknown.
Missing provider statistics remain unknown. Cache replays do not represent fresh inference performance.

A failed or cancelled request retains its attempt transcript and error receipt.
A later retry uses a fresh attempt ID. Its cache transcript shows the latest attempt.
Old cache files can lack the original request or timing data. Backfilled transcripts mark those fields as unknown.

## Restore accepted memory

[`pm.memory.store.MemoryStore.backup`][] holds the commit lock while it captures and writes a complete checkpoint.
The backup includes records, revisions, GWT turns, cycle state, managed runs, and commit receipts.
Its tick identifies the last accepted tick before the attempt.
Retries retain separate files, including retries at the same tick.
An I/O error stops the cognition attempt before it changes memory.
An explicitly ephemeral store writes no backup.

1. Stop the agent and wait for its workers to exit.
2. Preserve the current `ghost.json` under another filename.
3. Copy the selected backup over `ghost.json`.
4. Start the agent with the same home.

A memory rollback does not reverse external actions or task artifacts.
Managed-run recovery still checks its durable control files and external outcomes.
Backups have no automatic deletion policy and can consume substantial disk space.

## Trace a stop or crash

[`pm.host.diagnostics.configure_logging`][] opens one process log in the home.
Lines show UTC time, process ID, thread ID, thread name, logger name, and severity.
Module, workflow, model, memory, and managed-run boundaries identify the last recorded operation.
Thread failures include their traceback and active invocation.
Native failures write all Python thread stacks to the same log when Python can handle the fault.

[`pm.host.diagnostics.join_thread`][] records each join.
If a worker remains active for five seconds, it records that worker's stack and continues to wait.
This preserves the location of an uncancelled external operation without abandoning its cleanup.

Ctrl+C, SIGTERM, and Windows Ctrl+Break signal cancellation before cleanup starts.
The terminal restores its previous signal handlers after cleanup.
If cleanup is stuck, a second stop signal records all thread stacks and forces exit with code 130.
Web sockets close, modules stop, and managed workers finish before the terminal returns.
The advertised web URL uses loopback when the server binds to all IPv4 interfaces.
Console memory searches also receive cancellation and join.
Partially started applications use the same cleanup path.

[`pm.host.processes.ensure_process_cleanup`][] places the Windows runtime and its children in an owned Job Object.
Windows kills remaining descendants after runtime exit, including a forced parent exit.
Existing external services stay outside that job.
See the [Windows Job Object contract](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).
If a host rejects the job, the process log records the Windows error. Normal cancellation and joins still apply.
