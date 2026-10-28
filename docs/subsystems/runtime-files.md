# Runtime files and recovery

## Summary

The shared cache holds model responses. Each owning home holds its inference transcripts.
Each agent home holds raw memory history, unlimited backups, process logs, and runtime artifacts.
Normal shutdown drains running work and saves parked workflows. A forced interruption remains separate.
Windows also removes child processes when their owning runtime exits.

Parent: [direct workflow execution](../runtime.md).
Related: [model connections](../models.md), [character home](character-import.md#home-folder).

## Locations

| Path | Content |
| --- | --- |
| `~/.pm/private-machine/cache/<model>/` | Completion JSON, readable transcripts, and inference receipts. |
| `<home>/ghost.pickle` | Full raw history. Normal loading opens its accepted causal view. |
| `<home>/ghost_<tick>_<UTC datetime>_<suffix>.pickle` | Stable memory after a successful main cognition commit. |
| `<home>/logs/prompts/` | Exact direct requests, responses, usage receipts, and native agent sessions. |
| `<home>/logs/runtime_<UTC datetime>_<pid>.log` | Process diagnostics and native crash traces. |
| `<home>/runtime/` | Kernel journals, managed tasks, and workflow artifacts. |

`--home` selects the agent home. No additional folder configuration is necessary.
Without `--home`, the terminal uses its selected runtime root for diagnostics and checkpoints.
Console workers use their instance root as the home.
`PM_CACHE_DIR` changes the single shared cache root for tests or another deployment.
A configured persona log directory takes precedence over `PM_PROMPT_LOG_DIR`.
Standalone tools can use that environment variable when they have no owning home.
`PM_PROMPT_LOG=0` disables inference transcripts. It does not disable process diagnostics.

Embeddings live on the knoxels inside `ghost.pickle`. There is no embedding cache.
The application accepts legacy JSON checkpoints and YAML seeds. It saves subsequent state with pickle.
A launcher without a home can add missing Markdown transcripts to existing completion response entries.

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
Main cognition writes the backup after its cycle commit and before it publishes bus events.
Its tick identifies that completed cycle.
Failed attempts, retries, and cancellations before commit create no backup.
Default recovery excludes a later message when no completed cognition run admitted its causal dependencies.
The raw backup can retain that provisional message for debugging.
A backup I/O error propagates after memory acceptance and prevents event publication. Earlier backups remain available.
An explicitly ephemeral store writes no backup.

1. Stop the agent and wait for its workers to exit.
2. Preserve the current `ghost.pickle` under another filename.
3. Copy the selected backup over `ghost.pickle`.
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

The first stop signal fences intake and timers, then drains runnable work.
Parked workflows retain their journals and workspace bytes in the ghost.
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

## Causal recovery and exports

K002 adds immutable transaction history to the normal ghost file.
Each logical knoxel change creates a new ID and records its predecessor.
Delivery and execution updates use separate operational receipts.
Main cognition admits new stimuli; completed approved actions retain their exact input dependencies.
See [causal memory and recovery](../work_items/k002/causal-memory.md).

Normal startup opens the accepted view and resumes unfinished approved actions.
An explicit raw load supports debugging.
A clean export writes a new file and preserves the source history:

```powershell
uv run --locked python main.py --home YOUR-HOME --clean .runtime/clean-ghost.pickle
```

The destination must not exist.
A bare legacy ghost cannot restore missing earlier payloads; its existing records become an explicit import baseline.

The [simulation](simulation.md) also retains the clock, timer deadlines, and participant routing ledger.
Failed multi-participant runs resume from a coordinated barrier for all participants.
Single-participant recovery retains its causally accepted progress.
