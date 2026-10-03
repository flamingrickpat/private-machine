# Operator service

## Summary

The console uses a loopback HTTP server and one subprocess per instance.
Workers record bus deliveries, module activity, workflow reports, model calls, and accepted memory snapshots.
The browser polls this evidence every second.

Parent: [system map](../concepts/architecture.md).
Operation: [console procedure](../workflows/console.md).

## Input, change, and output

The browser sends JSON commands to [`console.server.serve`][].
[`console.server.Instance`][] forwards worker commands through standard input.
The worker acknowledges or rejects each command.
The service appends worker events to `events.jsonl`.

The browser shows a graph from observed source-to-recipient deliveries.
Node details show module subscriptions and recorded workflow evidence.
Workflow details include the manifest, request, report, status, duration, and artifact path.
Model details include prompts, generation arguments, responses, and durations.
The internal workflow graph shows observed parent-child links and active child runs.
Each workflow start includes its manifest phases, roles, tools, and result contracts.

## Files

Each deployment creates `.runtime/console/<instance-id>/`.

| File | Contents |
| --- | --- |
| `instance.json` | Name, seed path, model configuration path, cognition mode |
| `ghost.json` | Accepted checkpoint after the first commit |
| `events.jsonl` | Persistent runtime trace |
| `worker.log` | Standard error and dependency output |
| Workflow folders | Kernel artifacts and journals |

The service retains the latest 10,000 events in memory.
The full trace remains on disk.
The memory list shows the first 250 search matches.
The HTTP service sends a snapshot only after that snapshot changes.
Chat session labels stay in browser storage.
Chat messages stay in the trace and accepted Ghost records.

## State and side effects

Start launches a worker with a hidden window on Windows.
Stop requests cooperative module and managed-run cancellation.
Kill terminates the worker tree on Windows.
Startup restores an existing checkpoint before it uses the seed.

The console installs observation hooks only inside its worker process.
Memory inspection, public dialogue history, and vector search use the running Ghost.
The HTTP service does not read checkpoints to reproduce those operations.
The bus observer receives detached envelopes before mailbox delivery.
Observation errors propagate instead of silently losing evidence.
Normal terminal runs have no observer by default.

## Traps

Stop can wait for a synchronous model or external call.
Kill cannot undo external effects.
Notifications require browser permission.
The graph shows observed connections after traffic occurs.
It does not show an unexecuted static topology.

The console has no remote authentication or remote deployment mechanism.
The HTTP service binds to `127.0.0.1` and rejects mutations from other browser origins.
Trace files contain user messages and model prompts.
Trace serialization and storage add some overhead to measured execution times.
The UI does not expose model credentials from configuration files.

## Go deeper

Read [memory experiments](../workflows/memory.md), [evidence](../concepts/evidence.md), and [`console.worker.run`][].

## Endpoint configuration

[`runtime.operator_services.operator_config`][] reads the project model configuration.
It reads embedding overrides from `PM_ENV_FILE`, the repository `.env`, or the legacy `../pm_next/.env`.
An existing credential environment variable takes precedence over a legacy file credential.
The service never includes credentials in connection reports.

`MINECRAFT_MCP_URL` selects the MCP endpoint.
Otherwise, `PM_MC_MCP_PORT` selects its port.
`PM_MINECRAFT_HOST` and `PM_MINECRAFT_PORT` select the TCP probe target.
