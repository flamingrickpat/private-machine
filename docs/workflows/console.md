# Operate the console

## Summary

Deploy an instance definition, start its worker, and send a message.
Inspect the trace, workflow evidence, timings, and accepted memory.
Stop or kill the worker from the same console.

Parent: [manual](../index.md).
Design: [operator service](../subsystems/console.md).

## Start the service

1. Open PowerShell in the repository root.
2. Run the console command.

```powershell
uv run --locked python -m pm.console.server
```

3. Open `http://127.0.0.1:8090`.

Use `--port 8091` for another port.
Use `--root PATH` for another deployment directory.

## Deploy and chat

1. Enter an instance name.
2. Enter a persona YAML path, or keep the default seed.
3. Enter a model configuration path, or keep the default configuration.
4. Select real cognition or the labeled dummy fixture.
5. Select **Deploy**.
6. Select **Start**.
7. Enter the user name that the model must see.
8. Select **New session** for a separate transcript.
9. Send a message.

All sessions within an instance share accepted memory.
Jesse is the default user.
After startup, the console reads public Jesse–Spamuel dialogue from Ghost.
Select **Jesse / accepted history** to see those records.
Memory and dialogue controls require a running instance.
The left panel disables operations that the current state does not support.
The console permits one pending exchange per instance.
The worker rejects another exchange until the current reply arrives.

## Inspect an exchange

1. Select a chat entry to open its correlation filter.
2. Select an edge to show its delivered messages.
3. Select a node to show its activity and workflow events.
4. Select an event to show its complete recorded payload.
5. Open **Timing** to compare durations and error counts.
6. Select **Read memory** to refresh accepted records and managed-run receipts.

The timing table sorts work by total elapsed time.
It does not assign a usefulness score.
The [experiment guide](../work_items/experiments.md) explains how to compare usefulness.

## Notifications and lifecycle

1. Select **Enable notifications**.
2. Accept the browser permission prompt.

The browser shows a notification for replies outside the current visible session.
An unread counter also records those replies.

Select **Stop** for cooperative shutdown.
If a worker does not stop, select **Kill**.
The kill dialog explains the interruption before execution.
Accepted checkpoint data remains available on restart.

## Connections and source files

Open **Connections** and select **Check connections**.
The result shows the chat endpoint, embedding endpoint, Minecraft TCP port, and MCP tool discovery.
These checks do not start services or perform Minecraft actions.
TCP reachability does not establish a ready Minecraft world.

Select a network node or workflow event to show its source links.
Select **Open** to use the Windows Markdown association.
Python and YAML links use Sublime Text directly.
The service accepts source paths only within this repository.

## Failures

The console shows command rejections in the error area.
The **Worker log** tab shows startup and runtime errors.
A missing model endpoint prevents real cognition from completing.
The dummy fixture verifies wiring without establishing companion quality.

## Code links

[`pm.console.server.serve`][], [`pm.console.server.Instance.command`][], and [`pm.console.worker.run`][] implement this procedure.
Read the [development loop](development.md) for code changes and verification.
