# Durable body leases and epochs

## Summary

One operating-system lock owns each configured body endpoint.
A durable counter issues a new body epoch after each acquisition.
The Minecraft MCP independently rejects obsolete tokens.
A replacement claim waits for the previous server command, including work from a dead client.

Parent: [body recovery](body-recovery.md).
Related: [Minecraft actor controls](minecraft-actor-controls.md) and [durable projects](minecraft-tasks.md).

## Local ownership

[`workflows.minecraft_body.lease.BodyLeases`][] keys ownership by the normalized MCP URL.
It removes a trailing slash and normalizes `localhost` and `::1` to `127.0.0.1`.
The request's `body_id` remains a label.
Use one configured endpoint per avatar.

`PM_BODY_LEASE_ROOT` selects the shared registry.
The Windows default is `%LOCALAPPDATA%/pm-next-v2/body-leases`.
The other-host default is `~/.local/share/pm-next-v2/body-leases`.
All cooperating PM processes must use this same directory.
Do not remove or replace lock files while a process holds a lease.

Each endpoint has a hashed filename and these durable files:

| File | Owner and meaning |
| --- | --- |
| `.owner.lock` | One actor owns its byte lock until cleanup or process exit |
| `.command.lock` | Local reads and commands serialize at the dispatch boundary |
| `.json` | Endpoint, owner, epoch, session UUID, PID, active status, and timestamps |

[`pm.host.durable_files.ProcessLock`][] uses `msvcrt` on Windows and `flock` elsewhere.
A second owner receives `BodyBusy` immediately.
A managed worker retains that expected contention as a failed task and publishes its detail without stopping the application.
[Singleton admission](managed-singletons.md) prevents overlapping starts within one supervisor before actor execution.
A queued command checks cancellation while it waits.
The operating system releases a dead process's lock.
There is no elapsed-time takeover and no PID-only ownership test.
[`pm.host.durable_files.atomic_json`][] synchronizes the new epoch before atomic publication.

## Server ownership

The owning repository is `C:/source/pm/pm-minecraft`.
`mcmcp.body_fence.BodyFence` retains `body-epoch.json` in the configured character directory.
`mcmcp.server.CommandGate` orders claims against active commands.
The standard body tools keep their existing argument and result schemas.
Three transport tools add the fencing boundary:

| Tool | Inputs and behavior |
| --- | --- |
| `minecraft_body_claim` | `owner`, positive `epoch`, and nonempty `session`; waits for the old command and commits the higher epoch |
| `minecraft_body_call` | Token fields, original `tool`, and `arguments`; validates token and original tool arguments before dispatch |
| `minecraft_body_release` | Token fields; waits for cleanup and retains an inactive high-water epoch |

An equal active claim is idempotent only for the identical owner and session.
An older epoch or changed session raises `PermissionError` before body access.
An inactive equal epoch cannot reactivate an earlier session.
Direct read tools remain available.
Direct mutations without the active token fail while an owner holds the body.
Horizon scans require ownership because they rotate the camera.
The actor's model does not receive these transport tools in its schema list.

The server cannot release an in-flight command merely because its HTTP client exited.
The command thread retains the gate until its runtime call finishes.
The next claim waits at that gate before its fresh observation.
Concurrent `minecraft_stop` validates the same token and stops the current epoch's command.
This control path can complete while normal dispatch remains busy.

## Restart and deployment

MCP startup retains the high-water epoch and invalidates an earlier active token.
A test-controlled body disconnection also invalidates its active token.
The replacement actor must obtain a higher local epoch and observe the new body.
Keep the character directory during MCP restart.
Keep the local registry during PM restart.
A restored registry with an older counter receives an explicit server rejection.
It cannot silently reuse an earlier token.

The live transport remains `http://127.0.0.1:6767/mcp`.
The body HTTP service remains on port 6768 and the viewer remains on port 6769.
Deploy the tested pm-minecraft checkout before enabling A12 clients.
[`workflows.minecraft_body.mcp_client.build_minecraft_tools`][] carries the fenced capabilities on the granted transport closure.
The root dependency lock does not install pm-minecraft; its separate process uses that repository's own interpreter and checkout.

The root client and native actor use a 660-second request timeout.
This covers the body's permitted 600-second navigation action and result reads.
The pinned pm-coder loader honors `requestTimeoutMs: 660000` in the actor's MCP descriptor.
An earlier five-minute client deadline left an actual navigation command active after its client timed out.
The repaired loader retains transport headers, environment expansion, discovery, and automatic compaction.

The protocol assumes trusted local clients and one active MCP service per avatar.
Epoch tokens provide stale-command fencing, rather than an authentication boundary.
The gate serializes effects; it does not roll them back.
Read [A12 verification](../work_items/verification/a12-body-recovery.md) for captured real-server evidence.
