# Test a Minecraft run

Parent: [managed work](../subsystems/managed-work.md).
Next: [evidence and cleanup](minecraft-evidence.md).

## Summary

The default Minecraft run uses a [sustained specialist](../subsystems/minecraft-actor.md) with many real tool calls.
Its request has no action-count ceiling.
Explicit W8 actions retain the narrow observe/action/observe path described below.
The [durable project workflow](../subsystems/minecraft-tasks.md) adds generic task trees and independent completion verification.
[Interruption recovery](../subsystems/body-recovery.md) is delivered.
[Expected gameplay failures](../subsystems/gameplay-recovery.md) have an implementation with real acceptance pending.
Cooperative survival remains later alpha work.

The normal workflow requires an existing Minecraft server and MCP service.
It opens MCP client connections when needed. It does not start or stop the MCP process.
The isolated live-test harness can start its own server and MCP. This is a separate test path.

## Prepare the services

1. Start your Minecraft server. Your configured game port is `12345`.
2. Configure the MCP in `C:/source/pm/pm-minecraft/.env_real`.
3. Set `MINECRAFT_PORT=12345` in that file. Check the host, player, MCP port, and RCON settings.
4. From the `pm-minecraft` directory, start the MCP:

```powershell
.venv/Scripts/python.exe main.py .env_real
```

5. Add the actual MCP URL, including `/mcp`, to Spamuel's model configuration:

```yaml
minecraft:
  mcp_url: http://127.0.0.1:6767/mcp
  request_timeout_seconds: 660
  probe_timeout_seconds: 5
```

Use the file that you give with `--config`, or the file selected for the console instance.
`MINECRAFT_MCP_URL` remains a test override. Normal operation reads the configuration section.
6. Start the Spamuel instance. A wrong endpoint stops the start before the first cycle.
7. Open **Connections** in the console and check each endpoint.

The game port and MCP HTTP port are different ports.
A reachable game socket does not prove that the MCP body has joined the world.
The MCP checks that RCON and its player body connect to the same server.
Configure RCON on your actual server; a listening game port alone is insufficient.
Tool discovery proves MCP transport availability. A successful `minecraft_observe` proves more about the body connection.
The connection panel probes services. It does not start them or perform world actions.

The MCP configuration sets `MCMCP_AGENT_HOME`, its persistent character directory.
Startup creates missing directories and templates. Startup does not clear existing character files.

## Run the explicit boundary check

Ask Spamuel to start the managed `minecraft` workflow with a precise goal and an allowed tool list.
For the first check, allow only `minecraft_observe`.
For a world-changing check, use an explicit action and arguments that match the discovered tool schema.
Natural-language requests still depend on cognition choosing the managed-work action.
They are not a guaranteed command interface.

Use `mode: boundary_check` for the narrow experiment.
An explicit `action_tool` also selects this path.
The request accepts `goal`, `action_tool`, `action_arguments`, `allowed_tools`, and `include_image`.
An empty allowed list does not restrict action names. Supply a list for a bounded experiment.
The console **Managed runs** panel controls existing runs; it does not directly create these requests.

Each run follows this sequence:

1. The supervisor stores a unique run receipt and task directory.
2. The role reads the request and current managed instructions.
3. It calls `minecraft_observe` and writes preliminary evidence.
4. It consumes pending instruction updates at its explicit boundary.
5. It checks cancellation and selects one allowed action.
6. It executes that action and observes the world again.
7. It writes the report, world records, and screenshot files.
8. The supervisor stores terminal state and calls consolidation.

An explicit action bypasses planning. Otherwise, the planner receives observation text and discovered tool schemas.
Screenshot pixels remain evidence files; they do not enter the planning prompt.
The report accepts a run only when required observations and the chosen action succeed.
Check both the terminal receipt and `report.accepted`. Kernel completion alone does not prove task success.

## Update, cancel, and stop

An update changes the receipt revision immediately.
It changes execution only when the role consumes that revision at its boundary.
After the final boundary, further updates fail.
Cancellation is cooperative. It does not undo mined blocks or built structures.
An in-flight external call can finish before cancellation takes effect.

Keep the MCP running across consecutive runs for the same character.
After all active runs finish, stop the MCP when you no longer need its body connection.
Stop the Minecraft server separately when you want to end the world session.
Stopping Spamuel does not stop either external server.

## Source and tests

Read [`workflows.minecraft.schemas.MinecraftRequest`][], [`pm.runs.supervisor.ManagedRunSupervisor`][], and [`workflows.minecraft_body.mcp_client.build_minecraft_tools`][].
Read [actor controls](../subsystems/minecraft-actor-controls.md) for sustained updates, cancellation, and direct inventory dialogue.
The role is `workflows/minecraft/steps.py`; its manifest is `workflows/minecraft/minecraft.workflow.md`.

`tests/live/test_minecraft_body.py` and `tests/live/minecraft_stack.py` define the isolated live gate.
The harness starts services only for its eligible local endpoint when autostart is enabled.
It stops services that it owns and removes its disposable server copy.
It does not provide normal application lifecycle management.
