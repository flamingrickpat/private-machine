> Historical dummy-prototype handoff, September 12, 2026.
> For the next implementation, start with [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
> The new plan supersedes the dummy-only scope and the earlier supporting specifications.
> This file remains as evidence for the existing prototype.

Build a small, runnable Spamuel prototype in `C:\source\pm\pm_next_v2`.
Use fixed cognitive outputs, real Ghost objects, the real Bash Machine, and the real workflow kernel.
The purpose is to explore module boundaries before we add agentic behavior.

Read `ADAPTER_NOTES.md` for source locations and integration details.
Keep those details outside the module wiring and cognitive logic.
Read `IMPLEMENTATION.md` for the fixed interfaces and file-by-file changes.
Read `DIRECT_TESTS.md` for module checks that require no threads.
These specifications replace the open implementation choices in this handoff.

The destination already contains a PyCharm starter, `.venv`, and `pyproject.toml`.
The source project is `C:\source\pm\pm_next`.
The repeated dependency name in the request means `pm-workflows` and `pm-coder`.

The user authorizes direct changes in `C:\source\pm\pm-workflows` on its `steward` branch.
Implement the kernel changes that this prototype needs, including support for the virtual working target.
Prefer a small shared interface over workarounds in the application.
The user accepts experimental changes and can revert them.
Keep these changes focused on this milestone.
The branch name does not add steward-agent implementation to the scope.

Use the `hackable` and `simple-english` skills.
The user explicitly requests ABC interfaces and fixed cognitive outputs.
These requests take precedence over the skill defaults about abstractions and simulated behavior.
The fixed outputs are the prototype behavior, not evidence of LLM integration.

Build only this scope:

- One process with module threads and an in-memory bus.
- One Ghost loaded from the existing Spamuel YAML data.
- A Ghost working target with a virtual Bash environment.
- A Ghost MCP interface with read and analysis tools.
- A shared workflow module with small subclasses for cognitive behavior.
- Fixed outputs for dialogue, main cognition, inner thoughts, subprocess scheduling, subprocess status, and tool-plus-dialogue.
- One runnable demonstration and one interactive terminal entry point.

Defer Minecraft execution, coding execution, triage, HTTP transport, containers, stewards, optimization, and replay systems.
Do not copy the old supervisor, web interface, worker leases, or persistence machinery.
Do not change the running Hadal workflow or its task files.

The application wiring is small:

```text
UserInputProxy -> bus -> MainCogLoop
                           |
                 GWT + action schemas
                           |
              +------------+--------------+----------------+
              v            v              v                v
           Dialogue    Schedule       CheckSubprocess  ToolPlusDialogue
                       Subprocesses
              |            |              |                |
              +------------+--------------+----------------+
                           |
                   results / telemetry
                           v
                      MainCogLoop

InnerThoughtLoop -> thought knoxels -> MainCogLoop
Dialogue -> public reply -> UserInputProxy
All memory changes -> shared in-memory Ghost
```

All user input enters the MainCogLoop before any action module.
The MainCogLoop can emit several action schemas from one input batch.
Each action carries the GWT context for that decision.
Action modules return results through the bus.
Public dialogue also reaches the user output subscription.

The MainCogLoop remains active throughout the session.
Each iteration drains the currently available input batch.
Messages that arrive during processing belong to the next batch.
An empty iteration waits for input or the next configured tick.
The loop does not produce unlimited empty GWT records.

Each module selects relevant messages through one predicate.
The bus delivers each message to every matching module inbox.
A consumer only removes messages from its own inbox.
The console has its own output subscription.
Published messages are immutable by convention.

Use a small envelope with `id`, `source`, `kind`, `correlation_id`, and `payload`.
Use `target` for commands that address one module.
The payload carries existing knoxel objects or GWT records.
Use `data`, `telemetry`, and `control` as the message kinds.
Only `terminate` is a control command in this prototype.

Use these shared interfaces:

| Interface | Shared behavior | Subclass behavior |
| --- | --- | --- |
| `ShellModule` | Inbox, thread lifecycle, one iteration method, termination | Message interest and batch handling |
| `ShellModuleWorkflow` | Input files, workflow invocation, output files | Conversion between bus messages and workflow input/output |
| `WorkingTarget` | File access and Bash environment contract | Real folder or virtual Ghost environment |

`run()` starts the module thread and returns.
The thread repeatedly calls the same iteration method.
The demonstration uses that iteration method too.
The common base owns lifecycle behavior.
Subclasses do not copy thread loops, queue handling, or termination code.

Use small hooks such as `accepts(message)`, `prepare_input(batch)`, and `publish_output(report)`.
Use the exact signatures in `IMPLEMENTATION.md`.
Keep workflow execution in the shared adapter.
Keep fixed cognitive behavior in the corresponding Python workflow roles.
Use subclasses to express routing differences.

A workflow module waits for data input.
Each accepted input batch starts one workflow invocation.
The adapter writes the request into the target inbox.
The workflow writes its final report into the target outbox.
The subclass converts that report into bus messages.
The module then waits for more input.
One module runs one invocation at a time.

The base handles `terminate` without an agent.
An idle module stops promptly.
An active dummy workflow stops at its next cooperative boundary.
It emits a terminated result instead of a completion report.
Do not claim that Python threads support forced termination.

The fixed modules have these responsibilities:

| Module | Prototype behavior |
| --- | --- |
| `UserInputProxy` | Wrap terminal input and print public replies. |
| `MainCogLoop` | Store incoming knoxels, create a fixed GWT, and select actions through a few explicit rules. |
| `Dialogue` | Read its GWT and Ghost view, then return a fixed public response. |
| `InnerThoughtLoop` | Produce a fixed thought on a configured tick. |
| `ScheduleSubprocesses` | Record a dummy task and emit accepted, progress, and completed telemetry. |
| `CheckSubprocess` | Read the shared dummy task records and return current status. |
| `ToolPlusDialogue` | Call a local Ghost analysis tool and return a fixed response with the result. |

The scheduler owns one small task table.
The status module reads that same table.
No external subprocess is necessary for this milestone.
Telemetry carries the task identity and original correlation identity.
The MainCogLoop stores telemetry without rescheduling the same task.
An inner thought does not automatically trigger another inner thought.

For a Minecraft request, emit a scheduling action and a dialogue action.
The reply states that the request will be scheduled.
The scheduler reports that actual Minecraft execution is unavailable in this prototype.
Use a generic dummy task to demonstrate progress and completion.

The Ghost remains the shared memory object in RAM.
Modules add knoxels through one small synchronized write method.
That method owns identity allocation and index updates.
The prototype does not require Ghost exports, disk watchers, or Git commits for memory changes.

Keep the source layout small:

```text
main.py                 construct objects, connect modules, start application
shell.py                bus, envelope, shared module lifecycle
modules.py              cognitive subclasses and their routing
working_target.py       folder and virtual environment adapters
ghost_mcp.py            Ghost tools and MCP registration
workflow_adapter.py     pm-workflows integration details
workflows/              manifests and fixed Python roles
data/                   copied Spamuel seed data
demo.py                 repeatable scenarios through the same runtime
```

Keep copied Ghost models in one local package.
Use functions for stateless helpers.
Do not create a separate class for every conversion.

Implement in this order:

1. Inspect the destination and preserve user changes.
2. Initialize Git if the destination is not a repository.
3. Install compatible local packages, with `pm-workflows` editable from its `steward` checkout.
4. Copy the required Ghost models, loader, and Spamuel seed data.
5. Load Ghost without an LLM or embedding endpoint.
6. Build the Ghost working target and its read tools.
7. Connect one real Python-role workflow through `ShellModuleWorkflow`, including the required changes in `pm-workflows`.
8. Add the remaining fixed modules through the same interfaces.
9. Connect the application in `main.py`.
10. Run the scenarios and record the actual output.

Completion requires these observable results:

- A user message passes through MainCogLoop and produces a correlated public reply.
- Two queued inputs appear in one cognitive batch.
- A thought and a GWT appear in the in-memory Ghost.
- Two interested subscribers both receive the same bus message.
- A dummy task emits telemetry that enters a later cognitive batch.
- A status request reads the task record through MainCogLoop routing.
- Dialogue reads permitted virtual Ghost files and cannot change those files.
- A second invocation sees a memory change from the first invocation.
- A Ghost MCP tool returns data from the same Ghost instance.
- An idle module and an active cooperative workflow both accept termination.
- The process exits cleanly after the demonstration.
- No LLM, Minecraft server, or embedding endpoint receives a request.

Use real integration runs and plain assertions in `demo.py`.
Use standard-library unittest for the requested direct module tests.
Do not add mocked workflow behavior or a third-party test framework to the application.
Report the command, actual output, and any untested path.
Label fixed outputs as dummy behavior in the demonstration.

Let programming errors produce their original traceback.
Expose worker errors to the application so the console cannot wait forever after a crash.
Handle permanent external input errors with a clear message.
Keep transient retries at the external adapter that owns the failed call.
Do not add retries around cognitive logic.

Finish with a short README and the exact run command.
Describe the changes in both repositories and the verification results.
List any remaining pm-workflows integration gap separately.
The next development step replaces fixed Python roles with agents while preserving the module wiring.
