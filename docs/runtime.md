# Direct workflow execution

## Summary

The shared runner calls the real pm-workflows kernel synchronously.
Direct debugger calls and module threads use this same runner.
The runner supports Python roles and real pm-coder roles on folder and virtual Ghost targets.

Parent: [project overview](index.md).

## Input and result

`WorkflowRequest` contains a unique run ID and a YAML-compatible payload.
`WorkflowRuntime` supplies the working environment, tool functions, cancellation signal, and Python error policy.
The runner writes `inbox/request.yaml`, calls the kernel, and normalizes its result.
Only a completed run loads `outbox/report.yaml`.

| Status | Meaning |
| --- | --- |
| `completed` | The expected terminal status and exit route match. The report is a YAML mapping. |
| `failed` | The role contract or terminal route failed. No report returns. |
| `cancelled` | The cancellation signal or kernel termination prevents a returned report. |
| `waiting_for_input` | The kernel suspended for external user input. No report returns. |

An explicit successful `stop` route sets the kernel's `ok` field to `False`.
The runner checks the terminal status and exit reason instead of that field alone.
The result preserves the raw kernel summary and artifact directory.
Invalid report YAML and Python errors remain visible exceptions, not successful empty results.

## Debugger entry point

Use `.venv/Scripts/python.exe` as the debugger interpreter.
Run these direct tests without module threads:

```powershell
uv run --locked python -m unittest tests.test_workflow_runner tests.test_agent_file_scope -v -b
```

Place a breakpoint in `WorkflowRunner.invoke()` or the Python role's `run(context)` function.
The role runs on the calling thread.
Separate threaded tests cover module lifecycle behavior.

This example runs the existing dummy inner-thought workflow without a module, bus, model, or Ghost commit:

```python
from pathlib import Path
from threading import Event
from uuid import uuid4

from pm_workflows.runtime import WorkflowRuntime
from runtime.workflow_runner import WorkflowRequest, WorkflowRunner
from working_target import GhostBashEnvironment

environment = GhostBashEnvironment({}, {}, "read")
runtime = WorkflowRuntime(
    environment, Event(), {}, propagate_python_errors=True,
)
runner = WorkflowRunner(
    Path("workflows/inner_thought.workflow.md"), Path(".runtime/debug"),
)
result = runner.invoke(WorkflowRequest(uuid4().hex, {"messages": []}), runtime)
print(result.status)
print(result.report)
```

The report contains the fixed thought `I wonder what Jesse will ask next.`
The caller owns proposal acceptance and Ghost commits. The generic runner knows nothing about Ghost.

## Module behavior

`ShellModuleWorkflow.handle_batch()` opens the target and prepares the request.
It passes that request to the shared runner.
`last_result` exposes the typed outcome. `last_kernel_summary` retains compatibility with existing diagnostics.
Only completed runs reach `publish_output()`.
A failed run raises an error. Cancelled and waiting runs return without publication.
The optional `agent_model` constructor argument enables pm-coder roles through the same runner.
The caller owns model activation and the supplied context window.

## Agent roles

Supply `WorkflowRuntime.agent_model` with a resolved `pm_workflows.runtime.AgentModel`.
It contains `base_url`, `model`, `api_key`, and a positive `context_window`.
The optional `enable_thinking` setting defaults to true. `live_test=True` makes inference errors reach the caller without reconnect retries.
The runner selects pm-coder when this connection is present. Python skill files still use the Python driver.

`FolderEnvironment` binds tools to its real root.
`GhostBashEnvironment` binds tools to its BashMachine and the `workflow` user.
The virtual user's permissions also apply to read, write, edit, glob, and search tools.
Automatic host instructions, skills, and MCP discovery are disabled for these runtime roles.
The driver supplies the selected role's instructions directly.

Run the actual agent checks with the configured model endpoint:

```powershell
uv run --locked python -m unittest tests.live.test_agent_workflows -v -b
```

These synchronous checks retain evidence under `.runtime/w3/agent-targets/`.
Each agent reads a request and a memory file, then writes a report through the supplied target.
The checks establish the report, tool calls, unchanged memory, and absence of a report in the host journal workspace.
One check starts a scoped Ghost MCP service on a loopback port and records the agent call.

## Scoped services and run control

`WorkflowRuntime.tools` is the run-level Python grant.
A role that declares `tools` sees exactly those names.
An undeclared name is a `ManifestError`.
`WorkflowRuntime.mcp_servers` supplies named scoped service descriptors.
This field replaces shared `.mcp.json` discovery when present.
The kernel writes a private configuration for each attempt.
[`runtime.mcp_services.ScopedMcpRegistry`][] starts each named FastMCP service on an OS-assigned loopback port.
It holds the bound socket for its lifetime.
It closes the socket after success, error, or cancellation.
`WorkflowRuntime.child_factory(scope)` opens the runtime for one child invocation.
Without this factory, a child reuses the parent runtime.

A run's cancellation is separate from module shutdown.
`ShellModuleWorkflow.cancel_run()` cancels one run.
`stop()` and a terminate control also cancel the active run.
A run that suspends for user input publishes `workflow.question` and exposes its question ID and resume phase. A `workflow.answer` data message resumes it through a durable `pm.answer-receipt.v1`.
`ghost_mcp.build_tools(environment, batch)` adds `ghost_propose_create` and `ghost_propose_replace` over the run's `GhostWriteBatch`.
Proposals stay private until the batch commits.

## Current limits

Python cancellation is cooperative. The runner cannot forcibly stop arbitrary Python code.
Agent cancellation interrupts supported asynchronous inference and waits for session cleanup.
Synchronous tools remain cooperative. Completed external effects cannot be undone.
The permanent-error classifier is marker and status based, so unusual provider wording still enters the reconnect loop.
MainCogLoop consumes managed questions and lifecycle events. Assistant and SelfManagement can deliver correlated answers.
The cognitive parent invokes seven child workflows through scoped runtimes.

## Code and evidence

- [`runtime.workflow_runner.WorkflowRunner`][]
- [`workflow_adapter.ShellModuleWorkflow`][]
- Direct runner tests: `tests/test_workflow_runner.py`
- Direct file-permission tests: `tests/test_agent_file_scope.py`
- Run control tests: `tests/test_run_control.py`
- MCP lifetime tests: `tests/test_mcp_lifetime.py`

## Related guides

Read the [system map](concepts/architecture.md) and [development loop](workflows/development.md).
