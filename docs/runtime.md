# Direct workflow execution

## Summary

The shared runner calls the real pm-workflows kernel synchronously.
Direct debugger calls and module threads use this same runner.
The runner supports Python roles and real pm-coder roles on folder and virtual Ghost targets.
The complete W3 gate remains unfinished.

Parent: [project overview](../README.md).
Related: [runtime plan](plan/runtime.md), [Ghost commits](ghost.md), and [W3 evidence](work-items/W3.md).

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
The checks verify the report, tool calls, unchanged memory, and absence of a report in the host journal workspace.

## Current limits

Per-run cancellation is not yet separate from module shutdown in `ShellModuleWorkflow`.
Standalone callers can supply independent cancellation events.
Python cancellation is cooperative. The runner cannot forcibly stop arbitrary Python code.
Agent cancellation interrupts supported asynchronous inference and waits for session cleanup.
Synchronous tools remain cooperative. Completed external effects cannot be undone.

A waiting result preserves evidence, but this checkpoint adds no bus question routing or resume operation.
Do not reuse a run ID to resume through `invoke()`. Each invocation starts a new kernel run.
Child namespaces, narrowed grants, and dynamic MCP listeners remain unfinished.
The production coder retry loop still needs permanent-error classification.
Runtime roles that request MCP services require explicit grants.

## Code and evidence

- [Shared runner](../runtime/workflow_runner.py)
- [Bus adapter](../workflow_adapter.py)
- [Direct runner tests](../tests/test_workflow_runner.py)
- [Direct file-permission tests](../tests/test_agent_file_scope.py)
- [W3 progress](work-items/W3.md)
