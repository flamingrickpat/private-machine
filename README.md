# pm-next-v2 — Spamuel dummy cognition prototype

This project builds a small, runnable prototype of the Spamuel companion. It
uses fixed (dummy) output, a real in-memory Ghost, a real virtual Bash
machine, and the real pm-workflows kernel. The goal is to explore module
boundaries before agentic behavior is added.

The banner says `DUMMY COGNITION — no LLM calls`. No LLM, Minecraft server, or
embedding endpoint is contacted.

## What it does

One process runs module threads over an in-memory bus:

```
UserInputProxy -> bus -> MainCogLoop
                           |
                 GWT + action schemas
                           |
     +------------+------------+----------------+
     v            v            v                v
  Dialogue     Schedule   CheckSubprocess   ToolPlusDialogue

InnerThought -> bus -> MainCogLoop
```

Every user input enters `MainCogLoop` first. It creates one GWT record and
selects fixed actions through a few explicit rules. Action modules return
results through the same bus. Public replies reach the terminal output.

The Ghost stays in RAM. Its records are projected into a virtual Bash
environment as read-only files (`/work/ghost/knoxels/<id>.yaml`). The Bash
machine lives in memory too. Kernel journals, requests, state, traces, and
result metadata go to disk under `.runtime/`. Ghost files stay virtual.

## Requirements

- Python 3.12 (the project declares `>=3.12,<3.13`).
- A virtual environment named `.venv312`.

The direct application dependencies are:

- `pm-workflows` (editable from `../pm-workflows`, branch `steward`)
- `pm-coder` (editable from `../pm-coder`)
- `pydantic`, `PyYAML`, `fastmcp`

## Run

```powershell
.\.venv312\Scripts\python.exe demo.py
```

This runs the full scenario set and prints one `PASS` line per check.
On this machine, PATH selects an unusable WSL Bash launcher.
For the real-folder demonstration, select Git Bash first:

```powershell
$env:PATH = 'C:\Program Files\Git\bin;' + $env:PATH
.\.venv312\Scripts\python.exe demo.py
```

```powershell
.\.venv312\Scripts\python.exe main.py
```

This starts the interactive terminal session.

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
```

This runs direct tests and the separate threaded integration tests.

For direct debugger checks without module threads:

```powershell
.\.venv312\Scripts\python.exe -m unittest tests.test_modules tests.test_shell tests.test_design_contracts -v
```

## Commands in the terminal

| Input | Fixed output |
| --- | --- |
| any text | `I heard you, Jesse.` |
| contains `minecraft` | schedules, then `I will schedule your Minecraft request.` |
| `dumb robot` | `You are a dumb human, Jesse! Dumb and nasty.` |
| `/demo-task` | schedules a dummy task, then `I will schedule a dummy task.` |
| `/memory` | one tool-plus-dialogue reply with a scoped search count |
| `/status <id>` | a status reply for the given task id |

These fixed outputs are prototype behavior, not real character policy.

## Debugger example

This defines no threads and performs no sleeps.

```python
import time
from main import build_app, DEFAULT_SEED

replies = []
app = build_app(
    seed_path=DEFAULT_SEED,
    runtime_root=".runtime",
    thought_period=30.0,
    task_period=1.0,
    clock=time.monotonic,
    emit_reply=replies.append,
)
correlation_id = app.user_input.submit("you are a dumb robot!")
app.main_cog.step()          # breakpoint: prepare input, role, Ghost commit, bus output
app.dialogue.step()          # breakpoint: real Python workflow and reply
app.user_input.step()        # breakpoint: public output
assert replies == ["You are a dumb human, Jesse! Dumb and nasty."]
```

You can inspect the `Ghost`, task table, latest environment, and recorded
messages after each line. After a direct workflow step, `module.last_environment`
and `module.last_kernel_summary` hold the most recent invocation.

## Source layout

```text
main.py                 build_app(), CLI, terminal input, shutdown
shell.py               bus, envelope, module lifecycle
modules.py              cognitive subclasses and TaskTable
working_target.py       virtual Ghost Bash environment and folder target
ghost_mcp.py            scoped Ghost tools and MCP registration
workflow_adapter.py     pm-workflows invocation
workflows/              six manifests and six fixed Python roles
pm_core/                copied Ghost model subset and loader
data/                   copied Spamuel seed data
demo.py                 scripted scenarios
tests/                  thread-free unittest checks
```

The copied Ghost model subset is `pm_core/model/*.py` (except
`knoxel_list.py`), plus `pm_core/consts.py`, `pm_core/utils/datetime_utils.py`,
`pm_core/persist/dataprovider_yaml.py`, and `pm_core/ghost/ghost_base.py`. The
LLM import and the `CodeletState` import were removed from the copied subset.

## Changes in pm-workflows

On the `steward` branch:

- `pm_workflows/working_target.py` — new `CommandResult`, `WorkingEnvironment`,
  and `WorkingTarget` ABCs.
- `pm_workflows/runtime.py` — new `WorkflowRuntime` and `WorkflowTerminated`.
- `pm_workflows/python_role.py` — added `runtime` to `RoleContext`.
- `pm_workflows/kernel.py` — accepts an optional `runtime`, passes it to Python
  roles, and handles cooperative cancellation at phase boundaries.
- `pm_workflows/drivers/python_driver.py` — re-raises `WorkflowTerminated` and
  optionally re-raises other Python exceptions.
- `tests/test_runtime_context.py` — integration checks for the new behavior.

The existing real-folder workflow path still works without a runtime.

The audit also adds two compatible options to the local pm-coder checkout:
`BashMachine.write_texts()` registers a lazy projection in one batch.
`exec(..., strip_output=False)` preserves file whitespace.
The prototype uses both through its working target.

## Verification

- `python demo.py`: 21 checks passed with Git Bash first on PATH.
- unittest discovery: 34 tests passed, including the full Spamuel seed and threaded integration checks.
- `pm-workflows`: 55 runtime-context, driver, and kernel tests passed.
- `pm-coder`: 30 real BashMachine tests passed.

The audit used the prototype's `.venv312` interpreter for all three repositories.

## Remaining integration gap

The manifest template in the handoff omitted `on_invalid` for role phases. The
current kernel requires it. Every manifest here adds
`on_invalid: {action: stop}`. The audit records other design corrections in
`AUDIT.md`. Virtual targets currently work through Python roles. Agent-driver
integration remains a future step.

## Next step

The next development step replaces the fixed Python roles in `workflows/roles.py`
with agents, while keeping the module wiring in `modules.py` unchanged.

Read [AUDIT.md](AUDIT.md) for the design audit and remaining prototype limits.
