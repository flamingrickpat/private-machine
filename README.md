# pm-next-v2 — Spamuel dummy cognition prototype

This project builds a small, runnable prototype of the Spamuel companion. It
uses fixed (dummy) output, a real in-memory Ghost, a real virtual Bash
machine, and the real pm-workflows kernel. The goal is to explore module
boundaries before agentic behavior is added.

The terminal and demo banner says `DUMMY COGNITION — no LLM calls`.
Those entry points contact no LLM, Minecraft server, or embedding endpoint.
The W1 model tests use real generation and embedding endpoints.

## Next implementation plan

The current code remains the dummy prototype described here.
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) specifies the next real companion implementation.
It separates wiring, runtime contracts, models and memory, workflow behavior, tests, and ordered work items.
The plan supersedes the old dummy-only handoff. It does not claim that the new features already work.

W1 is complete: model presets, explicit activation, direct chat calls, and compatible embedding cache access.
Read the [model guide](docs/models.md) for setup and debugger examples.
The [W1 delivery record](docs/work-items/W1.md) contains commands, actual results, and implementation limits.
W2 adds isolated bus delivery, scoped read views, atomic Ghost batches, and checkpoint recovery.
Read the [Ghost and bus guide](docs/ghost.md) for the persistence boundary and direct debugger examples.
The [W2 delivery record](docs/work-items/W2.md) contains its evidence and limits.
W3 is the next unfinished work item: the complete workflow-to-agent boundary.
Its current checkpoint installs pm-coder 0.8.0 with separate sessions and user-scoped virtual file tools.
The actual workflow agent driver still needs the full target, MCP, and cancellation integration.
Application version 0.4.0 also provides the [shared synchronous runner](docs/runtime.md) for direct debugger calls and module execution.
Read the [W3 progress record](docs/work-items/W3.md) for completed checks and remaining work.

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

The Ghost stays in RAM and saves accepted state to a checkpoint in the terminal session.
Its records appear as read-only virtual files (`/work/ghost/knoxels/<id>.yaml`).
The Bash machine lives in memory. Kernel journals, requests, traces, and results go to disk under `.runtime/`.
The checkpoint is one JSON file, not a real directory of projected Ghost files.

## Requirements

- Python 3.12 (the project declares `>=3.12,<3.13`).
- `uv`, Git, and access to the private GitHub repository.
- One virtual environment named `.venv`, managed by uv.

The direct application dependencies are:

- `pm-workflows` from GitHub `main`, pinned to a commit in `uv.lock`
- `pm-coder` from GitHub `main`, pinned to a commit in `uv.lock`
- `pydantic`, `PyYAML`, `fastmcp`

## Run

```powershell
uv sync --locked
uv run --locked python demo.py
```

This runs the full scenario set and prints one `PASS` line per check.
On this machine, PATH selects an unusable WSL Bash launcher.
For the real-folder demonstration, select Git Bash first:

```powershell
$env:PATH = 'C:\Program Files\Git\bin;' + $env:PATH
uv run --locked python demo.py
```

```powershell
uv run --locked python main.py
```

This starts the interactive terminal session.
It restores `.runtime/ghost.json` when that checkpoint exists.
Otherwise, it loads the Spamuel seed and saves the first accepted change there.
`--checkpoint PATH` selects another checkpoint. `--ephemeral` disables accepted-state persistence.
An invalid checkpoint raises an error instead of silently resetting memory.

```powershell
uv run --locked python -m unittest discover -s tests -v -b
```

This runs direct tests, threaded integration tests, and required live model tests.
The live tests require Qwen, embedding credentials, and the compatible Spamuel cache.
Missing requirements fail the suite. Read the [model guide](docs/models.md#tests) for W1 foundation-only commands.

For direct debugger checks without module threads:

```powershell
uv run --locked python -m unittest tests.test_modules tests.test_shell tests.test_design_contracts -v -b
```

`.python-version` selects Python 3.12. The old Python 3.13 `.venv` and editable `.venv312` are obsolete.
Select `.venv\Scripts\python.exe` as the IDE interpreter.
Normal startup uses the committed lockfile and requires no sibling source checkouts.

## Update from GitHub

```powershell
git pull --ff-only
uv sync --locked
uv run --locked python main.py
```

Implementation agents publish dependencies before refreshing this application's lockfile.
Read [commits and GitHub delivery](docs/plan/delivery.md) for the authorized branches, version rules, and release sequence.
Do not replace the lockfile's tested commits with local editable dependencies for final validation.

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
This `build_app()` example uses an in-memory Ghost.
Add `checkpoint_path=".runtime/debug-ghost.json"` for persistent debugger sessions.

## Source layout

```text
main.py                 build_app(), CLI, terminal input, shutdown
bus/                    serialized envelopes, private mailboxes, in-process delivery
shell.py                compatibility imports for bus and module lifecycle
modules.py              cognitive subclasses and TaskTable
working_target.py       virtual Ghost Bash environment and folder target
ghost_mcp.py            scoped Ghost tools and MCP registration
workflow_adapter.py     pm-workflows invocation
workflows/              six manifests and six fixed Python roles
pm_core/                Ghost model subset, loader, and accepted state
  ghost/service.py      scoped snapshots and atomic batch acceptance
  ghost/batch.py        private new records and replacements
  persist/snapshot.py   versioned JSON checkpoint and atomic replacement
  embeddings.py         shared embedding service and compatible cache
runtime/                model config, HTTP requests, activation, direct chat
  shell.py              common module step, thread lifecycle, application owner
ghost.py                compatibility import for the Ghost service
config.example.yaml     credential-free model and cache config
data/                   copied Spamuel seed data
demo.py                 scripted scenarios
tests/                  direct tests and separate threaded integration checks
  live/                 required real generation and embedding tests
```

The copied Ghost model subset is `pm_core/model/*.py` (except
`knoxel_list.py`), plus `pm_core/consts.py`, `pm_core/utils/datetime_utils.py`,
`pm_core/persist/dataprovider_yaml.py`, and `pm_core/ghost/ghost_base.py`. The
LLM import and the `CodeletState` import were removed from the copied subset.

## Changes in pm-workflows

On `main`, included in version 0.3.0:

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

The audit also adds two compatible options, published in pm-coder 0.7.0:
`BashMachine.write_texts()` registers a lazy projection in one batch.
`exec(..., strip_output=False)` preserves file whitespace.
The prototype uses both through its working target.

## Verification

- `python demo.py`: 21 checks passed with Git Bash first on PATH.
- unittest discovery: 90 tests passed, including the full Spamuel seed, threaded integration, and live model checks.
- W1 foundation checks: 24 tests passed without model inference.
- W2 checks: 29 tests passed for bus isolation, commits, restoration, concurrency, and direct persistent workflows.
- `pm-workflows`: 55 runtime-context, driver, and kernel tests passed.
- `pm-coder`: 30 real BashMachine tests passed.

The original audit used `.venv312`. The current delivery setup uses the single `.venv` and GitHub-installed dependencies.
The W1 and W2 delivery records identify the model results and the unchanged legacy cache.

## Remaining integration gap

The manifest template in the handoff omitted `on_invalid` for role phases. The
current kernel requires it. Every manifest here adds
`on_invalid: {action: stop}`. The audit records other design corrections in
`AUDIT.md`. Virtual targets currently work through Python roles. Agent-driver
integration remains a future step.

## Next step

The next development work follows [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
Some workflows use direct model calls through Python roles. Others use tool-capable agents.
The plan retains the shared execution path and moves cognitive behavior into separate workflow folders.

Read [AUDIT.md](AUDIT.md) for the design audit and remaining prototype limits.
