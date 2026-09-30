# pm-next-v2 — Spamuel companion

This project builds a runnable Spamuel companion on a real in-memory Ghost, a
real virtual Bash machine, and the real pm-workflows kernel. The terminal uses
the real cognitive path by default: seven ordered child workflows, a committed
GWT, typed action dispatches, and a real Story reply. A labeled dummy fixture
remains available for the scripted demonstration and direct module tests.

`main.py` prints `REAL COGNITION — model activates on first request` and
contacts the configured chat endpoint on first generation. `demo.py` prints
`DUMMY COGNITION — no LLM calls` and keeps the fixed keyword policy.
The W1 model tests use real generation and embedding endpoints.

## Next implementation plan

The current code implements the plan's W1 through W5 checkpoints.
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) specifies the next real companion implementation.
It separates wiring, runtime contracts, models and memory, workflow behavior, tests, and ordered work items.
The plan supersedes the old dummy-only handoff.

W1 is complete: model presets, explicit activation, direct chat calls, and compatible embedding cache access.
Read the [model guide](docs/models.md) for setup and debugger examples.
The [W1 delivery record](docs/work-items/W1.md) contains commands, actual results, and implementation limits.
W2 adds isolated bus delivery, scoped read views, atomic Ghost batches, and checkpoint recovery.
Read the [Ghost and bus guide](docs/ghost.md) for the persistence boundary and direct debugger examples.
The [W2 delivery record](docs/work-items/W2.md) contains its evidence and limits.
W3 is complete: the full workflow-to-agent boundary.
Application version 0.9.0 installs pm-coder 0.9.2 and pm-workflows 0.5.0.
Real Python and pm-coder roles run on folder or virtual Ghost targets with explicit model settings and cancellation.
Runs have scoped Python tool grants, optional child namespaces, temporary loopback MCP services, Ghost proposal tools, separate run cancellation, and waiting-input resume.
Read the [runtime guide](docs/runtime.md) and the [W3 delivery record](docs/work-items/W3.md) for the contracts, evidence, and limits.
W4 is complete: standalone Story dialogue.
It builds the exact five-message Story sequence, continues the assistant prefill, cleans the continuation to one companion utterance, applies a private-ontology boundary, and runs a separate evaluator phase over saved quality trials.
It is a separate direct command, not part of the terminal session.
Read the [W4 delivery record](docs/work-items/W4.md) for the commands, evidence, and limits.
W5 is complete: the seven real cognitive children and the typed parent pipeline are wired to the bus, commit the GWT, dispatch typed actions, and return a real correlated Story reply.
The terminal uses the real path by default. The fixed keyword policy remains only as a labeled demonstration fixture.
Read the [W5 delivery record](docs/work-items/W5.md) for the exact scope, evidence, and remaining work.
W6 is complete: Assistant dialogue, management, and stewardship.
It adds a separate tool-capable Assistant conversation, one durable managed-run table with shared start/status/update/terminate operations exposed as scoped MCP tools and SelfManagement, a dormant steward with code-only termination, a small real local task used to exercise concurrency and blocking questions, and evidence-backed report-to-knoxel consolidation.
Read the [W6 delivery record](docs/work-items/W6.md) for the exact scope, evidence, and remaining work.
W7 is complete: first-delivery acceptance.
It adds nine topology checks over every real workflow manifest and a ten-scenario, twenty-exchange conversation acceptance with a separate evaluator phase, nine good trajectories of ten, and no hard contract violation.
Read the [W7 delivery record](docs/work-items/W7.md) for the exact scope, evidence, and remaining work.
The next unfinished work item is W8, Minecraft body and richer consolidation.

## What it does

One process runs module threads over an in-memory bus:

```
UserInputProxy -> bus -> MainCogLoop -> action intents
                            |
        +-------------------+-------------------+
        v                   v                   v
  Story (dialogue)   Assistant          SelfManagement
        |                   |                   |
   dialogue.reply     dialogue.reply      managed runs
                                                |
                                          workflow.event
                                                v
                                          MainCogLoop
```

Every user input enters `MainCogLoop` first. The terminal uses the real path by
default: seven ordered child workflows, a committed GWT, typed action
dispatches, and a real correlated Story reply. `dialogue_assistant` offers a
separate tool-capable conversation whose management claims require an executed
tool receipt. `self_management` starts, reads, updates, and terminates durable
managed workflow runs. A dormant steward can read committed progress and
submit data or control requests. Terminal managed runs consolidate their real
text artifacts into evidence-backed memory. A labeled dummy fixture remains
only for the scripted demonstration and direct module tests.

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
The terminal uses real cognition by default and reads the model config from `PM_MODEL_CONFIG`, then `config.local.yaml`, then `config.example.yaml`.
The first generation activates the model. `--cognition dummy` selects the labeled fixed-output fixture, and `--config PATH` selects another model config.

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

## Standalone Story dialogue (W4)

The terminal reaches Story dialogue through the real cognitive path: an action
targeting `dialogue_story` starts the Story module, and its accepted reply
returns as `dialogue.reply`. W4 still also provides a separate direct command
that runs the fixed Spamuel situation with no module threads and no bus:

```powershell
uv run --locked python -m workflows.dialogue_story.run show
uv run --locked python -m workflows.dialogue_story.run accept
uv run --locked python -m workflows.dialogue_story.run trials --trials 10
```

- `show` prints the exact five messages without contacting a model.
- `accept` runs once, applies the ontology boundary, and commits an accepted reply to `.runtime/w4/<run>/ghost.json`.
- `trials` runs ten independent generations from the same initial Ghost state, then activates the evaluator over the saved artifacts and reports the quality gate.

Every run saves its request, messages, report, and kernel summary under `.runtime/w4/`.
Generation always finishes before the evaluator starts. The evaluator reads saved artifacts and never rewrites a candidate.
No generation deadline exists; a missing endpoint fails the command instead of retrying forever.

## Main cognitive cycle (W5)

The real cognitive pipeline is implemented under `workflows/main_cognition/`.
It runs the seven ordered children — triage, appraisal, context, attention, goals, workspace, and action selection — over one request and one shared proposal batch, then releases typed action dispatches with dependency receipts.

Each child is a normal Python-role workflow with its own manifest, prompts, schema, and direct tests beside the workflow.
The parent assigns `action-<index>` ids and rejects dependent actions whose dependency was not released.
The pipeline is connected to the bus. `MainCogLoop` commits one attention frame, one GWT turn, and one action schema per released dispatch, publishes `action.request`, and `StoryDialogue` returns one correlated public reply. A rejected dependency publishes an explicit receipt and no action.

The live smoke check runs all seven children against the real Qwen backend:

```powershell
uv run --locked python -m unittest tests.live.test_cognition_pipeline -v -b
```

Direct tests need no model, module thread, or bus:

```powershell
uv run --locked python -m unittest tests.test_main_cognition_contract -v -b
```

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

The terminal uses real cognition by default. Pass `--cognition dummy` (or run
`demo.py`) for the labeled fixed-output fixture below.

| Input | Fixed dummy output |
| --- | --- |
| any text | `I heard you, Jesse.` |
| contains `minecraft` | schedules, then `I will schedule your Minecraft request.` |
| `dumb robot` | `You are a dumb human, Jesse! Dumb and nasty.` |
| `/demo-task` | schedules a dummy task, then `I will schedule a dummy task.` |
| `/memory` | one tool-plus-dialogue reply with a scoped search count |
| `/status <id>` | a status reply for the given task id |

These fixed outputs are the labeled dummy fixture, not real character policy.

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
modules.py              real and dummy cognitive subclasses and TaskTable
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
  cognitive_model.py    lazy first-use model activation for the real path
  shell.py              common module step, thread lifecycle, application owner
  workflow_runner.py    shared synchronous kernel invocation and typed outcomes
  mcp_services.py       temporary loopback MCP listeners for one run
ghost.py                compatibility import for the Ghost service
workflows/dialogue_story/  Story dialogue (manifest, prompts, steps, context, engine, command, module tools, tests)
workflows/main_cognition/   ordered cognitive children, parent pipeline, typed dispatch, module tools, tests
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

Version 0.4.0 bound pm-coder roles to explicit targets, model connections, and
cancellation. Version 0.5.0 scoped role `tools`, `mcp_servers` descriptors, and
`child_factory(scope)` namespaces, and mapped pm-coder's permanent provider
rejection to provider exhaustion.

The existing real-folder workflow path still works without a runtime.

The audit also adds two compatible options, published in pm-coder 0.7.0:
`BashMachine.write_texts()` registers a lazy projection in one batch.
`exec(..., strip_output=False)` preserves file whitespace.
The prototype uses both through its working target.

## Verification

- `python demo.py`: 21 checks passed with Git Bash first on PATH.
- unittest discovery: 192 tests passed in 971 seconds, including the full Spamuel seed, threaded integration, Story dialogue, the seven-stage cognition cycle, the real app path, the attention counterexamples, and the live model and agent checks. Log: `.runtime/w5-unittest-final.log`.
- W1 foundation checks: 24 tests passed without model inference.
- W2 checks: 29 tests passed for bus isolation, commits, restoration, concurrency, and direct persistent workflows.
- W3 focused checks: pm-workflows 133 tests, pm-coder 72 tests (including 2 live), and the application live agent checks passed.
- W4 Story dialogue: 15 direct contract tests, one real accepted dialogue, and a 9/10 good quality gate over ten independent trials.
- W5: 43 direct cognition tests, 8 direct wiring tests, one real seven-stage cycle, one real app-path exchange with a committed GWT and correlated reply, and three live attention counterexamples.
- `pm-workflows`: 133 runtime-context, driver, kernel, and scope tests passed.
- `pm-coder`: 72 focused tests passed, including the live session and virtual-permission checks.

Live agent checks include real Qwen roles on both target types and one role calling a dynamically started loopback Ghost MCP service.
The original audit used `.venv312`. The current delivery setup uses the single `.venv` and GitHub-installed dependencies.
The W1, W2, and W3 delivery records identify the model results and the unchanged legacy cache.

## Remaining integration gap

The manifest template in the handoff omitted `on_invalid` for role phases. The
current kernel requires it. Every manifest here adds
`on_invalid: {action: stop}`. The audit records other design corrections in
`AUDIT.md`. Virtual targets work through Python roles and real pm-coder roles.
The W3 acceptance gate passed.
W4 Story dialogue also runs as a direct command.
W5 wires the real cognitive children, the GWT commit, typed dispatch, and the real Story reply into the terminal path.
`demo.py` and `main.py --cognition dummy` keep the labeled fixed keyword policy.

## Next step

The next development work follows [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
Some workflows use direct model calls through Python roles. Others use tool-capable agents.
The plan retains the shared execution path and moves cognitive behavior into separate workflow folders.

Read [AUDIT.md](AUDIT.md) for the design audit and remaining prototype limits.
