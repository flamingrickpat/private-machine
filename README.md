# pm-next-v2 ? Spamuel companion

Spamuel runs on a shared Ghost, an in-memory bus, and the `pm-workflows` kernel.
The terminal uses real cognition by default. The model activates on the first request.

Each input passes through seven stages: triage, appraisal, context, attention, goals, workspace, and action selection.
The parent commits an attention frame and a global workspace turn (GWT), then dispatches actions.
Story dialogue produces a character utterance. Assistant dialogue can use scoped memory and workflow tools.

Managed workflows support start, status, instruction updates, questions, and cancellation.
The application stores their receipts in Ghost and consolidates their artifacts into memory.
Minecraft supports sustained specialist leaves and durable projects through the real Minecraft MCP.
The project planner orders subtasks and independently verifies their world effects.
Internal thought generation remains disabled in real mode.

## Setup and run

The [engineering manual](docs/index.md) explains operation, architecture, memory experiments, and the development loop.
For a personal Minecraft test, read [Your own test Spamuel: local LLM and Jesse chat](docs/workflows/jesse-quickstart.md).
Start the local operator console with `uv run --locked python -m console.server`.
Open `http://127.0.0.1:8090` to deploy and control instances, chat, and inspect execution evidence.
Each console instance has a separate checkpoint. Sessions within one instance share memory.
Read the [console procedure](docs/workflows/console.md) for details.

Requirements: Python 3.12, `uv`, Git, and access to the PM dependency repositories.
The Git dependencies follow `main` for `pm-workflows` and `pm-coder`.
Refresh both dependencies during installation to fetch their current branch heads.
The lockfile records the commits resolved by that installation.
Normal startup needs no sibling source checkout.

```powershell
uv sync --upgrade-package pm-workflows --upgrade-package pm-coder
uv run --locked python main.py
```

The model config comes from `--config`, `PM_MODEL_CONFIG`, `config.local.yaml`, or `config.example.yaml`, in that order.
Read the [model guide](docs/models.md) for endpoints, credentials, activation, and the embedding cache.
Keep credentials in the environment or ignored local config files.

The terminal restores `.runtime/ghost.json` when that file exists.
Otherwise, it loads `data/spamuel_agentic_canon.yaml`.
Accepted changes reach the checkpoint before public output.
`--checkpoint PATH` selects another checkpoint. `--ephemeral` disables checkpoint storage.

On Windows, folder workflows need Git Bash:

```powershell
$env:PATH = 'C:\Program Files\Git\bin;' + $env:PATH
```

`--cognition dummy` selects the fixed-output demonstration.
`uv run --locked python dev/smoke_demo.py` runs its finite checks without inference.
Dummy behavior does not establish real companion quality.

## Managed workflows

`local_task` writes a note. `local_task_question` first asks which artifact to produce.
`minecraft` runs one sustained native specialist session and retains original tool evidence.
`minecraft_project` plans durable subtasks, dispatches fresh specialists, and independently verifies each outcome.
Read the [project guide](docs/subsystems/minecraft-tasks.md) for revision rules and evidence.
Cognition can dispatch these targets through SelfManagement.
Assistant tools use the same supervisor operations.

An update first increments `revision`. It does not claim immediate application.
A worker consumes the latest instruction through `managed_instructions` at a declared boundary.
The receipt then advances `applied_revision`.
After the final instruction boundary, the supervisor rejects new updates.
Cancellation wakes a worker that waits for an update.

The local worker can consume updates while held.
Minecraft specialists consume updates before each tool call.
Project instruction changes invalidate the selected leaf and request replanning from retained world effects.
A revised goal invalidates an earlier explicit action.
Completed external effects cannot be undone.

Minecraft reads `MINECRAFT_MCP_URL`, with `http://127.0.0.1:6767/mcp` as the default.
Its planner uses the configured model and the MCP tool descriptions and argument schemas.
Callers can restrict `allowed_tools` or supply an explicit `action_tool` and `action_arguments`.
A plain instruction becomes the planning goal.
A tool's own `ok` result determines success. MCP transport success alone is insufficient.

Consolidation preserves text, spatial records, and screenshot evidence.
Screenshot bytes stay outside ordinary text prompts and embedding requests.
Task folders remain available after consolidation.
The steward remains dormant until explicitly activated.

## Standalone Story dialogue

```powershell
uv run --locked python -m workflows.dialogue_story.run show
uv run --locked python -m workflows.dialogue_story.run accept
uv run --locked python -m workflows.dialogue_story.run trials --trials 10
```

`show` prints the five-message sequence without inference.
`accept` generates and stores an accepted reply.
`trials` generates ten independent samples, then evaluates the saved results.
Artifacts stay under `.runtime/w4/`.

## Tests

Run all foundation tests without model or Minecraft endpoints:

```powershell
$testModules = Get-ChildItem tests/test_*.py | ForEach-Object { 'tests.' + $_.BaseName }
uv run --locked python -m unittest $testModules -b
```

The regression suite checks failure reporting, instruction consumption, action routing, and conversation evidence:

```powershell
uv run --locked python -m unittest tests.test_reconciliation -v -b
```

Run the real application path and model checks:

```powershell
uv run --locked python -m unittest tests.live.test_real_app_path tests.live.test_models -v -b
```

Full conversation acceptance uses ten scenarios with twenty exchanges each:

```powershell
uv run --locked python -m unittest tests.live.test_conversations -v -b
```

Each exchange enters through UserInputProxy and the cognitive loop.
The harness records committed GWT turns, dispatches, replies, and tool receipts.
An error entry does not count as a completed exchange.
Full acceptance requires 200 completed cycles, no infrastructure or contract failures, and at least seven good trajectories.
The evaluator runs after generation and receives execution evidence with the transcript.

For a smaller generation smoke test:

```powershell
$env:PM_W7_SCENARIOS = 'casual_banter,managed_start_and_status'
$env:PM_W7_EXCHANGES = '2'
$env:PM_W7_EVALUATE = '0'
uv run --locked python -m unittest tests.live.test_conversations -v -b
Remove-Item Env:PM_W7_SCENARIOS, Env:PM_W7_EXCHANGES, Env:PM_W7_EVALUATE
```

A smoke test does not establish full conversation acceptance.
The earlier Assistant-only conversation results do not establish acceptance for the current application path.

The Minecraft gate starts its own fresh test server and MCP when no endpoint is running:

```powershell
uv run --locked python -m unittest tests.live.test_minecraft_body -v -b
```

It copies the configured server template into a fresh run directory, starts the server and the local pm-minecraft MCP, and stops both after the run.
Set `PM_MINECRAFT_AUTOSTART=0` to require an external endpoint instead.
`PM_MINECRAFT_REPO` and `PM_MINECRAFT_ENV` select the checkout and its env file.
The test checks a mined block, updated instructions, cancellation, screenshots, and checkpoint recovery.
It retains its artifacts under `.runtime/w8/live-minecraft/`.

`unittest discover -s tests` includes live suites.
The model, embedding, and Minecraft checks start or require their configured endpoints.
A missing model or embedding endpoint fails its suite.

## Current limits

Durable Minecraft projects require real acceptance evidence. Sustained survival and interruption recovery remain separate alpha gates.
Python cancellation is cooperative. A synchronous external call can finish before cancellation takes effect.
Context selection re-ranks supplied candidates; the main cognitive candidate list does not yet use embedding retrieval.
Memory and quality checks cover different contracts. Passing foundation tests does not establish dialogue quality.

## Code and documentation

- [Ghost, bus, and checkpoints](docs/ghost.md)
- [Models and embeddings](docs/models.md)
- [Workflow runtime and debugger examples](docs/runtime.md)
- [Application wiring](main.py)
- [Cognitive modules](modules.py)
- [Managed-run supervisor](runtime/managed_runs.py)
- [Consolidation](runtime/consolidation.py)

Workflow manifests are executable Markdown under `workflows/`.
Historical plans, handoffs, audits, and delivery notes remain in Git history.
The current guides describe supported behavior and verification commands.
