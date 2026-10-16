# Development loop

## Summary

Start with a saved exchange and a specific behavior change.
Modify the owning workflow or subsystem.
Run its contracts, inspect a real exchange, and record the result.
Update the documentation with the changed behavior.

Parent: [manual](../index.md).
Planning: [next experiments](../work_items/experiments.md).
Alpha delivery: [baseline commands](../work_items/alpha-plan/baseline/commands.md) and [session handoff](../work_items/alpha-plan/baseline/handoff.md).
Optional decisions: [client boundary](../subsystems/decisions/index.md), [live procedure](../subsystems/decisions/operation.md), and [A02 results](../work_items/verification/a02-decisions.md).
Companion quality: [A18B commands and reports](../subsystems/cognitive-evaluation.md) and [scenario authoring](../subsystems/cognitive-evaluation-authoring.md).

## Read the session instructions

1. Read `AGENTS.md` in the repository root and the [work-item process](../work_items/process.md).
2. Read the selected alpha work item and its linked design.
3. Open the simple-english and project-documentation skills before writing explanatory prose.

The user requires real behavioral acceptance and detailed documentation for the delivered implementation.
The root instructions record the actual services, data, world, and maintenance requirements.
The user supplies a separate session handoff with recent results and outstanding work.

## Prepare the environment

1. Install Python 3.12, Git, and `uv`.
2. Run the locked dependency installation.

```powershell
uv sync --locked
```

3. Select `config.alpha.example.yaml` for the user's local chat and embedding services.

`config.local.yaml` or `PM_MODEL_CONFIG` can select another verified complete configuration.

Read the [model guide](../models.md) for connection details.
On Windows, real folder workflows need Git Bash on `PATH`.

```powershell
$env:PATH = 'C:\Program Files\Git\bin;' + $env:PATH
```

## Change and inspect

1. Deploy a separate console instance for the experiment.
2. Capture a baseline exchange.
3. Identify the owning workflow through its trace event.
4. Change its prompt, schema, or Python step.
5. Stop and restart the instance to load changed Python code.
6. Repeat the input and compare its trace.

An existing checkpoint overrides a changed seed.
Deploy a new instance to start from another persona seed.
Use the [memory editor](memory.md) for changes to an existing instance.

## Foundation tests

Run only the root foundation modules to exclude live endpoints.

```powershell
uv run --locked python -m unittest discover -s tests/unit -t . -b
uv run --locked python -m unittest discover -s tests/contract -t . -b
```

Run the owning stage's direct contracts after a stage change.

```powershell
uv run --locked python -m unittest tests.contract.main_cognition.stages.test_attention_contract -v -b
```

`unittest discover -s tests` includes live suites.
Live suites need their configured endpoints.
Read [evidence and quality](../concepts/evidence.md) before you interpret the results.

## Live application smoke

```powershell
$env:PM_MODEL_CONFIG = 'config.alpha.example.yaml'
uv run --locked python -m unittest tests.live.test_real_app_path tests.live.test_models -v -b
Remove-Item Env:PM_MODEL_CONFIG
```

The full conversation gate uses `tests.live.test_conversations`.
The primary companion seed is `test_data/personas/spamuel/canon.yaml`, including the real Spamuel/Jesse history.
Tiny invented seeds and fixed model responses cannot establish companion acceptance.
Existing deterministic foundation tests remain useful for their stated contracts.

The real Minecraft server is `C:/source/minecraft-server-real`.
The historical generic W8 autostart path uses a superflat TDD template.
Use the [A01 real-server wrapper](../work_items/alpha-plan/baseline/commands.md#real-minecraft-world-and-reset) for acceptance on the user's actual world.
If the actual server cannot be identified, ask the user for its location.
Do not substitute dummy tests or the superflat template for a required real-world gate.

These gates require actual services and model responses.
They retain evidence under `.runtime/`.

## Write maintainable code prose

1. Use simple-english for new or revised comments and docstrings.
2. Explain state ownership and the reason for important ordering constraints beside the relevant code.
3. Document commit boundaries, external effects, cancellation, retries, and recovery from an uncertain outcome.
4. Describe public API inputs, results, expected exceptions, side effects, units, and limits in docstrings.
5. Update explanatory comments when behavior changes.

Comments must explain facts that a human needs to modify the implementation safely.
A comment that repeats an assignment does not explain an invariant or a design decision.
Precise type annotations and source-generated API references complement that explanation.

## Document the real implementation

Every delivered mechanism needs a detailed explanation of its actual execution path and operational behavior.
Small prototype summaries are insufficient once a real implementation exists.
Split complex mechanisms into focused child pages and connect them through navigation and related links.

Each mechanism's documentation must cover:

- Purpose, callers, entry points, and ownership.
- Concrete input and output contracts, schemas, accepted values, and representative real data.
- Stage order, decisions, routing, input selection, and fallback behavior.
- State revisions, persistence, commit order, publication, and memory provenance.
- External effects, cancellation, retries, interruption, restart, and uncertain action outcomes.
- Configuration, actual endpoints, activation, model and tool dependencies, and measured limits.
- Evidence files, console inspection, trace fields, and worked examples from observed runs.
- Failure symptoms, diagnostic procedures, recovery commands, and remaining defects.
- Reproducible acceptance commands, actual results, and the limits of each result.
- Links from concepts to the relevant source-generated Python API objects.

Active behavior, verified evidence, planned changes, and unknown behavior must remain distinguishable.
The failure path needs the same operational detail as the successful path.
Update affected workflow guides, subsystem pages, API docstrings, and navigation with the implementation change.
Generated API references describe the code contract and must not replace the conceptual explanation.

## Build the documentation

1. Build the complete site with strict validation after each delivery.

```powershell
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

2. Open `site/index.html`.
3. Inspect the changed guide and its generated API links.
4. Fix unresolved links and stale explanations before reporting delivery.

Use `uv run --locked --with-requirements requirements-docs.txt mkdocs serve` for automatic page reloads.
Generated HTML stays outside version control.
The API generator reads public source modules during each build.
Each completed item must regenerate the HTML, even when the visible change is a code comment or API docstring.

## Debugger entry points

Use `.venv/Scripts/python.exe` as the interpreter.
Place a breakpoint in [`pm.host.application.BusModule.step`][] for mailbox processing.
Place a breakpoint in [`pm.runs.workflow_runner.WorkflowRunner.invoke`][] for workflow execution.
Place a breakpoint in [`pm.memory.store.MemoryStore.commit`][] for accepted memory.

Read [workflow runtime](../runtime.md) for direct calls without module threads.
