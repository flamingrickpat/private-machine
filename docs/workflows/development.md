# Development loop

## Summary

Start with a saved exchange and a specific behavior change.
Modify the owning workflow or subsystem.
Run its contracts, inspect a real exchange, and record the result.
Update the documentation with the changed behavior.

Parent: [manual](../index.md).
Planning: [next experiments](../decisions/experiments.md).

## Prepare the environment

1. Install Python 3.12, Git, and `uv`.
2. Run the locked dependency installation.

```powershell
uv sync --locked
```

3. Configure the model endpoint through `config.local.yaml` or `PM_MODEL_CONFIG`.

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
$testModules = Get-ChildItem tests/test_*.py | ForEach-Object { 'tests.' + $_.BaseName }
uv run --locked python -m unittest $testModules -b
```

Run the owning stage's direct contracts after a stage change.

```powershell
uv run --locked python -m unittest workflows.main_cognition.stages.attention.tests.test_attention_contract -v -b
```

`unittest discover -s tests` includes live suites.
Live suites need their configured endpoints.
Read [evidence and quality](../concepts/evidence.md) before you interpret the results.

## Live application smoke

```powershell
uv run --locked python -m unittest tests.live.test_real_app_path tests.live.test_models -v -b
```

The full conversation gate uses `tests.live.test_conversations`.
The real Minecraft gate uses `tests.live.test_minecraft_body`.
These gates can start or require external services.
They retain evidence under `.runtime/`.

## Build the documentation

1. Install the documentation dependencies into the project environment.

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-docs.txt
```

2. Build the site with strict validation.

```powershell
.venv/Scripts/python.exe -m mkdocs build --strict
```

3. Open `site/index.html`.

Use `.venv/Scripts/python.exe -m mkdocs serve` for automatic page reloads.
Generated HTML stays outside version control.
The API generator reads public source modules during each build.

## Debugger entry points

Use `.venv/Scripts/python.exe` as the interpreter.
Place a breakpoint in [`runtime.shell.ShellModule.step`][] for mailbox processing.
Place a breakpoint in [`runtime.workflow_runner.WorkflowRunner.invoke`][] for workflow execution.
Place a breakpoint in [`pm_core.ghost.service.Ghost.commit`][] for accepted memory.

Read [workflow runtime](../runtime.md) for direct calls without module threads.

