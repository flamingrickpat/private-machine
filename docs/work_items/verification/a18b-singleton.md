# A18B singleton admission verification

## Summary

The repair adds an explicit singleton policy to managed workflows.
Minecraft's direct and project workflows share one admission key.
A busy request returns the existing owner to cognition without another worker or automatic cancellation.
The earlier failed run remains unchanged.
Complete A18B acceptance and human review remain pending.

Parent: [A18B verification](a18b-cognitive-evaluation.md).
Mechanism: [singleton managed workflows](../../subsystems/managed-singletons.md).
Original failure: [overlapping projects](a18b-errors/body-conflict.md).

## Evidence scope

`test_data/captures/alpha/minecraft-overlapping-starts.json` projects three actual start requests from the failed case.
It records the original runtime revision and case name.
Boundary checks replay these requests through real local kernel roles, threads, checkpoints, and operating-system locks.
Those local roles establish admission behavior. They do not establish Minecraft execution.

The separate real-service gate uses the complete 11,202-record Spamuel seed and the real server.
An explicit narrow body workflow obtains a real observation and retains its lease while it awaits an update.
A competing project request must return the existing owner without launching a worker.
One real main-cognition cycle consumes that busy event through OpenRouter.
Its selected actions remain available for human inspection and are not scored against a forced control choice.
The gate then explicitly terminates the held run, waits for cleanup, and verifies a new real observation workflow.
It does not reset the world, substitute fake responses, or establish general gameplay acceptance.

## Commands

```powershell
uv run --locked python -m unittest tests.unit.pm.runs.test_singleton -b
uv run --locked python -m unittest discover -s tests/unit -t . -b
uv run --locked python -m unittest discover -s tests/contract -t . -b
$env:PM_SINGLETON_LIVE = '1'
uv run --locked python -m unittest tests.live.test_workflow_singleton -v
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

Current verification receipts belong under `.runtime/a18b-current/singleton/`.
Each real gate creates its own `live-<timestamp>-<id>/` directory.
It retains a source archive, effective configuration identity, actual world observations, busy event, cognitive output, and terminal receipt.
The test restores environment values and stops the server and MCP processes that it owns.
Failed attempts remain available beside later successful evidence.

## Recorded check status

The original live gate stopped at the owner's request on 16 October 2026.
Its partial observations have no terminal pass receipt.
The cancellation record remains `.runtime/a18b-current/singleton/cancelled.json`.
The later A18C foundation run includes this repair and passed 896 unit tests and 83 contract tests.
Its logs remain `.runtime/a18c/unit-resume-03-source-final.log` and `.runtime/a18c/contract-resume-03-source-final.log`.
Those checks do not replace the unfinished real-server singleton gate.

## Remaining boundaries

Singleton admission is local to one supervisor.
The global body lease still arbitrates other processes and uncertain external calls.
Rejected requests remain failed operation receipts, rather than fabricated task completions.
The companion decides whether to wait, update, or terminate through self-management.
The repair does not force speech after a deliberately quiet choice.
Provider rejection, reviewer quality, and complete conversation acceptance remain separate work.
