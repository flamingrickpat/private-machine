# PM artifact classification demonstration

## Summary

This opt-in workflow selects a real foundation report, classifies its captured summary, and publishes an inspectable result.
INVALID invokes the existing isolated coding handler once.
The demonstration uses normal chat and works with Laya stopped.
Its separate long-inspection workflow checks actual agent compaction.

Parent: [workflow runtime](../runtime.md).
Connections: [PM model sessions](../subsystems/workflow-inputs/pm-connections.md).
Evidence: [A05 verification](../work_items/verification/a05-pm-classification.md).

## Trigger and results

The manifest is `workflows/classification_demo/classification_demo.workflow.md`.
It is a separate demonstration, rather than a replacement cognitive stage.
The application does not add it to its automatic managed-work catalog.
The caller supplies `artifact_folder`, a granted relative folder under `logs`.
Absolute paths and traversal fail the existing input contract.

The direct path uses folder metadata to select the latest foundation log.
Its UTC filename supplies recency when Ghost has no modification time.
The kernel checks the listing again, captures the selected version, and classifies the current summary.
The demonstration declares `all_good`, `some_failed`, and `all_failed`, plus implicit INVALID.
Missing, incomplete, contradictory, or stale evidence cannot establish a class.

The primary routes accepted classes to `publish` and INVALID to `inspect`.
The dedicated handler routes its observed result to `publish`, including unresolved INVALID.
[`workflows.classification_demo.steps.run`][] reads the latest classification for the same task and item from the kernel journal.
The role session ID differs from the workflow run ID.
The publisher therefore uses retained classification scope and reports the journal's workflow ID.

It writes `outbox/report.yaml` through the supplied environment:

| Field | Meaning |
| --- | --- |
| `schema` | `pm.classification-demo.v1` |
| `run_id`, `phase` | Workflow run and latest classification phase |
| `classification` | One declared class or INVALID |
| `answer` | Validated status, reason, exact quotes, candidate ID, and captured-value digest |
| `route_history` | Primary and handler phases, observed status, and concrete invalid cause |

A completed PM run means report publication completed.
Its report can still contain INVALID.
The classification reports source facts; it does not claim that this workflow executed the tests.

## Direct Python and debugger operation

Use the locked package and local configuration:

```python
from pathlib import Path
from threading import Event
from pm_workflows.runtime import WorkflowRuntime
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config
from pm.llm.workflow_sessions import WorkflowModels
from runtime.workflow_runner import WorkflowRequest, WorkflowRunner
from pm.host.folder_target import FolderEnvironment

models = WorkflowModels(load_config("config.alpha.example.yaml"), ModelActivation())
runtime = WorkflowRuntime(FolderEnvironment(Path("your-test-workspace"), "bash"), Event(), {})
runner = WorkflowRunner(
    Path("workflows/classification_demo/classification_demo.workflow.md"),
    Path(".runtime/your-classification-run"), models=models,
)
result = runner.invoke(WorkflowRequest("your-unique-run", {"artifact_folder": "logs"}), runtime)
```

Prepare actual UTF-8 test logs in the granted folder before invocation.
Use a unique run ID and separate runtime directory.
Breakpoints in the runner, session provider, and publisher stay on the calling thread.
The kernel retains full inference requests and responses in its classification receipts.
The publisher performs no further inference.

## Long-file inspection

`workflows/classification_demo/long_inspection.workflow.md` reads the exact `logs/overflow.log` input.
Its 512-token direct budget deliberately requires INVALID recovery.
Its acceptance skill requests one full virtual `cat`, then focused footer inspection after automatic compaction.
The source contains an original Spamuel history prefix and an actual preserved test footer.
Historical conversation remains explicitly separate from current results.

The gate measures the full tool result with the server tokenizer.
It retains an actual provider context rejection, real summary calls, a checkpoint, and the resumed final answer.
The temporary loopback relay forwards discovery and inference unchanged.
It supplies no fabricated answer or forced HTTP error.
The original question, classes, exact test count, OK evidence, and captured-value digest must survive.

## State and recovery

Classification reads the supplied target and appends its private journal.
The publisher writes only the demonstration report.
It does not accept Ghost state changes or launch Minecraft work.
The caller owns the environment and cancellation signal.
The [recovery contract](../subsystems/workflow-inputs/classification-recovery.md) owns snapshot limits, origin reuse, deadlines, and unsupported evidence rejection.

The same report path can be overwritten by a later demonstration in that environment.
Use separate targets when retained report bytes matter.
Journals and unique run IDs retain the previous results.
Missing journal evidence raises a Python error; the publisher does not invent a result.
Cancellation prevents report loading and public success delivery at the runner boundary.

Run the complete [M1 gate](../work_items/verification/a05-pm-classification.md#reproduce-m1) for independently checked real evidence.
