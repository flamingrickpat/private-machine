# Operate the local decision service

## Summary

The client uses Laya on port 8082.
Chat uses 8080 and embeddings use OpenRouter.
A02 does not add a decision entry to PM's model activation configuration.
The live runner verifies the service and tests an independently owned failure process.

Parent: [optional typed decisions](index.md).
Measured results: [A02 verification](../../work_items/verification/a02-decisions.md).
Model limitations: [quality observations](quality.md).
Historical deployment: [A01 commands](../../work_items/alpha-plan/baseline/commands.md).

## Verified deployment

| Property | Value |
| --- | --- |
| Shared endpoint | `http://127.0.0.1:8082/v1/systemone` |
| Returned alias | `laya` |
| Binary | `C:/workspace/llama-systemone-b11374/llama-server.exe` |
| Build | `b11374-b92761a51` |
| Model | `C:/source/ai/Laya-BF16.gguf` |
| Active device | CUDA1 |
| Context, batch, microbatch | 8192 tokens each |
| Parallel slots | One |
| Owned fault service | CPU, `127.0.0.1:8084`, same binary and model |

Inspect `/props` and `/v1/models` before inference.
Compare the build, model path, context, and alias with the table.
The receipt's `provider_identity` records independently observed provenance.
The client itself validates only the returned alias.
The tested server ignores a different `model` field in the request body.

The [pm-coder operation guide](https://github.com/flamingrickpat/pm-coder/blob/main/docs/decisions/operation.md) supplies the hidden Windows startup command.
Use that command only when port 8082 is free.
Do not place another decision process on the console smoke's port 8093.
The earlier chat binary in `C:/workspace/llama13` does not expose System One on port 8080.

## Public Python use

```python
from pathlib import Path
from pm_decision import DecisionClient

report_path = Path(
    ".runtime/alpha/a01/console-spamuel/instances/"
    "fd71e4655da44430b9779eb6739d45f5/managed_tasks/"
    "dd8a80caec16462b97e0e85cbdf87346/outbox/report.yaml"
)
result = DecisionClient().choice(
    report_path.read_text(encoding="utf-8"), "What is the task status?",
    {"COMPLETE": "The task completed.", "FAILED": "The task failed.",
     "INVALID": "The report cannot establish a result."},
)
print(result.answer.choice)
print(result.receipt.to_dict())
```

This input is a captured Spamuel application report.
The example's question and classes are generated demonstration definitions.
Their outputs are not reviewed workflow routes.
No holdout labels enter this demonstration.

## Repeat the live procedure

```powershell
uv sync --locked
uv run --locked python dev/work_items/alpha_decision_gate.py --out .runtime/alpha/a02-repeat/decisions
```

The output directory must be new.
The runner reads all 11,202 records from the unchanged real seed.
It selects exact history prefixes for measured token budgets.
It reads the completed A01 task report and its checkpoint run state.
The `--report`, `--checkpoint`, and `--seed` arguments select different real evidence when necessary.

The runner stores provenance, layouts, successful answers, failure receipts, and service identities.
It measures three warm chat calls per contention condition.
It starts one owned CPU service on port 8084 after verifying that the port is free.
Active-slot observations precede task cancellation, synchronous event cancellation, and process termination.
Its `finally` block stops the owned process and retains the log.
Shared services remain outside its process ownership.

## Diagnose errors

If the client raises `DecisionHTTPError`, inspect the status and server log.
The 8080 deployment returned 404 for the decision route.
Laya returned 400 for empty questions and an input beyond the configured context.
If the client raises `DecisionSchemaError`, compare the expected alias and complete answer map.
If transport fails, inspect the receipt's concrete HTTPX error and service health.
The client makes no automatic retry.

Malformed local input has no dispatched-request receipt.
Question constructors raise Pydantic `ValidationError` for invalid typed definitions.
Caller cancellation raises `DecisionCancelled`, which ordinary `except Exception` does not catch.
Store its receipt explicitly when the application requires cancellation evidence.

The model's head budget silently truncates long question and option text.
The client byte limit cannot detect that model-specific loss.
Keep descriptions short and evaluate actual application inputs before adoption.
