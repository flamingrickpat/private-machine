# Optional typed decisions

## Summary

A02 adds a public System One client to the existing pm-coder distribution.
PM installs the client but does not route existing workflows through it.
Callers can request choice, score, probability, or several independent decisions.
Receipts preserve inputs by digest and retain validated provider answers.

Parent: [model connections](../../models.md).
Delivery: [A02](../../decisions/alpha-plan/work-items/a02.md).
Evidence: [A02 verification](verification.md).
Measured behavior: [quality and limits](quality.md).
Operation: [local deployment](operation.md).

## Owning boundary

The implementation is `pm_decision.py` in `C:/source/pm/pm-coder`.
Version 0.10.0 installs this module beside the existing coding agent modules.
The client imports HTTPX and Pydantic directly.
It imports no Ghost, application state, agent loop, or workflow engine.
The application and pm-workflows locks select the tested distribution revision.

[`pm_decision.DecisionClient`][] provides synchronous calls.
[`pm_decision.AsyncDecisionClient`][] provides asynchronous calls.
Their `decide` methods accept concrete JSON state and typed questions.
Their convenience methods return specifically typed choice, score, and probability results.
[`pm_decision.DecisionResult`][] exposes complete answers and one [`pm_decision.DecisionReceipt`][].

## Request, validation, and result

The client snapshots state and question definitions before dispatch.
It sends one request to the separately configured `/v1/systemone` endpoint.
It requires the returned model alias and question IDs to match the request.
It validates every answer before returning any result.
Numeric validation covers finite probabilities, distributions, confidence, score range, and the weighted score.
No automatic retry or fallback runs inside this boundary.

Choice allows two to 255 options.
Score allows two to ten ordered levels.
Probability preserves the wire primitive `noul` as a float in zero to one.
Each request allows one to 64 questions and at most 1,048,576 encoded bytes by default.
These byte and count limits do not establish the model's reliable token context.

## Ownership and cancellation

The client creates no durable state and writes no evidence files.
The caller owns receipt storage, publication, fallback, and later state commits.
State digests cover compact sorted-key UTF-8 JSON.
Request digests cover the exact transmitted bytes.
Receipts exclude source state and authorization credentials.
Question descriptions and provider answers can still contain private information.

Each request owns its HTTP connection.
Async task cancellation or an explicit event closes that connection and joins local tasks.
Synchronous calls can use `threading.Event` cancellation.
[`pm_decision.DecisionCancelled`][] preserves `asyncio.CancelledError` semantics and attaches a receipt.
The receipt leaves server completion unknown.
The client does not claim that closing a socket stopped inference.

## Uncertainty and adoption

Protocol success does not establish correct interpretation or calibrated confidence.
The real A02 results include incorrect score and probability decisions.
Existing LLM workflows remain usable without Laya.
The [model policy](../../decisions/alpha-plan/model-policy.md) uses the normal LLM by default.
Only the simplest Laya classifications can qualify through realistic live-endpoint unit tests.
If qualification requires expensive effort, retain the client for later and keep the LLM.
Later experiments can retain the technology without adopting it in any workflow.
A suitable future use requires application-specific facts, limits, and a declared LLM fallback.

The [pm-coder client manual](https://github.com/flamingrickpat/pm-coder/blob/main/docs/index.md) documents detailed contracts and recovery.
The [generated API](../../reference/pm_decision.md) exposes exact installed objects and source.
