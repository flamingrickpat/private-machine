# PM classification connections

## Summary

A05 connects normal chat presets to A04 classification through PM.
Activation occurs at inference, rather than application construction.
The service lease covers direct classification or the complete isolated recovery session.
Existing Python roles and cognitive chat tools keep their execution paths.
Optional Laya wiring remains deferred.

Parent: [classification](classification.md).
Operation: [PM demonstration](../../workflows/classification-demo.md).
Evidence: [A05 verification](../../work_items/verification/a05-pm-classification.md).

## Owners and injection

PM owns [`pm.llm.workflow_sessions.WorkflowModels`][].
[`workflows.companion.build_app`][] constructs it beside the normal lazy stage model.
Both objects share one [`pm.llm.activation.ModelActivation`][] owner.
Construction resolves credentials once and performs no network request.
The lazy cognitive model reuses that owner's ready alias, including during an active classification lease.
It rechecks ownership before each client use after an application-owned model switch.

Shell modules pass this provider to [`pm.runs.workflow_runner.WorkflowRunner`][].
[`pm.runs.supervisor.ManagedRunSupervisor`][] supplies it to managed workers.
The synchronous runner also accepts `models` directly for debugger use.
An injected custom stage model needs an explicit `workflow_models` provider when its workflows use classification.
Other custom stage models need no additional configuration.

The runner binds an invocation copy of [`pm_workflows.runtime.WorkflowRuntime`][].
It installs `classification_session` without changing the caller's runtime.
The workflow package calls this callback around actual inference.
An explicit `classification_models` entry takes precedence for direct calls.
An explicit `agent_model` takes precedence for recovery.
Existing explicit sessions also take precedence over PM binding.

The kernel requests a named session before inference.
The provider returns an `AgentModel` under an active lease.
The kernel exits that session after completion, cancellation, or an error.

## Connection and context contract

The manifest's `connection` names a configured chat preset, such as `Default` or `UnitTestEvaluator`.
Unknown names produce `missing_connection`; PM does not substitute another classifier.
Recovery uses `Default` unless the provider receives another `recovery_preset`.
The normal LLM therefore remains available independently of optional decisions.

The resolved [`pm.llm.config.Connection`][] retains URL, alias, credentials, startup script, and optional `context_window`.
A nonempty script follows the existing finite-script activation contract.
The owner checks the advertised alias before admitting a model group.
An active lease prevents an application-owned model switch.
It does not control unrelated external server processes.

[`pm.llm.workflow_sessions.serving_context`][] uses an explicit positive integer `context_window` when supplied.
Otherwise, it reads llama.cpp's `/props` and uses `default_generation_settings.n_ctx`.
Training context does not supply a serving bound.
Remote services without this route require an explicit bound in each affected chat preset:

```yaml
chat:
  Default:
    base_url: https://your-chat-service.example/v1
    model: your-model-alias
    api_key_env: YOUR_CHAT_KEY
    startup_script: ""
    context_window: 32768
```

Keep the existing evaluator and embedding entries in the complete configuration.
The bound must describe the deployed service.
Booleans, strings, fractions, zero, and negative values fail configuration validation.
This field affects workflow classification and recovery; it does not alter ordinary chat generation.

## Lifetime, cancellation, and failures

Each session obtains readiness and service ownership before context discovery.
The provider serializes readiness acquisition but permits concurrent calls after lease acquisition.
Waiting for that lock checks cancellation every 0.1 seconds.
Direct inference releases ownership after each selection or classification request.
Recovery retains ownership through all agent calls, summaries, and cleanup.

Cancellation closes readiness I/O or follows the kernel's existing model cancellation path.
Cancelled work publishes no accepted classification.
Readiness errors become `endpoint_unavailable` with a redacted reason.
Unknown configuration becomes `missing_connection`.
Unexpected Python errors retain their traceback.
The provider releases active ownership on all session exits.

The workflow callback selects pm-coder only for the dedicated recovery operation.
An ordinary Python role retains its driver.
The parent environment, tools, and runtime connection values remain unchanged.
Recovery still uses A04's read-only snapshots and producer/item guards.
It sends no Ghost proposal and dispatches no action.

`live_test` on the provider makes readiness errors immediate in acceptance runs.
The supplied agent model keeps `live_test=False`, so pm-coder retains automatic context recovery.
Neither pm-coder's implementation nor its context mechanism changes.
Direct [`pm.llm.activation.ChatClient.complete`][] still sends one supplied message list.

## Service evidence

The runner retains `kernel/<run-id>/model-sessions.jsonl` beside the journal.
No classification call means no service log and no model request from this provider.
The callback records `pm.workflow-model-session.v1` events:

| Event | Evidence |
| --- | --- |
| `ready` | Requested and selected preset, recovery flag, endpoint, alias, context, context source, activation reuse, script use, readiness milliseconds |
| `failed` | Concrete configuration or readiness cause and redacted reason |
| `cancelled` | Cancellation during session preparation or execution |
| `released` | Completed lease cleanup and elapsed milliseconds |

Credentials never enter these events.
`time` records the session start in Unix seconds; readiness and elapsed durations use milliseconds.
Children inherit the callback unless their runtime factory supplies another one.
Their service events use the parent invocation log; child journals retain their separate classification scope.
Resume appends events to the retained log and checks readiness through the current owner.
The log establishes connection use, rather than classification correctness or task success.

Go deeper: [answer validation](classification.md#answer-and-routing), [context recovery](classification-recovery.md), and [PM diagnosis](pm-failures.md).
