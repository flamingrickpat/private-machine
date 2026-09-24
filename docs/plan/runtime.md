# Runtime contracts

## Summary

The runtime owns execution mechanics, not cognition.
These contracts apply to direct calls and threaded runs.
They extend the existing implementation instead of replacing the workflow interpreter.

Parent: [implementation handoff](../../IMPLEMENTATION_PLAN.md).
Related: [architecture](architecture.md), [memory](models-memory.md), [tests](tests.md).

## Bus and envelopes

Retain `id`, `source`, `kind`, `topic`, `correlation_id`, `target`, and `payload`.
Add `causation_id` for the immediate triggering message.
`kind` remains `data`, `control`, or `telemetry`.
Payloads contain serialized values and record references, not live mutable Ghost objects.

Each subscriber owns a mailbox.
Publication delivers an independent envelope and payload to every matching subscriber.
Mutation by one consumer cannot change another consumer's message.
Constructing a nested immutable representation is unnecessary. Serialization and independent delivery copies are sufficient.

Bus callbacks only select and enqueue messages or set a cancellation signal.
They never run a workflow or call a model under the bus lock.
Each iteration drains the available batch once.
Messages that arrive during processing belong to the next iteration.
An idle MainCogLoop waits. It does not create unlimited empty GWT records.

The bus provides in-process delivery, not durable replay across process crashes.
No exactly-once claim applies to external tool effects.

Use these initial topics. Keep their ownership explicit in application wiring.

| Topic | Kind | Producer | Consumer |
| --- | --- | --- | --- |
| `user.input` | data | UserInputProxy | MainCogLoop |
| `action.request` | data | MainCogLoop | Target workflow module |
| `action.receipt` | telemetry | Action executor | Managed-run bookkeeping |
| `dialogue.reply` | data | Either dialogue workflow | UserInputProxy |
| `workflow.event` | telemetry | Managed external run | MainCogLoop |
| `workflow.question` | telemetry | Managed external run | MainCogLoop |
| `workflow.answer` | data | Selected response action | Target run's input boundary |
| `workflow.update` | control | Authorized management operation | Target run's update boundary |
| `terminate` | control | Application or management operation | Target run or module |

Dispatch selection matches both topic and target.
Lifecycle receipts update execution state without creating a dialogue-to-dialogue feedback loop.
Only relevant external-run events become new cognitive stimuli.
Dialogue already commits its text before publication, so MainCogLoop does not consume its own public reply as new user input.

## Shared entry points

| Entry point | Contract |
| --- | --- |
| `build_app(...)` | Construct objects and subscriptions without starting threads, servers, or models |
| `ShellModule.accepts(envelope)` | Select relevant input without side effects |
| `ShellModule.step(block=False)` | Run at most one batch or due tick, synchronously |
| `ShellModule.run()` | Start the common thread loop |
| `ShellModuleWorkflow.prepare_input(batch)` | Convert selected messages into a typed workflow request |
| `ShellModuleWorkflow.publish_output(result, batch)` | Convert accepted results into bus messages |
| `WorkflowRunner.invoke(request, runtime)` | Run the real kernel synchronously and return a typed result |

The new `WorkflowRunner` is the extracted shared kernel-invocation function or small stateful object.
Do not put a second interpreter behind this name.
Standalone child tests call the same invocation path with a supplied request.
Direct exceptions reach the debugger. Threaded exceptions reach the application error handler.

## Per-run context

A run receives a run ID, parent run ID, model connection, working environment, cancellation signal, tool bindings, and event sink.
Application objects stay behind the supplied tools and working target.
The generic runtime does not require a Ghost argument.

Retain file-based exchange: `inbox/request.yaml` and `outbox/report.yaml`.
Use Markdown for task descriptions and natural-language answers.
The YAML request carries their paths and correlation metadata.
Each run and child gets a separate writable directory or virtual namespace.
Children must not overwrite a parent's request or report.

Child cancellation inherits parent cancellation.
Stopping one run does not stop unrelated module runs.
Stopping a module also cancels its active run.
Keep module shutdown and individual run cancellation as separate signals.

## Dynamic MCP bindings

Application wiring constructs named scoped services, such as `attention_mcp`.
A workflow role names the services it needs.
The effective service set is the intersection of run grants, parent grants, and role grants.
An unknown or forbidden name produces a config error.
An empty grant never means unrestricted discovery.

Use a loopback listener with an operating-system-assigned port for each temporary HTTP MCP server.
Hold the bound listener throughout its lifetime. Do not find a free port and release it before startup.
Keep servers alive through final post-processing.
Close them after success, error, or cancellation.
Keep connection files private to the run.
Do not replace a shared workspace MCP config during concurrent runs.

The same scoped service functions serve trusted Python and MCP clients.
Ghost capability checks belong in those functions, not only the advertised tool list.
Role access to Python functions must follow the same grant selection.
Trusted Python remains host code. A virtual view is not a security sandbox for arbitrary Python.

## Working target and agent integration

Preserve the public `WorkingTarget` and `WorkingEnvironment` contracts in `pm-workflows`.
Keep the Ghost-specific projection in this application.
A run has a stable visible Ghost snapshot and writable scratch files.
Child environments also see accepted proposals from earlier children in the same cognitive run.

All agent file tools must use the supplied environment.
This includes Bash, read, write, edit, glob, search, and artifact reads.
They must not fall back to host paths for virtual files.
Read-only Ghost files remain read-only through every agent tool.
Python finalizers apply proposals through scoped Ghost operations instead of editing projected YAML.

Kernel journals can remain on disk outside the virtual target.
An agent must not require the full Ghost snapshot on disk.
Bind pm-coder session state per invocation. Remove cross-run reliance on its global `active_session` path.
Preserve existing standalone pm-coder and real-folder behavior.

## Results and failure ownership

Normalize kernel results at one adapter boundary.
Application statuses are `completed`, `failed`, `cancelled`, and `waiting_for_input`.
Preserve raw kernel status and the artifact location for debugging.
Do not treat every `ok=False` as failure: the existing kernel also uses it for an explicit successful terminal stop route.

Code errors propagate with their tracebacks.
Invalid model output follows the workflow's explicit invalid-output route.
It never becomes an invented business category or a successful empty answer.
The initial invalid-output route records the response and stops with an actionable error.
Later workflows can add explicit repair phases.

Production transport retries remain cancellable for transient connection errors.
Authentication errors, exhausted credit, invalid config, and oversized input require a clear error.
The runtime must not impose a generation deadline or silently truncate content.
Live-test mode reports endpoint failures instead of entering an endless reconnect loop.

## Limited steward

The steward is an optional representative of a managed run, not a phase of that run.
It remains dormant until a request or configured status schedule activates it.
Its first tools read committed progress and submit data or control requests.
It cannot edit prompts, code, hidden workflow state, or read-only files.

| Request | Initial behavior |
| --- | --- |
| Status | Read journal and published artifacts, then return a report |
| Answer | Match an outstanding question ID and deliver the answer |
| Change instructions | Queue a revision for the next declared safe boundary |
| Terminate | Set the run cancellation signal through code |

Instruction changes return an accepted or rejected receipt with request ID and applied revision.
The steward never edits a file while its worker consumes that file.
Python workflows expose explicit phase boundaries for updates.
Agent workflows consume approved updates between agent turns or phases.
Requests that cannot apply to the current run remain visibly pending or rejected.

Code emits lifecycle events even without an LLM.
Narrative status reports name their observation time and evidence.
A report cannot claim access to unfinished inference or unrecorded tool results.
Human questions include a question ID and accepted answer format.
The adapter turns question telemetry into MainCogLoop input. Responses return as correlated data.

Terminate remains available during all waits.
Cancellation interrupts supported I/O and prevents subsequent publication.
An arbitrary in-process Python function remains cooperative. Do not claim Python threads can forcibly stop it.
An already completed external tool effect is not undone by cancellation.
