# Workflow classification

## Summary

`kind: classify` selects eligible evidence and routes one declared class.
The normal LLM performs bounded selection and classification.
INVALID sends unusable evidence or responses to a dedicated coding-agent handler.
Existing workflows keep their original manifests and connections.
A05 supplies [lazy PM connection wiring](pm-connections.md) and a [demonstration workflow](../../workflows/classification-demo.md).

Parent: [scoped inputs](index.md).
Details: [recovery and context](classification-recovery.md) and [verification](../../work_items/verification/a04-classification.md).
Delivery: [A04](../../work_items/alpha-plan/work-items/a04.md).

## Manifest contract

[`pm_workflows.manifest.parse_workflow`][] accepts the following primary and recovery nodes:

```yaml
roles:
  inspector:
    skill: inspect-tests.md
    readable_paths: ["logs/**"]
phases:
  - name: check
    kind: classify
    backend: llm
    connection: Default
    input_folder: logs
    data_selection: Select the latest foundation test log by filename and metadata.
    instruction: Classify the current completed test summary.
    result_contract:
      type: classes
      enum:
        passed: At least one test ran and all tests passed.
        failed: The completed run reports a failed test.
    on_class: {passed: stop, failed: stop, INVALID: inspect}
  - name: inspect
    kind: classify
    fallback_for: check
    role: inspector
    on_class: {passed: stop, failed: stop, INVALID: stop}
```

This example uses reserved terminal routes for an engine demonstration.
An application can replace them with declared accept, repair, wait, or failure phases.
An unresolved handler cannot route back to either classification node, even through another phase.
Its successful class routes must match the primary routes.
Both nodes must occupy the same loop scope.

INVALID exists implicitly. The class enum must contain nonempty labels and descriptions, without INVALID.
`on_class` must route every declared class and INVALID, with no extra labels.
The INVALID target must name this classifier's dedicated `fallback_for` handler.
Unknown destinations, conflicting route fields, Python handlers, MCP grants, and writable handler paths fail manifest loading.
`backend: llm` is the only active backend. Optional Laya remains disabled.

## Inputs and selection

The primary node declares exactly one [A03 binding](bindings.md).
It can use the shown source fields directly or one entry under `inputs`.
Exact files, bounded exact files, request fields, literal values, named outputs, pointers, and type assertions remain available.
Conflicting source forms fail loading.

An exact `input_file` skips selection.
An accompanying `input_folder` enforces its boundary.
A folder-only input requires `data_selection` and produces two actual model calls:

1. Capture a direct-child listing through the granted working environment.
2. Select a known candidate ID from its names and metadata.
3. Check the complete listing again and capture the selected file version.
4. Apply any pointer and type assertion to the selected content.
5. Classify that captured value and validate the answer.

Selection does not inspect unseen contents or impose a minimum log size.
Unknown timestamps and producer metadata remain null.
The model must return INVALID when the requested selection cannot follow the available metadata.
The [version rules](versions.md) reject changed files and previous-item reuse.

## Answer and routing

[`pm_workflows.classification.answer_contract`][] supplies the same schema to both model paths.
It uses typed JSON Schema. `status` is one string label, never an array.

| Field | Required meaning |
| --- | --- |
| `status` | One declared class or `INVALID` |
| `reason` | A nonempty explanation or concrete invalid cause |
| `evidence` | Exact, nonempty quotes from the captured value for a class |
| `candidate_id` | The selected listed ID, or null for an exact or structured source |
| `input_sha256` | The captured `value_sha256`, or null for INVALID |

INVALID uses null identifiers and can use an empty evidence list.
The value digest hashes canonical JSON, including string quotes and escapes.
It can differ from the digest of raw file bytes.
Structured evidence uses canonical JSON text for quote matching.

[`pm_workflows.classification.validate_answer`][] rejects unknown fields, missing fields, labels, candidates, digests, and unsupported quotes.
A normal HTTP response does not establish semantic correctness.
Exact quote validation establishes source correspondence; the live cases establish their independently expected classifications.
Ambiguous interpretation still requires human review.

## Runtime connection boundary

[`pm_workflows.runtime.WorkflowRuntime`][] supplies `classification_models`, a mapping from connection names to resolved [`pm_workflows.runtime.AgentModel`][] values.
The selected entry contains endpoint, model, credentials, serving context, and diagnostic mode.
Recovery uses the separate `agent_model` connection.
A missing classification connection produces `missing_connection`; the engine does not silently select another endpoint.
Ordinary workflows need no new mapping.

PM installs the optional `classification_session` callback for lazy service ownership.
Explicit connections retain their existing path.
The [connection guide](pm-connections.md) defines activation, context discovery, child inheritance, and service receipts.

## Receipts and debugger output

[`pm_workflows.classification.execute_classification`][] appends a `classify` execution entry to the kernel journal.
Its `result.receipt` uses `pm.classification-receipt.v1`.
It retains scope, attempts, question, classes, configuration digest, limits, captures, decisions, complete requests, complete responses, usage, and timings.
Credentials stay outside receipts.
The `StepResult.data` field exposes this same result to a synchronous debugger caller.
`state.md` shows the class and errors in execution history.

[`pm_workflows.dryrun.build_graph`][] includes every class edge, INVALID, and its handler without model execution.
[`pm_workflows.classification.replay_classification`][] validates retained values, digests, quotes, and the answer without filesystem reads or inference.
Replay does not repeat semantic interpretation.
Accepted output declarations use the existing named-output scope and attempt rules.
Failed declarations block older successful values.
Primary and handler output names remain distinct producer names.
