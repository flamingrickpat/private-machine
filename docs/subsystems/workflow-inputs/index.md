# Scoped workflow inputs

## Summary

A03 supplies explicit inputs for roles, child workflows, and classifiers.
The kernel captures files, folder metadata, request fields, literal data, and named producer results through the working environment.
Receipts retain exact values and version evidence.
A04 supplies [folder selection, classification, and INVALID recovery](classification.md).
The normal LLM remains the default.

Parent: [workflow runtime](../../runtime.md).
Delivery: [A03](../../decisions/alpha-plan/work-items/a03.md).
Details: [binding syntax and producer scope](bindings.md), [capture and stale-data rules](versions.md), and [A03 verification](verification.md).

## Owners and entry points

`pm-workflows` owns manifest parsing, capture, output projection, and journal receipts.
PM supplies the actual folder or Ghost environment.
[`runtime.workflow_runner.WorkflowRunner.invoke`][] passes the request payload to [`pm_workflows.kernel.Kernel`][].
Python roles receive bound values through [`pm_workflows.python_role.RoleContext`][].
Agent roles receive the same capture receipts in their request prompt.
Child phases merge bound values into their explicit request.

[`pm_workflows.kernel.Kernel.capture_inputs`][] is also available to a phase-kind extension.
Classification uses it without a second host file loader.
[`pm_workflows.inputs.InputResolver`][] exposes separate listing and selected-file capture operations.
It performs no model inference.

```mermaid
flowchart LR
    M[Manifest input binding] --> E[Granted working environment]
    E --> C[Captured value and version]
    C --> R[Role or child request]
    R --> V[Validated result]
    V --> O[Named output in journal]
```

## Source modes and results

| Source | Consumer result | Version evidence |
| --- | --- | --- |
| `input_file` | Exact UTF-8 text or a structured projection | Relative path, file identity, revision, digest, and available producer metadata |
| `input_folder` | A direct-child candidate listing | Candidate IDs, metadata, listing revision, and selection instruction digest |
| `output` | A projection from a validated producer | Task, run, item, phase, attempt, revision, pointer, and value digest |
| `request` | A dotted request field | Request digest, field name, selected value, and current scope |
| `literal` | Declared JSON data | Exact source value, selected value, and digests |

`pointer` selects structured fields after source resolution.
`expected_type` asserts a JSON type.
The [binding guide](bindings.md) defines the canonical syntax and dynamic paths.
A folder binding preserves its complete listing for a later selection decision.
It cannot infer the latest virtual file from a fabricated timestamp.

## State and external effects

Capture reads through [`pm_workflows.working_target.WorkingEnvironment`][].
[`working_target.FolderEnvironment`][] supplies host stat metadata.
[`working_target.GhostBashEnvironment`][] supplies content versions and leaves unavailable times unknown.
Virtual paths never become host `Path` objects.
Child factories continue to supply the granted environment for each invocation.

Capture changes no working-target files.
The kernel appends input evidence and validated output projections to its private journal.
An output revision describes the producer candidate. It does not establish a later Git commit.
Existing checkpoint and route boundaries remain in control.
Consumer values and stored evidence use separate object copies.

Cancellation propagates between captures and before dispatch.
It does not undo a completed tool call or external effect.
The model and pm-coder paths retain their existing cancellation and compaction behavior.
No model, service port, or Minecraft lifecycle changes belong to A03.

## Failure and recovery

Invalid role inputs take the declared `on_invalid` route before dispatch.
The failed entry retains earlier successful captures and the concrete input error.
Incorrect named output projections invalidate the producer result and block older-value reuse.
[`pm_workflows.inputs.InputError`][] exposes a stable cause code for the classification INVALID route.
The [version guide](versions.md) explains stale-data causes and recovery limits.

Old manifests retain empty binding maps and their existing dispatch behavior.
Legacy child references now use the supplied environment instead of host-only reads.
Legacy host globs retain recursive patterns.
Virtual globs require exact folder components because the environment exposes direct-child file listings.
Trusted Python roles and shell tools retain their existing trust model.

## Operation

Use the local chat configuration for the [verification gate](verification.md).
Ordinary input capture needs no model or embedding endpoint.
The live agent evidence gate uses chat on port 8080 and the unchanged real Spamuel seed.
Laya remains optional and disabled for this mechanism.
The [classification recovery guide](classification-recovery.md) defines direct context bounds and agent compaction.
