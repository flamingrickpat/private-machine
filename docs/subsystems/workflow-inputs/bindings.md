# Input bindings and producer scope

## Summary

An input binding declares one source, an optional JSON pointer, and an optional type assertion.
Named outputs project validated results without a file round trip.
Their identity includes the exact producer attempt and logical item.
Dynamic file paths retain those dependencies in the capture receipt.

Parent: [scoped workflow inputs](index.md).
Related: [version rules](versions.md) and [verification](verification.md).
API: [`pm_workflows.protocol.InputBinding`][] and [`pm_workflows.protocol.OutputBinding`][].

## Phase syntax

Bindings belong to a phase's `inputs` mapping.
Output declarations belong to its `outputs` mapping.
Names use `[A-Za-z_][A-Za-z0-9_]*`.
Built-in role and child-workflow phases support these fields.
A phase extension can consume the same parsed contracts.
The following maps illustrate separate consumer and producer phases.

```yaml
inputs:
  report:
    input_file: "${request.report_path}"
    input_folder: reports
    pointer: /tests/0/result
    expected_type: object
  candidates:
    input_folder: test_results
    data_selection: Select the most recent unit-test log.
  conclusion:
    output: accepted_report
    pointer: /summary
    expected_type: string
  job:
    request: task.id
    expected_type: string
  policy:
    literal: {accept_incomplete: false}
    expected_type: object
outputs:
  accepted_report:
    pointer: /report
    expected_type: object
```

Each input declares exactly one source mode.
`input_file` can also declare `input_folder`, which bounds that exact file.
That combination skips selection and rejects `data_selection`.
A folder-only source requires a nonempty selection instruction.
Its result is a listing, rather than selected content.

The parser rejects unknown binding keys, malformed types, ambiguous modes, and duplicate output names.
A direct `output` binding must name a declared producer.
Runtime checks reject unresolved path variables and missing fields.
Paths use relative POSIX spelling, such as `reports/current.yaml`.
Absolute runtime paths, drive letters, backslashes, and traversal fail capture.

## Structured selection

An empty `pointer` returns the whole source.
A nonempty pointer starts with `/` and uses RFC 6901 escapes.
`~1` represents `/`. `~0` represents `~`.
Array indexes use canonical nonnegative decimal spelling.
Missing keys, out-of-range indexes, and malformed escapes fail.

File pointers parse JSON, YAML, Markdown YAML frontmatter, or the first JSON/YAML fenced block.
An unprojected file remains exact text.
[`pm_workflows.inputs.select_pointer`][] performs the same selection for request, literal, and output values.
[`pm_workflows.inputs.require_type`][] accepts `string`, `object`, `array`, `integer`, `number`, `boolean`, and `null`.
Booleans do not satisfy numeric assertions.
Nonfinite numbers, foreign objects, and nonstring mapping keys fail JSON validation.

`request` uses the existing dotted expansion convention.
For example, `task.items.0.id` selects a field from the first item.
[`runtime.workflow_runner.WorkflowRunner.invoke`][] supplies its original payload through `Kernel.request_data`.
A child receives only the explicitly resolved child input mapping as its request data.
The kernel snapshots that mapping at construction.

## Dynamic paths

Known paths skip all selection inference.
Paths can use `${request.report_path}` or `${outputs.report_path}`.
Foreach child bindings can use their existing item variable, such as `${job.id}`.
Exact variable matches preserve structured values in existing child inputs.
A file-path expression must resolve to text.

Each dynamic capture records the original binding, its digest, and named-output dependencies.
Those dependencies retain the exact producer attempt.
A later file consumer therefore cannot silently treat an old producer result as a new path.
Existing child `{from: "report.yaml#/field"}` references also read through the supplied environment.
Legacy host absolute references remain permitted only inside the host workspace.

## Output acceptance and item isolation

[`pm_workflows.inputs.InputResolver.project_outputs`][] runs after the producer result satisfies its role or child contract.
Every output must satisfy its own pointer and type assertion.
A bad projection invalidates the whole attempt.
The journal stores null declarations for failed producer attempts.
The latest failed declaration blocks reuse of an older successful value.

A named record binds task ID, run ID, current item, producer phase, attempt, candidate revision, value, pointer, and digest.
The consumer must match the task, run, and item scope exactly.
The digest must still match the recorded value.
Child kernels keep separate journals and output namespaces.
Required parent values cross that boundary through explicit child inputs.

Foreach captures include the parent item, phase, and stable child item ID.
An unchanged file captured for an earlier item cannot become the next item's report.
The resolver records missing producer metadata as a limitation.
It cannot prove producer identity from a report filename or prose alone.
The [version guide](versions.md) describes these checks and their limits.

## Consumer interfaces

Python roles receive `context.inputs` and `context.input_evidence`.
Agent roles receive the full evidence mapping in their prompt.
Child phase bindings merge into resolved `task.input` values.
A duplicate key between those mappings fails instead of choosing one silently.
Output declarations on a child phase project its normalized result, including `children` and `status`.
Existing role contract validation remains compatible for manifests without output declarations.

The [A03 live gate](verification.md) supplies a real report answer and original Spamuel memory through this interface.
Its generated manifest remains in the ignored evidence directory for direct inspection.
