# Input selection and workflow classification

## Summary

A classification phase uses the LLM to select a file before classifying its contents.
Known paths and named outputs remain available.
Both passes use concrete captured inputs.
Every classification has an INVALID outcome and a scoped coding-agent fallback.

Parent: [models and workflow inputs](decisions.md).
Delivery: [A03](work-items/a03.md) and [A04](work-items/a04.md).
Policy: [LLM default and optional Laya tests](model-policy.md).

## Explicit data flow

The user confirmed folder selection plus classification.
The engine already stores structured role results and supports child request expansion.
Extend those mechanisms and the working-environment contract.

A03 now supplies the [binding contracts](../../subsystems/workflow-inputs/bindings.md) and
[scoped version checks](../../subsystems/workflow-inputs/versions.md).
These run on role and child-workflow phases today.
The classification node and inference routes in this page remain A04 proposals.

| Input mode | Meaning |
| --- | --- |
| `input_file` | Classify this exact file; skip file-selection inference |
| `input_folder` with `data_selection` | Select from this folder's scoped listing and metadata through the LLM or explicit deterministic rules |
| Named output | Select a validated result field from a named producer |
| Request field or literal | Use declared structured data or concrete text |

When both path fields exist, `input_folder` bounds the permitted explicit file.
The node does not run selection again for that exact path.
Reject contradictory input modes.

Use `WorkingEnvironment` for listing, metadata, and reads.
A virtual path must never become a host `Path`.
Preserve child grants and relative-path rules.

## The two-pass selection path

1. List eligible files inside the declared folder and capture their metadata.
2. Ask the selected LLM which candidate satisfies `data_selection`, unless an explicit deterministic rule establishes the result.
3. Read and snapshot that selected file.
4. Ask the selected LLM to classify the captured content into declared classes.
5. Route the accepted label or INVALID.

Selection receives stable candidate IDs, names, sizes, timestamps or revisions, and available producer metadata.
The selected ID maps to a known listed path.
The model cannot invent a path outside the candidate set.
Optional Laya can replace either inference only after live-endpoint tests establish that exact simple use.
Selection and content classification need separate tests.
Do not add expensive Laya tuning to deliver this path.

File metadata differs across environments.
Extend the environment contract when necessary.
Expose unavailable fields as unknown instead of fabricating modification times.

Record both decisions, the listing revision, selection instruction, selected path, input digest, and result.
Preserve producer attempt, loop item, and instruction revision when available.
When producer metadata is absent, require fresh listing and file evidence.
Do not claim an agent-created report has provenance that was never recorded.

Check the selected file again before capture.
A changed file invalidates the selection attempt.
A loop cannot quietly reuse a previous item's file as current evidence.

## Proposed low-code manifest

This is a proposed interface, **not supported syntax today**.
The existing engine uses `kind`; the user's sketch used `type`.
A04 must choose one canonical spelling.

```yaml
- name: unit_test_checker
  kind: classify
  backend: llm
  connection: Default
  input_folder: "test_results/"
  data_selection: "Which file is the most recent unit-test log? {files}"
  instruction: "Classify the observed test results."
  result_contract:
    type: classes
    enum:
      all_good: All unit tests passed.
      some_passed: Some unit tests passed and some failed.
      all_failed: All unit tests failed.
  on_class:
    all_good: accept
    some_passed: repair_tests
    all_failed: inspect_failure
    INVALID: unit_test_classification_agent
```

INVALID always exists, even when omitted from the user's enum.
It can represent a semantic invalid answer or a runtime failure to classify.
Its receipt records the concrete reason.

The proposed `backend` field makes the normal LLM path explicit.
A04 resolves the final spelling and connection contract against the existing engine.
Optional `backend: laya` is permitted only for a use that passes the [live test requirements](model-policy.md#before-enabling-a-laya-use).

For a known file, replace selection with:

```yaml
input_file: "test_results/known-result.txt"
```

Named outputs remain optional.
Existing workflows do not need to structure every report before using classification.

## INVALID and the coding-agent path

Missing files, empty listings, inaccessible data, oversized inputs, malformed answers, and uncertain selection or classification produce INVALID.
A near-equal distribution is an uncertainty example.
For optional Laya, measure confidence and margin thresholds on that use's realistic live tests.
Never assume confidence establishes correctness.
LLM uncertainty can follow missing or contradictory evidence without a probability distribution.

The manifest names the coding-agent phase that handles this particular classification.
It receives the same instruction, class definitions, INVALID reason, and captured selection/answer evidence.
Its configured tools can inspect or process long files within its grants.
Its result must satisfy the same class contract.

If the agent still lacks evidence, it returns INVALID with a useful cause.
The workflow declares a wait, repair, or failure destination.
Do not create an infinite classifier-to-fallback recursion.

The earlier 95% Laya / 5% coding-agent target is withdrawn.
The normal LLM can handle every classification in a workflow.
Measure selection errors, classification errors, fallback frequency, and critical misroutes.
If enabling Laya requires extensive effort or still produces incorrect answers, keep it disabled.

## Limits and compatibility

Candidate lists and content must fit measured model limits.
Oversized listings or content produce INVALID unless an explicit selection/compaction stage is configured.
Silent clipping cannot discard route-defining evidence.
Laya's decision-head budget does not constrain chat or pm-coder.
Use direct LLM calls for suitable captured inputs and pm-coder for agentic long-file inspection.
Retain class definitions, evidence identifiers, and necessary conclusions through agent compaction.
Direct chat calls do not automatically inherit that agent recovery loop.
Diagnose and repair actual LLM context or incomplete-output failures before delivery.

Validate every label and destination, including INVALID.
Include the generated invalid edge and fallback in static/dry-run topology.
A replay uses the captured listing, content, and receipts.

Preserve old manifests and existing transactional coding behavior.
Minecraft classification retries retain actual world effects.
Score and probability decisions remain available through the public client.
Additional workflow result kinds can use the same decision machinery after class routing is verified.
