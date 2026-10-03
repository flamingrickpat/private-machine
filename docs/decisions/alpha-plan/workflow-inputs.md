# Input selection and workflow classification

## Summary

A classification phase can let Laya select a file before classifying its contents.
Known paths and named outputs remain available.
Both passes use concrete captured inputs.
Every classification has an INVALID outcome and a scoped coding-agent fallback.

Parent: [Laya decisions](decisions.md).
Delivery: [A03](work-items/a03.md) and [A04](work-items/a04.md).

## Explicit data flow

The user confirmed folder selection plus classification.
The engine already stores structured role results and supports child request expansion.
Extend those mechanisms and the working-environment contract.

| Input mode | Meaning |
| --- | --- |
| `input_file` | Classify this exact file; skip file-selection inference |
| `input_folder` with `data_selection` | Let Laya choose from this folder's scoped file listing and metadata |
| Named output | Select a validated result field from a named producer |
| Request field or literal | Use declared structured data or concrete text |

When both path fields exist, `input_folder` bounds the permitted explicit file.
The node does not run selection again for that exact path.
Reject contradictory input modes.

Use `WorkingEnvironment` for listing, metadata, and reads.
A virtual path must never become a host `Path`.
Preserve child grants and relative-path rules.

## The two-pass fast path

1. List eligible files inside the declared folder and capture their metadata.
2. Ask Laya which candidate satisfies `data_selection`.
3. Read and snapshot that selected file.
4. Ask Laya to classify the captured content into declared classes.
5. Route the accepted label or INVALID.

Selection receives stable candidate IDs, names, sizes, timestamps or revisions, and available producer metadata.
The selected ID maps to a known listed path.
The model cannot invent a path outside the candidate set.

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
  connection: Laya
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

For a known file, replace selection with:

```yaml
input_file: "test_results/known-result.txt"
```

Named outputs remain optional.
Existing workflows do not need to structure every report before using classification.

## INVALID and the coding-agent path

Missing files, empty listings, inaccessible data, oversized inputs, malformed answers, and uncertain selection or classification produce INVALID.
A near-equal distribution is an uncertainty example.
Measure confidence and margin thresholds on PM examples.
Never assume confidence establishes correctness.

The manifest names the coding-agent phase that handles this particular classification.
It receives the same instruction, class definitions, INVALID reason, and captured selection/answer evidence.
Its configured tools can inspect or process long files within its grants.
Its result must satisfy the same class contract.

If the agent still lacks evidence, it returns INVALID with a useful cause.
The workflow declares a wait, repair, or failure destination.
Do not create an infinite classifier-to-fallback recursion.

The 95% Laya / 5% coding-agent split is a design target.
It applies only to workflows that deliberately choose this classifier.
It is not a quota for replacing LLM stages elsewhere in PM.
Measure selection errors, classification errors, fallback frequency, and critical misroutes.
Do not obtain that ratio by forcing uncertain answers.

## Limits and compatibility

Candidate lists and content must fit measured model limits.
Oversized listings or content produce INVALID unless an explicit selection/compaction stage is configured.
Silent clipping cannot discard route-defining evidence.

Validate every label and destination, including INVALID.
Include the generated invalid edge and fallback in static/dry-run topology.
A replay uses the captured listing, content, and receipts.

Preserve old manifests and existing transactional coding behavior.
Minecraft classification retries retain actual world effects.
Score and probability decisions remain available through the public client.
Additional workflow result kinds can use the same decision machinery after class routing is verified.
