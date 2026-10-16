# Workflow review modes

## Summary

Workflows can accept a sufficient result or a permitted improvement without another repair loop.
Each workflow selects its review mode.
The shared runtime helper classifies the judgment; it does not perform work or create completion evidence.

Parent: [workflow runtime](../runtime.md).
Current caller: [Minecraft projects](minecraft-tasks.md).
Planning interface: [draft sessions](draft-sessions.md).
Observed failure and repair: [A14 evidence](../work_items/verification/a14-cooperation.md).

## Passing judgments

[`pm.runs.review.review_passes`][] implements these modes.
[`pm.runs.review.review_policy`][] supplies their permitted judgments to the model.

| Mode | Passing judgments |
| --- | --- |
| `flexible` | `accepted`, `good_enough`, `equivalent_or_better` |
| `strict` | `accepted` |

`good_enough` means the result meets the practical request despite minor wording or implementation gaps.
`equivalent_or_better` means a permitted alternative meets or improves the requested outcome.
A passing judgment releases review directly.
These are model judgments, not human quality ratings.

The workflow still owns its schema, state, required evidence, and commit boundary.
Other workflows can adopt the helper with their own validators.
There is no global switch that changes all reviews.
Public speech also accepts sufficient or equivalent replies while retaining advisory findings and deliberate-deception choices.
[Review materiality](review-materiality.md) explains the classes, examples, prompt coverage, and A18B scoring behavior.

## Minecraft policy

[`workflows.minecraft_project.schemas.ProjectRequest`][] defaults to `review_mode: flexible`.
A caller can set `review_mode: strict` in the project arguments.
Both modes retain explicit user requirements and the existing independent completion checks.

The planner, reviewer, and actor receive `project_intent` containing the original request, explicit constraints, and current instruction.
A newer footprint instruction changes dimensions without erasing an earlier preservation requirement.
The actor receives this global intent beside its selected leaf instructions.

The reviewer judges practical feasibility and required outcomes.
It can pass work that uses already-owned components, routine recipe resolution, or a coherent leaf spanning gathering through construction.
Task names do not require extra crafting.
Optional aesthetics or extra components do not become mandatory work.
A survey can report an occupied or unsuitable site without clearing a protected structure.
Refinement remains available when work needs separate scheduling or a demonstrated failure needs a smaller scope.

[`workflows.minecraft_project.planning.review.record_review`][] records the permitted judgment and reviewed revision.
The role maps each passing judgment to the existing `accepted` kernel route.
An approving plan review does not create a task acceptance certificate.
Strict-mode rejection removes an earlier review approval.

## Completion evidence and bookkeeping

[`workflows.minecraft_project.planning.verify.verify_node`][] still requires every backend condition check to pass.
Leaves also require their original actor report, paired journal calls, correct task identity, and the current instruction boundary.
A model judgment cannot make missing construction pass those checks.
Permitted alternatives can revise an unaccepted specification before fresh verification establishes their actual result.

The backend now records complete criterion coverage from its checks.
The model can omit `criterion_ids` or cite only relevant IDs.
Its original list stays in `reported_criterion_ids` for diagnosis.
This avoids rejecting a valid result because the model failed to copy a bookkeeping list.

The original model decision remains in the decision artifact.
Verified receipts retain `review_judgment` and `review_mode`.
Their normalized `review.status: accepted` preserves the existing certificate-reader contract.
`criterion_coverage_source: backend_checks` identifies the owner of the coverage list.
Cancellation and stale instructions still prevent the acceptance commit.

## Inspect and repeat

Read the project decision's `input.json` and `output.json` together.
The input records `review_policy` and `project_intent`; the output records the actual judgment and reason.
Then inspect the retained tree, verification reads, checks, actor journal, and acceptance certificate.
Keep plan approval separate from completed execution.

The A14 captured door case held two doors and required one.
Its former reviewer rejected that sufficient stock because the task name implied crafting and optional components might exist.
A fresh real Qwen review with the revised policy accepted the same captured evidence.
Evidence: `.runtime/alpha/20261014-a14-review-leniency-original-intent/receipt.json`.
That result establishes the repaired review case, not a newly completed house or general latency improvement.
