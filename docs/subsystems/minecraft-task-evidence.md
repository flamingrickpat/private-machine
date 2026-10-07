# Independent Minecraft task evidence

## Summary

An actor proposes completion.
A separate verifier reads the world through read-only tools.
Mechanical conditions and an independent interpretation must both pass.
Acceptance retains specification guards, prerequisite guards, and hashes of the original evidence files.
Public task claims recompute those retained facts.

Parent: [durable projects](minecraft-tasks.md).
Related: [public publication](public-publication.md) and [A11 verification](minecraft-task-verification.md).

## Observation primitives

[`workflows.minecraft_tasks.schemas.Condition`][] supplies six generic primitives.
They do not encode house, mining, or farm goal categories.

| Kind | Independent tool | Required fact |
| --- | --- | --- |
| `inventory` | `minecraft_observe` | The sum of matching item counts meets `minimum` |
| `block` | `minecraft_inspect_block` | The exact cell contains an explicitly accepted block name |
| `region` | `minecraft_analyze_area` | Complete visible coverage and enough explicitly named blocks |
| `position` | `minecraft_observe` | Correct dimension and distance within `radius` blocks |
| `alive` | `minecraft_observe` | Observed health is greater than zero |
| `accepted` | Retained independent certificate | A descendant or prerequisite node has valid verified attainment |

Inventory names accept globs.
Block and region names use exact tool identifiers.
The vanilla `minecraft:` namespace prefix is optional and normalizes to the MCP identifier.
Block conditions can list several exact positions with one common set of accepted names.
Every listed position must match.
Only region conditions use `end` and count thresholds.
An unsupported block-volume/count combination fails before planning acceptance.
The configured deployment reports `overworld` as its dimension name.
Block conditions can explicitly require `air`.
The inspection tool reports empty cells with `ok=false` and `reason=not_found`.
Only that specific empty-cell result can satisfy an air condition.

Region bounds use inclusive integer corners.
The evaluator requires the expected cell volume and zero unseen cells.
A histogram with hidden cells cannot establish an enclosed roof or room.
Exact cell inspection supplies that evidence instead.
Block and region checks also require a fresh dimension observation.

## Evidence lifetime

Acceptance records attainment at the verifier's observation time.
For example, a gathering leaf can establish that materials entered inventory.
A dependent crafting leaf can consume those materials without erasing the earlier achievement.
This retained acceptance does not claim that the inventory remains unchanged.
Aggregate criteria use `accepted` with `node_id` to refer to earlier achievements.
This permits a material story to cite gathering and crafting after construction consumes the stock.
It also permits a usability story to cite entry and exit from their separate verified positions.
The evaluator requires the exact current certificate hash and its verification path.
The verifier receives the original source specification, checks, attempt identity, and observation times.
Hierarchy and dependency guards reject self references, missing sources, and unrelated work.
They also reject cycles across child, dependency, and certificate references.
Refinement preserves references to a stable node when that leaf becomes a story.
The story must meet its original criteria after all its children pass.
Its certificate releases dependent work before that work consumes the observed materials.
Final structure criteria still use fresh block reads.

Assumptions represent facts that must remain true.
The selector reads accepted nodes' assumptions before it releases another leaf.
It also reads the selected node's assumptions and its ancestors' assumptions before dispatch.
Changed assumptions remove affected acceptance and request replanning.
Each check retains its exact identity, original reads, and mechanical results under `planning/assumptions/`.
The existing world remains intact.

## Attempt identity and original calls

Every decision input names the task ID, tree revision, leaf ID, specification revision, actor attempt, and verifier attempt.
Planning decisions leave irrelevant attempt fields empty.
An actor child receives the selected leaf and revision in `task_reference`.
Its acceptance context includes prerequisite specifications, verifier identities, checks, and certificate hashes.
This lets the fresh actor use earlier verified work without inventing another session's history.
Ancestor specifications provide the project goal and reserved construction cells.
Prerequisite actor summaries identify observed side effects, such as a placed workbench.
Repeated attempts also receive the original handoff for that leaf.
Current specifications and fresh observations take precedence over that past handoff.
The verifier requires that same reference before acceptance.

Each native actor has its own `attempts/actor-NNNN/` folder and session reference.
Its original gateway journal pairs dispatches with results by sequence, tool, and arguments.
Unpaired calls, duplicate results, uncertain outcomes, and stale task references fail the completion gate.
The independent verifier stores its own original reads under `verification/`.
It writes those reads before inference so cancellation preserves them.
Interpretation receives relevant structured fields and hashes of the complete original results.
The actor cannot supply those observations through its summary.

[`workflows.minecraft_tasks.verification.observe_conditions`][] can issue only the declared observation primitives.
It holds the shared command boundary for each read.
It has no movement, construction, mining, or item-grant tools.
[`workflows.minecraft_tasks.verification.evaluate`][] recomputes every criterion from its matching read.
Missing or duplicate criterion reads fail closed.

## Acceptance and public claims

[`workflows.minecraft_tasks.tree.TreeStore.accept`][] requires a current specification, independently sufficient reads, and accepted interpretation.
It also requires valid prerequisite acceptance.
The record hashes the verifier receipt and original actor report and journal.
Dependent records retain hashes of their prerequisite acceptance records.
Changed evidence invalidates the affected acceptance chain.

[`workflows.minecraft_tasks.verification.verified_project`][] checks the complete retained chain again.
It recomputes numeric conditions, interpretation coverage, actor identity, and original call pairing.
[`runtime.reply_recovery.task_evidence`][] invokes that function before public task dispatch.
A managed run's `completed` state alone cannot support a Minecraft success claim.

The public result describes verified attainment from retained observations.
It does not describe a new live inspection of the current world.
External players can change the world after verification.
Current-state questions therefore require a fresh observation.

## Diagnostic procedure

1. Open the managed receipt's `artifact_path`.
2. Read `planning/state.json` and its active `planning/specs/NNNN.json`.
3. Find the selected leaf and its exact actor attempt.
4. Read that attempt's `inbox/request.yaml` and `scratch/actor-journal.jsonl`.
5. Read the named verifier receipt under `verification/`.
6. Compare each criterion with its original read and mechanical result.
7. Read the corresponding verifier decision input and output under `decisions/`.
8. Inspect prerequisite and evidence-file guards before interpreting the public report.

## Traps

Progress and completion candidates are separate outcomes.
An accepted past inventory count cannot establish a current resource reserve.
An unavailable observation cannot establish absence.
An image file cannot establish visual interpretation.
Git changes cannot restore a mutable world.
Saved attempt state cannot establish safe replay after a crash.

## Go deeper

Read [project routing](minecraft-tasks.md) for lifecycle and instruction boundaries.
Read [verification](minecraft-task-verification.md) for recorded real examples.
