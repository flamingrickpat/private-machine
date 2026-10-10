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

[`workflows.minecraft_tasks.schemas.Condition`][] supplies seven generic primitives.
They do not encode house, mining, or farm goal categories.

| Kind | Independent tool | Required fact |
| --- | --- | --- |
| `inventory` | `minecraft_observe` | The sum of matching item counts meets `minimum` |
| `block` | `minecraft_inspect_block` | The exact cell contains an explicitly accepted block name |
| `region` | `minecraft_analyze_area` | Complete visible coverage and enough explicitly named blocks |
| `position` | `minecraft_observe` | Correct dimension and distance within `radius` blocks |
| `alive` | `minecraft_observe` | Observed health is greater than zero |
| `health` | `minecraft_observe` | Observed health is at least `minimum` points on the 0..20 scale, in the requested dimension |
| `food` | `minecraft_observe` | Observed food meets `minimum` on the 0..20 scale in the requested dimension |
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
The area tool counts empty cells in `air_cells`, outside its nonempty block histogram.
That count combines `air`, `cave_air`, and `void_air`.
To use the count, a region condition must explicitly accept all three names.
An exact air variant requires individual block inspections.
Missing, invalid, or out-of-range aggregate counters cannot establish those empty cells.
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

### Retained certificates after an accidental revision

[`workflows.minecraft_tasks.tree.TreeStore.archive_acceptances`][] retains valid certificate records before a specification changes.
Each archive includes the exact specification, original certificate, and source receipt.
The files live under `planning/acceptance-archive/` in the owning task.
The planner receives candidates for invalid work and its changed certificate dependencies.
The native planner must explicitly restore the exact earlier specification.

[`workflows.minecraft_tasks.tree.TreeStore.restore_acceptances`][] then applies the normal specification, original-file, and dependency-hash guards.
It restores prerequisites before their dependents.
Changed files, changed operands, and different dependency certificates prevent reuse.
Fresh assumption failures revoke the original certificate digest permanently.
Historical attainment cannot establish current inventory or final geometry.

The real acceptance wrapper imports older valid certificate snapshots before restarting its worker.
It checks the retained predecessor receipt's digest and owning task first.
This supports tasks that predate certificate archives without changing their accepted specifications.
M3-43 changed an accepted recovery dependency and removed three completed iron certificates.
The captured regression retains 24 original evidence files with their exact bytes and hashes.
The compressed fixture is `a13-recovery-certificate-preservation.json`.
Its checks establish safe reuse rules; they do not establish a new live acceptance.

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

[`workflows.minecraft_tasks.tree.TreeStore.restore_specifications`][] applies explicit `restore_nodes` requests from the native planner.
Every restored node requires its original files and recursive dependency certificate hashes.
The planner must list changed historical dependencies too.
The operation validates the proposed tree before it changes durable state.

Restoration can increase the current specification revision counter while recovering exactly the original specification content.
[`workflows.minecraft_tasks.verification.accepted_actor_matches`][] therefore compares actor counters with the original certificate identity.
It also requires exact current specification content, unchanged evidence files, matching task identity, and complete dispatch-result pairing.
The original actor, verifier, and certificate counters remain unchanged.
Fresh final project reads still establish current structure and inventory.

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
