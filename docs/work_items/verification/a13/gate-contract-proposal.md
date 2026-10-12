# Proposed contract for separate M3 checks

## Summary

This proposal separates historical attainment from current safety and usability.
Each decision receives its complete criteria and relevant original evidence.
Final integration still requires every retained M3 outcome together.
The proposal changes no runtime behavior, acceptance state, or model configuration.
The [implementation guide](../../../subsystems/gameplay-decision-scopes.md) describes the workflow changes and their current verification limits.

Parent: [task and gate review](task-gate-review.md).
Requirements: [A13](../../alpha-plan/work-items/a13.md).
Existing proof: [task evidence](../../../subsystems/minecraft-task-evidence.md) and [M3 verification](m3-house-and-iron.md).

## Preserve the dependency conflict explicitly

Revision 31 retains eight accepted leaves after M3-46.
The completed iron chain retains original proof, but its three certificates remain invalidated.
Their historical recovery dependency differs from the currently accepted recovery certificate.
An owned pickaxe alone cannot resolve that certificate conflict.

Exact restoration needs the original recovery specification and its complete dependency chain.
That historical recovery proves a past prerequisite, not present safety.
A separate incident node must own the new hazard and its fresh recovery evidence.
The root must require that current safety evidence before further physical work or publication.

The existing restoration operation cannot change dependency hashes or transfer proof to a new identity.
A migration must preserve original identities and use the normal planner, review, and commit path.
Migration failure leaves active acceptance unchanged and reports the conflicting dependencies.
The workflow now implements this migration through a separate native retention decision.
Its captured boundary check passed; real task acceptance remains pending.

## Inputs for every decision

Each request must include these inputs:

| Input | Required content |
| --- | --- |
| Identity | Task, tree revision, node specification digest, execution epoch, world, dimension, body session, and death count |
| Criteria | Complete unchanged criteria for the selected check, including thresholds, geometry, and lifecycle requirements |
| Current facts | Original observation path, digest, observation time, and freshness status. Cached facts remain explicitly historical |
| Dependencies | Full required certificate identities and hashes, including recursive dependencies |
| Relevant history | Original successful proof, failed methods, unresolved external effects, and relevant recovery incidents |
| Retrieval | Paths and digests for full real Spamuel history and task journals, with an available native retrieval path |
| Inference | Effective provider and model, context capacity, selected evidence, token count, measurement method, and response identity |

Referenced evidence must remain accessible to the decision.
A filename without retrieval cannot replace a required fact or criterion.
Context selection must preserve unresolved effects and successful material attainment.
All decisions must use real inference through the generic model path.
Model-specific branches and fixed responses cannot establish this contract.

## Separate decisions and proof

| Check | Prerequisite and evidence | Required result |
| --- | --- | --- |
| Proof retention | Archived specifications, original actor reports and journals, certificate files, and full recursive dependencies | Exact native restoration or a specific conflict. No repeat collection to resolve a protocol error |
| Iron provenance | Empty-iron baseline and ordered mining, interrupted-furnace recovery, smelting, and crafting receipts | Self-mined ore accounts for the crafted pickaxe. Supplied and village iron do not count |
| Structure | Original geometry and fresh independent observations of every required cell | Floor, walls, roof, both door cells, clear interior, and supported approach satisfy the retained criteria |
| Entry and exit | Verified structure, safe current body state, and unchanged position targets | Separate observed physical entry and exit in one body life |
| Home and return | Verified usable house, persisted waypoint, and ordered home/outside/home observations | Durable route evidence matches actual travel in one body life |
| Integration | All preceding results, original proof bindings, and fresh root observations | Every required retained outcome remains valid together before publication |

Proof retention precedes any material actor dispatch.
Current recovery takes priority over movement, building, or route travel.
Independent structure reads cannot infer hidden cells from a screenshot or partial area scan.
Stored coordinates cannot establish a tested return route.

Historical provenance remains distinct from current possession.
Integration needs both the original crafting chain and a fresh observation of the owned pickaxe.
A death or body-session change rejects an incomplete travel sequence.
An unrelated hazard cannot rewrite historical crafting criteria.

## Results, cancellation, and publication

Each result must record its check identity, input digest, model response, evidence paths, criterion results, and unresolved conflicts.
Result status distinguishes accepted proof, failed criteria, and unavailable evidence.
A transport error or cancelled read cannot become accepted proof.
New receipts belong in a fresh runtime directory.
Original receipts remain unchanged.

The task store remains the acceptance owner.
Check receipts cannot directly edit its acceptance index.
Plan changes must pass normal validation and review before durable commit.
The final integration must reject changed specifications, files, dependency hashes, or lifecycle identities.
Fresh world reads remain necessary after historical restoration.

Cancellation retains completed physical effects and their original journals.
An uncertain external effect requires reconciliation before another material action.
Only final integration can support the overall completion claim.

## Validation before another long run

1. Retain the repaired test-double context contract; the latest full foundation run passed 511 tests, including the caller-context regression.
2. Establish exact recursive restoration with retained real captures and native model inference.
3. Establish that a separate incident preserves historical proof while current safety remains independently required.
4. Verify rejection of altered files, identities, missing dependencies, and incomplete criteria.
5. Verify that selected context exposes full criteria and accessible original history within the measured budget.
6. Verify the effective hosted connection through the actual gate configuration path.
7. Resume the retained task without a reset, item grants, or manual certificate edits.
8. Complete fresh structure, entry, exit, home, return, and final integration checks.
9. Run fresh A13 compatibility scenarios against the delivered source.
10. Run an untouched real task to assess behavior beyond this repaired history.

The explicit A13 YAML still selects the local model endpoint.
The hosted `.env` values do not override that explicit connection.
M3-47 verified hosted native proof restoration, with no new gameplay actor.
Real native file-tool probes now establish reasoning-detail preservation across hosted tool turns.
Final hosted gameplay acceptance remains pending and requires its separate world audits.
The user permits workflow-script changes and generic runtime repairs.
Scenario-specific rules remain in the workflow and its acceptance gate.

Delivery still requires passing engineering checks, regenerated documentation, source review, and scoped commits and pushes after real acceptance.
A14 remains pending until A13 delivery.

Existing mechanisms: [`workflows.minecraft_tasks.tree.TreeStore.restore_specifications`][],
[`workflows.minecraft_tasks.tree.TreeStore.valid`][],
[`workflows.minecraft_tasks.verification.accepted_actor_matches`][],
and [`workflows.minecraft_tasks.verification.verified_project`][].
