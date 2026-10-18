# Cooperative survival choices

## Summary

The main cognition loop can choose game work, cooperation, or refusal from accepted affect, relationships, and current evidence.
Structured world events can start a cycle during user silence.
Goals retain their appraisal links and reasons.
Game effects still require actor receipts and independent observations.
Human partnership ratings remain separate.

Parent: [cognition](cognition.md).
World input: [world events](world-events.md).
Execution: [Minecraft projects](minecraft-tasks.md) and [specialists](minecraft-actor.md).
Evidence: [A14 verification](../work_items/verification/a14-cooperation.md).

## From evidence to a choice

[`workflows.main_cognition.host.MainCogLoop.prepare_input`][] supplies original stimuli, active goals, managed-run receipts, and selected memories.
User stimuli also supply the real user entity ID as `relationship_entity_id`.
World events supply no relationship entity ID.
A game hazard cannot directly become an adverse relationship update.

[`workflows.main_cognition.steps.run`][] runs the normal cognition phases.
Appraisal supplies causal evidence IDs and named axes or emotion deltas.
[`workflows.main_cognition.affect_dynamics.project_affect`][] projects affect once for the remaining stages and the final commit.
The model inputs contain all emotion values and the relationship fields, including trust and conflict tension.

Anger, resentment, jealousy, and boredom can influence a goal or action.
The model chooses the consequence.
No emotion threshold mandates refusal or destruction.
Cozy cooperation remains a target experience rather than an unconditional agreement rule.

Resentment, jealousy, and boredom use previously reserved vector positions 406, 407, and 408.
Existing positions and vector length remain unchanged.
Canonical axes retain the existing mapping for other emotions.
The new fields accept evidence-based named appraisal deltas, then follow normal decay and blending.

## Persistent relationship and goal reasons

A social user appraisal blends the existing relationship mapping with `BASE_APPRAISAL_GAIN`.
Only supplied social axes, `agency_other`, or `norm_violation` permit that update.
The appraisal must also cite an original user stimulus from the current batch.
An unrelated message preserves the relationship.
Elapsed-time emotion decay does not automatically reset trust or conflict tension.

[`workflows.main_cognition.stage_commits.apply_goal_changes`][] stages each accepted goal lifecycle change.
Its metadata now retains these fields:

| Field | Meaning |
| --- | --- |
| `appraisal_id` | Appraisal record accepted in the same cycle |
| `evidence_ids` | Original stimuli that caused the current choice |
| `reason` | Current accepted reason, including a later refusal or deferral |
| `choice_history` | Ordered prior choices with their reasons, evidence IDs, and appraisal IDs |

The action record also retains `appraisal_id`, `evidence_ids`, and `self_origin`.
`self_origin` means the committing batch contained no `user_input` envelope.
Task telemetry can also cause a self-origin choice; the field alone does not prove autonomous execution.
Inspect the original stimulus IDs and the resulting run receipt.

## Execution, updates, and public speech

Action selection can start a specialist or a generic project through SelfManagement.
Standalone specialist completion currently supports the explicit `wood_return` contract.
Other goals use `minecraft_project` for independent acceptance, including self-chosen mining or exploration.
An empty standalone acceptance contract retains actual effects but cannot establish terminal completion.
[`workflows.minecraft.host.persona_context`][] supplies accepted emotions and the relationship to the managed worker.
The body lease and command lock still govern external calls.
Game actor grants contain no fixture reset or RCON commands.

A footprint change can update an existing run through Assistant management tools.
Direct managed-workflow aliases also retain an existing `run_id` and map to an update unless an explicit operation selects another control.
Instruction revisions fence the next body call and cause project replanning.
The model must retain finished material work and discard stale construction assumptions.
Before action commit, the main module reads the current durable run table.
An active game run owns its original request correlation.
Progress for that request cannot start another game run while its worker remains active.
The rejected dispatch receives an explicit receipt. Existing management controls can still update or cancel the worker.
A new user input or a different request correlation can choose further work.
Completion releases this request boundary.
The [A13 gate review](../work_items/verification/a13/task-gate-review.md) records remaining generalization limits in this path.

Direct inventory questions use Assistant's current inventory observation and start no managed task.
Task journals and progress events enter [experienced memory](experienced-memory.md).
Return conversation uses that evidence through [public dialogue](public-dialogue.md) and [A09R speech](story-speech.md).
A shared plan, started worker, or completed workflow cannot alone establish a finished house or expedition.

## Diagnosis and acceptance

Inspect the appraisal record, its evidence IDs, the goal's `choice_history`, and the action's dispatch receipt.
Then inspect actor results and fresh inventory or exact block reads.
Compare public progress claims with the same dated evidence.
Retain failed receipts until a new real run establishes a repair.

The [A14 gate](../work_items/verification/a14-cooperation.md) retains full Spamuel history, the real world, and separate pending M4 ratings.
Its machine checks do not rate helpfulness, interruption quality, or immersion.
