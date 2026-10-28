# Private reflection experiments

## Summary

The private loop can record a prediction, change attention advice or a subjective belief, and inspect later choices.
Each experiment retains its original prediction, source revisions, failures, and reversal history.
The engineering implementation is in progress under [A19](../work_items/alpha-plan/work-items/a19.md).
The complete behavioral acceptance and M6 review remain pending.

Parent: [private loop](private-loop.md).
Related: [implementation inspection](introspection.md), [deliberate mental work](deliberate-mental-work.md), and [experiment lifecycle](reflection-lifecycle.md).
Evidence: [A19 verification](../work_items/verification/a19-reflection.md).

## Trigger and execution

`reflection_review` is a private operation in the existing inner loop.
Fresh workspace material, surprising outcomes, unresolved commitments, and pending expectations can make it available.
An experiment that needs application or a later evidence review adds an `experiment_review` trigger.
The trigger model still chooses whether to do this work.
Repetition or negative emotion alone does not establish a defect.

The ordinary trigger, thought, and integration stages still run.
The additional reflection child then makes three focused inference calls:

1. The monitor identifies a mismatch or successful pattern in retained evidence.
2. The assessor compares explanations, counterevidence, and unresolved uncertainty.
3. The guide proposes one experiment lifecycle step.

The host accepts the experiment step in the same memory batch as the private thought.
Public disclosure remains a separate choice.
Reflection does not require a public explanation, emotional reset, or agreement with the user.
The main appraisal path still owns goals and the mental-state vector.

## Evidence and prediction

An experiment plan names its triggering episode, finding, diagnosis, scope, and learning kind.
It contains at least two explanations and identifies original support, opposing evidence, and uncertainty.
The model's confidence stays separate from its source IDs.
Repeated interpretations of one record provide no additional independent evidence.

The plan also records a predicted effect, discriminating observation, future observation event, and permitted intervention.
The `proposed` commit stores the previous effective state for reversal.
A later `apply` step must read that committed proposal.
It cannot create a prediction after the outcome.

The request includes selected experiments and exact source records with revisions.
It selects at most eight open experiments and four terminal experiments.
The request reports the omitted count, and the checkpoint retains all experiments.
Scoped recall can add historical evidence through the existing private retrieval tool.
The [A18 episode index](introspection.md) resolves episode IDs against the same frozen source grant.
This view excludes managed-run receipts and records that evidence limit explicitly.

## Typed evidence and prompt rendering

[`workflows.reflection.schemas.ReflectionContext`][] holds the frozen evidence grant.
Ledger metadata, source records, episodes, and observation grants have named models.
The host passes an explicit [`pm.memory.snapshot.Snapshot`][] and [`pm.memory.write_batch.WriteBatch`][] to lifecycle functions.
Stored dictionaries are decoded at the persistence boundary. Internal logic uses typed attributes.

The trigger sees reflection status, not the complete record grant.
The reflection calls use the shared [context selector and renderers](llm-context.md).
Required experiment evidence stays together. Later calls retain the sources and episodes cited by earlier stages.
Optional groups use existing urgency and recency signals.
Each call records included and omitted groups, source IDs, episode IDs, and its conservative token estimate.
A model cannot cite an omitted source as supplied evidence.
The architectural view retains source content, actual memory inputs, uncertainty, and recorded execution details.
Runtime source-version maps and dispatch payloads remain on disk.

## Permitted owners

| Owner | Accepted change | Real consumer |
| --- | --- | --- |
| Attention | A new temporary advice record through the existing attention owner | Attention scoring and competition |
| Belief | A new subjective interpretation through the existing belief owner | Main context selection and later cognition |

Attention advice names promoted or reduced source IDs and a bounded strength.
The operator's `maximum_focus_bias` caps its effective strength.
Existing danger and direct-stimulus rules still apply.
The default proposal duration is 900 seconds, with a schema maximum of 86,400 seconds.
Expiry removes its active influence but preserves its history.

Belief changes supersede an active interpretation and retain both versions.
Attributed user statements and known facts cannot become reflection targets.
The original observations remain unchanged.
The revised belief retains evidence, counterevidence, relationship scope, and affective meaning.

This implementation exposes belief and temporary attention changes.
Source rewriting and arbitrary configuration mutation remain outside its contract.
Goals continue through ordinary private proposals and main acceptance.
A19B still owns consolidation and the removal of duplicate needs representations.

## Effective versions and limits

Main cognition adds effective beliefs and their original sources to its context candidates.
Its context prompt receives each experiment's validity scope.
Attention receives active advice through its existing reader.
The accepted action-selection record stores `reflection_versions` when the relevant owner version reaches the decision.

A version receipt establishes consumption.
It does not establish a useful reply, public delivery, completed external work, or a correct causal diagnosis.
The [lifecycle guide](reflection-lifecycle.md) explains observation checks, reversal, and recovery.

## Code links

- [`workflows.reflection.schemas.ExperimentPlan`][]
- [`workflows.reflection.experiments.context`][]
- [`workflows.reflection.experiments.stage_decision`][]
- [`workflows.reflection.experiments.effective_inputs`][]
- [`workflows.inner_thought.stages.reflection.steps.run`][]
- [`workflows.cognitive_control.attention.stage_focus`][]
- [`workflows.main_cognition.cycle_commits.revise_reflected_belief`][]
