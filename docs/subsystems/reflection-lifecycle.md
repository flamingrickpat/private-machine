# Experiment lifecycle and recovery

## Summary

An experiment uses one durable Feature record with schema `pm.reflection-experiment.v1`.
Its metadata retains the plan, source revisions, previous state, effective version, observations, result, and ordered history.
The private host commits each transition atomically with any owned change.
No transition erases the original prediction.

Parent: [private reflection](reflection.md).
Related: [accepted cognition state](cognition-state.md) and [private-loop recovery](private-loop.md).

## States and transitions

| State | Meaning |
| --- | --- |
| `proposed` | The prediction and reversal state exist. A later step can apply or interrupt it |
| `applied` | The owner accepted a new version |
| `awaiting_evidence` | No relevant outcome establishes a result |
| `evaluated` | Later evidence supports, contradicts, or leaves the prediction unresolved |
| `retained` | Support from at least two later observation ticks justifies scoped retention |
| `reversed` | A new owner version restores the captured previous state |
| `revised` | A new experiment replaces the hypothesis while the old history remains |
| `interrupted` | Work stopped with its actual effective version recorded |

`interrupt` does not undo an applied intervention.
The effective owner state remains authoritative until expiry, replacement, or explicit reversal.
Missing evidence and missing user replies cannot produce a successful result.
An unresolved experiment can remain in the ledger across restart.

## Commit and concurrency

The proposal records every cited source revision.
Application checks those revisions again at the real memory commit boundary.
A changed source rejects the complete batch through `GhostConflict`.
The experiment remains proposed, and the rejected private invocation retains its diagnostic.
The model can reassess or interrupt stale work in a later cycle.

Attention uses an append-only record family with a metadata guard.
The guard checks both membership and revisions, including the initially empty family.
Two concurrent first advice writes cannot both succeed.
A newer accepted attention change prevents application or reversal based on an older head.

Belief revision checks the prior record and supersedes it through the existing belief owner.
A concurrent belief revision prevents a stale replacement.
The caller never changes an original user report to make an interpretation fit.

## Observations and retention

Evaluation needs new accepted records after the intervention's application tick.
Original user reports, actual runtime events, world observations, and recorded consuming decisions can supply evidence.
An external outcome also needs an explicit reference to a consuming decision.
Time order alone cannot establish that connection.
An imagined concern or repeated interpretation cannot establish an outcome.
A supported or contradicted prediction also needs a later decision receipt with the experiment's exact effective version.

The request exposes `observation_grant` for each experiment.
This grant lists eligible outcome IDs and the consuming decisions that support their provenance.
Source rows also expose actual reference IDs and evidence roles.
The guide validates these constraints before host acceptance and can repair one invalid proposal.
The host repeats the check against its frozen view and guards the real commit.

The model still decides whether the observation discriminates the recorded prediction.
The host's checks establish provenance and ordering, not semantic correctness.
Human review remains necessary for ambiguous interpretations and persona quality.
Retention needs supported observations from at least two distinct later ticks.
Two records from one choice therefore cannot become two successful trials.

## Reversal and restart

Reversal restores the captured advice or belief as a new version.
It preserves the experiment's result and failure history.
Restored attention advice keeps its original absolute expiry, so reversal cannot extend its duration.
Expired advice stays inactive.
An empty previous advice state produces an inactive restoration record.

A revised experiment retains the old prediction and links its replacement experiment.
The replacement starts as a new proposal and needs a separate application commit.
Revision alone does not undo an applied intervention.
The owner head must still match before the replacement can apply.

The existing checkpoint stores the experiment and its owned change in one atomic replacement.
Cancellation before that replacement accepts neither change.
Cancellation after a committed proposal leaves that proposal available for inspection.
Restart reads the stored lifecycle and never invents a successful outcome.

## Inspect a run

1. Open the run's `receipt.json` and inspect its declared scope and failures.
2. Inspect the source snapshot and installed package origins in `source.json`.
3. Open the checkpoint and find the experiment's schema and record ID.
4. Compare `source_revisions`, `previous`, `effective_record_id`, `observation_ids`, and `history`.
5. Find later action-selection records with the matching `reflection_versions` entry.
6. Inspect the child kernel's `assessment.json` and rejected model proposals.

The gate introduces an explicit mistaken interpretation as a fixture.
Its receipt distinguishes that fixture from a model-produced baseline failure.
Its later choices establish application consumption, with public delivery and human quality outside its scope.
The [live-evidence page](../work_items/verification/a19/live-evidence.md) gives exact commands and evaluation recovery steps.

## Code links

- [`workflows.reflection.schemas.ReflectionDecision`][]
- [`workflows.reflection.experiments.stage_decision`][]
- [`workflows.reflection.experiments.observation_grant`][]
- [`pm.memory.write_batch.WriteBatch.expect_metadata_version`][]
- [`workflows.main_cognition.stage_commits.subjective_belief`][]
