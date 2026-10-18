# Cognitive pipeline

## Summary

Context selection repairs an invalid memory ID once. If repair still fails, it drops unavailable IDs and rebuilds memory text from supplied sources.
The original rejection remains in the stage artifacts. Current conversation can continue without fabricated memory references.

The main cognition workflow runs its typed phases in a fixed order.
Later stages receive earlier reports.
The module accepts the resulting attention frame and GWT before it dispatches typed actions.

Parent: [system map](../concepts/architecture.md).
Entry: [`workflows.main_cognition.steps.run`][].
Game initiative: [cooperative survival](cooperative-survival.md) and [observed world events](world-events.md).

## Input, change, and output

The request contains stimuli, memory candidates, user context, and available workflows.
One manifest runs the phases intake, triage, appraise, retrieve, codelets, structures, attention, goals, experience, decide, route, and commit.
The phases share one typed cycle state and add their results to it.
The commit phase hands the finished cycle to the application module.
The module commits accepted state in one write batch and publishes action envelopes.

| Phase | Main responsibility | Implementation |
| --- | --- | --- |
| Triage | Classify the incoming situation | [`workflows.main_cognition.phases.triage.run`][] |
| Appraise | Evaluate the situation | [`workflows.main_cognition.phases.appraise.run`][] |
| Retrieve | Select supplied memory candidates and add them to the CSM | [`workflows.main_cognition.phases.retrieve.run`][] |
| Codelets | Run codelets in waves, fill the CSM, and collect nominations | [`workflows.main_cognition.phases.codelets.run`][] |
| Attention | Score proposals, then select winners in Python | [`workflows.main_cognition.phases.attention.run`][] |
| Goals | Produce goal proposals | [`workflows.main_cognition.phases.goals.run`][] |
| Experience | Form the workspace frame and subjective experience | [`workflows.main_cognition.phases.experience.run`][] |
| Decide | Select typed actions | [`workflows.main_cognition.phases.decide.run`][] |

## Asynchronous boundary

[`workflows.main_cognition.host.build_main_cognition_tools`][] grants each run its model, embedding, and agent codelet tools. All phases run in one manifest and share one typed [`workflows.main_cognition.cycle_work.CycleWork`][].
The workflow does not need its own module thread or bus subscription.
[`workflows.main_cognition.host.MainCogLoop`][] connects the cycle to the bus and the memory store.

Story dialogue produces character speech.
Assistant dialogue can call scoped tools.
SelfManagement operates managed runs.
These paths have separate contracts and prompts.
They share the [public context and grounding contract](public-dialogue.md).
Accepted public speech uses [durable publication and recovery](public-publication.md).

## State and side effects

Model calls occur outside the Ghost commit lock.
Accepted cognition records reach the checkpoint before action publication.
Dialogue also stores accepted records before public replies.
Real internal thought remains disabled.

## Traps

Context selects supplied evidence from the scoped semantic and episodic retrieval pool.
The accepted action and dialogue receive the same selected records and original IDs.
A scheduled action does not prove that an external task succeeded.
Parent workflow timings include child timings.

## Go deeper

Read [codelets and attention competition](codelets.md) for the registry, the waves, the percepts, and the committed nominations.

Read [experienced memory](experienced-memory.md) for retrieval, attribution, and source constraints.

Read [emotional dynamics](emotional-dynamics.md) for the bounded affect step.
Read [accepted cognition state](cognition-state.md) for affect, goals, and
stable action identities.
Read [A06 verification](../work_items/verification/a06-cognition-state.md) for the real state
receipts.
Read [one exchange](../workflows/exchange.md), [managed work](managed-work.md),
and [next experiments](../work_items/experiments.md).
