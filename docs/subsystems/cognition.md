# Cognitive pipeline

## Summary

The parent workflow runs seven stages in a fixed order.
Later stages receive earlier reports.
The module accepts the resulting attention frame and GWT before it dispatches typed actions.

Parent: [system map](../concepts/architecture.md).
Entry: [`workflows.main_cognition.pipeline.run_cycle`][].

## Input, change, and output

The request contains stimuli, memory candidates, user context, and available workflows.
The stages produce reports and action proposals.
The parent collects those proposals into a typed cognitive report.
The application module commits accepted state and publishes action envelopes.

| Stage | Main responsibility | Implementation |
| --- | --- | --- |
| Triage | Classify the incoming situation | [`workflows.main_cognition.stages.triage.steps.run`][] |
| Appraisal | Evaluate the situation | [`workflows.main_cognition.stages.appraisal.steps.run`][] |
| Context | Select supplied memory candidates | [`workflows.main_cognition.stages.context.steps.run`][] |
| Attention | Select material for the frame | [`workflows.main_cognition.stages.attention.steps.run`][] |
| Goals | Produce goal proposals | [`workflows.main_cognition.stages.goals.steps.run`][] |
| Workspace | Form the workspace proposal | [`workflows.main_cognition.stages.workspace.steps.run`][] |
| Action selection | Select typed actions | [`workflows.main_cognition.stages.action_selection.steps.run`][] |

## Asynchronous boundary

[`workflows.main_cognition.engine.build_stage_runner`][] invokes each child through the shared kernel runner.
The child does not need its own module thread or bus subscription.
[`modules.MainCogLoop`][] connects this pipeline to the bus and Ghost.

Story dialogue produces character speech.
Assistant dialogue can call scoped tools.
SelfManagement operates managed runs.
These paths have separate contracts and prompts.

## State and side effects

Model calls occur outside the Ghost commit lock.
Accepted cognition records reach the checkpoint before action publication.
Dialogue also stores accepted records before public replies.
Real internal thought remains disabled.

## Traps

Context re-ranks supplied candidates.
The main cognitive candidate list does not yet perform embedding retrieval.
A scheduled action does not prove that an external task succeeded.
Parent workflow timings include child timings.

## Go deeper

Read [emotional dynamics](emotional-dynamics.md) for the bounded affect step.
Read [accepted cognition state](cognition-state.md) for affect, goals, and
stable action identities.
Read [A06 verification](cognition-state-verification.md) for the real state
receipts.
Read [one exchange](../workflows/exchange.md), [managed work](managed-work.md),
and [next experiments](../decisions/experiments.md).

