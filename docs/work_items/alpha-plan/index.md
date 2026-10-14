# Companion alpha implementation plan

## Summary

This proposed second plan contains 20 numbered work items, follow-up items, and seven manual milestones.
It develops persistent cognition, LLM classification, adaptive Minecraft work, and source-aware reflection.
Each item defines a handoff for a fresh coding session.

Parent: [engineering manual](../../index.md).
Earlier delivery: [W0–W8 plan](../plan.md).

## Current development priority

The owner prioritizes companion quality over conversation latency during development.
The next sequence is A17, A18, A19, then companion acceptance in A20.
Large Minecraft acceptance remains deferred.
Each item still verifies the actual interfaces it uses.
Read [companion development priorities](companion-development.md) before implementation.
The [curated cognitive sources](cognitive-sources.md) extract useful historical mechanisms and identify excluded ideas.

## Delivery and model policy

A01 supplies the [baseline and session handoff](baseline/index.md).
A02 supplies the [optional typed decision client](../../subsystems/decisions/index.md).
A03 supplies [scoped input bindings and version evidence](../../subsystems/workflow-inputs/index.md).
A04 supplies [classification and scoped INVALID recovery](../../subsystems/workflow-inputs/classification.md).
A05 supplies [PM connections and M1 evidence](../verification/a05-pm-classification.md).
A06 supplies [accepted cognition state](../../subsystems/cognition-state.md) and its
[A06 verification](../verification/a06-cognition-state.md).
A07 supplies [emotional dynamics](../../subsystems/emotional-dynamics.md) and its
[A07 verification](../verification/a07-emotional-dynamics.md).
A08 supplies [experienced memory](../../subsystems/experienced-memory.md) and its [verification](../verification/a08-experienced-memory.md).
A09 supplies [public dialogue](../../subsystems/public-dialogue.md), [publication recovery](../../subsystems/public-publication.md), and [verification](../verification/a09-public-dialogue.md).
Its strict claim review and clinical recovery produced poor, slow replies.
A story-speech prompt and a fixed hallucination policy replaced them.
A09R supersedes that policy with an agentic, persona-driven voice and a working private ontology.
A10 supplies the [sustained Minecraft actor](../../subsystems/minecraft-actor.md),
[body controls](../../subsystems/minecraft-actor-controls.md), and [verification](../verification/a10-minecraft-actor.md).
A11 supplies [durable Minecraft projects](../../subsystems/minecraft-tasks.md),
[task evidence](../../subsystems/minecraft-task-evidence.md), and [verification](../verification/a11-minecraft-projects.md).
A12 supplies [body recovery](../../subsystems/body-recovery.md), [durable publication](../../subsystems/recovery-publication.md),
and [real interruption evidence](../verification/a12-body-recovery.md).
A13 implementation is delivered at the operator's request; its remaining behavioral acceptance is deferred.
The best working state includes the completed house, self-crafted iron pickaxe, verified entry and exit, and saved home waypoint.
The revised return-route gate and remaining fresh acceptance cases have no completion claim.
A09R engineering repairs and real OpenRouter gates are recorded in the [current delivery](../verification/a09r-openrouter-delivery.md).
Human M2 review remains pending.
Conversation latency remains measured but is not a current acceptance threshold under the owner direction.
A14's cooperation mechanisms are implemented, with incomplete behavioral acceptance.
Its [evidence](../verification/a14-cooperation.md) retains successful away activity and failed or unverified gameplay cases.
A15's codelet scheduler and attention competition are implemented.
Its [verification](../verification/a15-codelets.md) retains remaining checks.
A16's grounded codelet repertoire, pathways, and proposal consumers are implemented.
Its [verification](../verification/a16-codelets.md) retains the human review and remaining live checks.
A17 through A19 remain planned.
A20 will verify the revised companion scope while retaining deferred gameplay status.
A14B supplies [direct memory tools](../../subsystems/agent-memory.md) for native decisions.
Its [verification](../verification/a14b-agent-memory.md) keeps the larger operator-designed test pending.
A13's [gate review](../verification/a13/task-gate-review.md) records the pending real acceptance and implemented workflow split.

The LLM is the default for selection, classification, and interpretation.
Laya was planned for frequent use, but A02 found poor interpretation quality.
Retain it only for the simplest classifications after realistic live-endpoint unit tests pass.
If making Laya work requires expensive effort, use the normal LLM.
The [model policy](model-policy.md) governs every work item and withdraws the earlier 95% target.
Historical routing-label review remains pending.
The operator completed A09's sampled M2 review; earlier baseline ratings remain separate.

## Target behavior

The companion has one continuous identity across conversation, private thought, memory, reflection, and Minecraft.
Its choices follow persistent goals, personality, relationships, and changing emotional state.
It can start useful activity within its configured action space.
It can stop, change its mind, recover, and discuss uncertainty.

Goals are open-ended within its configured action space.
House, diamond, farm, and mischievous game goals are examples, rather than hardcoded task categories.
Appraisal can lead it to accept or refuse a request.
Its inner loop can initiate work for its own reasons.
Simple tool questions can use Assistant dialogue directly.

Immersion depends on consequences and continuity.
A sentence about sadness has little value if sadness affects no later choice.
A statement about self-reflection has little value if reflection changes no belief or behavior.
A task announcement has little value without subsequent work and evidence.

Perfect sentience simulation and zero unwanted replies are aspirations.
This alpha evaluates concrete behavior and records the user's wish to regenerate each reply.
It does not claim to prove consciousness or reproduce every human emotion.

## Generalization ladder

Alpha uses inspectable deterministic rules for attention selection, routing, and public review.
The rules are simple, testable, and easy to diagnose.
Beta replaces a rule with an agentic equivalent when a tested reason exists.
The rule stays as the regression baseline.
A replacement must match or beat the rule-based behavior on the same cases.
This method turns explicit logic into a deliberate alpha stage, rather than a defect.

Apply the ladder in this order for each rule:

1. Record the rule and the cases that exercise it.
2. Keep the rule and the test as the baseline.
3. Build the agentic candidate behind the same interface.
4. Compare the two on the recorded cases and on new cases.
5. Keep the rule when the candidate is not better.

Read the [model policy](model-policy.md#generalization-ladder) and the [evaluation contract](evaluation.md).

## Beta idea items

The [beta ideas](../../work_items/beta-ideas/index.md) hold directions that are out of alpha scope.
They are rough notes, not plans.
After A20, promote the useful ones into proper beta work items.

## Read the plan

| Document | Purpose |
| --- | --- |
| [Current state](current-state.md) | What exists, what is disconnected, and what was verified |
| [Previous attempts](previous-attempts.md) | What to preserve, adapt, and leave behind |
| [Cognitive architecture](cognition.md) | State, emotion, codelets, internal thought, and self-reflection |
| [Companion development priorities](companion-development.md) | Quality first, revised dependencies, and deferred large gameplay acceptance |
| [Curated cognitive sources](cognitive-sources.md) | Selected paper passages and historical code with explicit reuse boundaries |
| [Felt-state projections](felt-state.md) | A17 channel measurements, reportability, and stable personal vocabulary |
| [Private ontology](ontology.md) | Capability-consistent expression and protection against polluted memory |
| [Model selection and reliability](model-policy.md) | Endpoint and capability requirement, the generalization ladder, and optional decision-classifier tests |
| [Models and workflow inputs](decisions.md) | Public pm-coder client, file selection, classification, and INVALID fallback |
| [Minecraft architecture](minecraft.md) | Task trees, specialist sessions, recovery, and shared body ownership |
| [Milestones and verification](verification.md) | Real manual tests and acceptance evidence |
| [Decision register](decision-register.md) | The user's confirmed architecture and behavior choices |

## Ordered delivery

| Order | Work item | First manual milestone |
| --- | --- | --- |
| A01 | [Capture baseline and evaluation scenarios](work-items/a01.md) — engineering delivery recorded | Preparation |
| A02 | [Provide a public pm-coder decision client](work-items/a02.md) — engineering delivery recorded | |
| A03 | [Bind files, folders, and named outputs](work-items/a03.md) — engineering delivery recorded | |
| A04 | [Add classification with INVALID fallback](work-items/a04.md) — engineering delivery recorded | Engine demonstration |
| A05 | [Wire LLM classification in PM and retain optional Laya](work-items/a05.md) — engineering delivery recorded | M1: LLM classification over real artifacts |
| A06 | [Commit affect, goals, and causal events](work-items/a06.md) — engineering delivery recorded | |
| A07 | [Make emotional state change behavior](work-items/a07.md) — engineering delivery recorded | |
| A08 | [Retrieve and learn from experienced memory](work-items/a08.md) — engineering delivery recorded | |
| A09 | [Unify public voice and capability grounding](work-items/a09.md) - engineering delivery recorded | M2: a continuous companion across restart |
| A09R | [Agentic public voice and the private ontology](work-items/a09r.md) — engineering repairs and real gates recorded; human review pending | Redo of A09's grounding mechanism |
| A10 | [Run one sustained Minecraft specialist session](work-items/a10.md) - engineering delivery recorded | Developer check: many calls per leaf |
| A11 | [Plan and verify durable Minecraft subtasks](work-items/a11.md) — engineering delivery recorded | |
| A12 | [Control and recover interrupted body work](work-items/a12.md) — engineering delivery recorded | |
| A13 | [Replan expected gameplay failures](work-items/a13.md) | M3: finished small house and self-crafted iron pickaxe |
| A14 | [Play as a cooperative survival partner](work-items/a14.md) | M4: house, expedition, and changed conditions |
| A14B | [Use editable drafts and scoped memory tools](work-items/a14b.md) — engineering implemented, larger test pending | Measured workflow efficiency, repair, and direct memory access |
| A15 | [Run selective codelets through the workspace](work-items/a15.md) — engineering implemented, remaining checks recorded | |
| A16 | [Deliver a grounded codelet repertoire](work-items/a16.md) — engineering delivery recorded | |
| A17 | [Run the second conscious loop](work-items/a17.md) | M5: observable private activity with consequences |
| A18 | [Inspect the implementation and update the self-model](work-items/a18.md) | |
| A19 | [Reflect, test hypotheses, and revise behavior](work-items/a19.md) | M6: source-grounded self-reflection |
| A20 | [Verify the integrated alpha in sustained use](work-items/a20.md) | M7: a mixed-session alpha trial |

The order supports useful stopping points.
A10 can start after A04 and A09, while later cognitive work remains pending.
Individual coding sessions must respect each item's explicit dependencies.
No dependency requires enabling Laya or making a failed Laya candidate pass.
LLM transport, context handling, and agent compaction failures require immediate repair in the affected item.

## Session handoff

Read this page, the item's linked design, and the item's source pointers.
Inspect current repository instructions and uncommitted changes before editing.
Resolve item-specific interface details before writing code.
Record repository revisions, changed contracts, commands, results, and remaining limits in the handoff.

Use real model and world endpoints for behavioral acceptance.
An unavailable endpoint leaves that live gate unverified.
Keep implementation scope inside the selected item.
A failed gate requires a concrete diagnosis before an item depends on the affected mechanism.
Deferred large gameplay acceptance does not block unrelated cognitive work under the current owner direction.

These are delivery work packages.
A complex package may need more than one coding session.
The plan does not promise that all 20 packages fit one weekend.
