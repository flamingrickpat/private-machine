# Companion alpha implementation plan

## Summary

This proposed second plan contains 20 ordered work items and seven manual milestones.
It develops persistent cognition, LLM classification, adaptive Minecraft work, and source-aware reflection.
Each item defines a handoff for a fresh coding session.

Parent: [engineering manual](../../index.md).
Earlier delivery: [W0–W8 plan](../plan.md).

## Delivery and model policy

A01 supplies the [baseline and session handoff](baseline/index.md).
A02 supplies the [optional typed decision client](../../subsystems/decisions/index.md).
A03 supplies [scoped input bindings and version evidence](../../subsystems/workflow-inputs/index.md).
A04 supplies [classification and scoped INVALID recovery](../../subsystems/workflow-inputs/classification.md).
A05 supplies [PM connections and M1 evidence](../../subsystems/workflow-inputs/pm-verification.md).
A06 through A20 remain pending.

The LLM is the default for selection, classification, and interpretation.
Laya was planned for frequent use, but A02 found poor interpretation quality.
Retain it only for the simplest classifications after realistic live-endpoint unit tests pass.
If making Laya work requires expensive effort, use the normal LLM.
The [model policy](model-policy.md) governs every work item and withdraws the earlier 95% target.
Historical routing-label review remains pending, and human immersion ratings still need operator review.

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

## Read the plan

| Document | Purpose |
| --- | --- |
| [Current state](current-state.md) | What exists, what is disconnected, and what was verified |
| [Previous attempts](previous-attempts.md) | What to preserve, adapt, and leave behind |
| [Cognitive architecture](cognition.md) | State, emotion, codelets, internal thought, and self-reflection |
| [Private ontology](ontology.md) | Capability-consistent expression and protection against polluted memory |
| [Model selection and reliability](model-policy.md) | LLM default, optional Laya test requirements, and immediate LLM repairs |
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
| A06 | [Commit affect, goals, and causal events](work-items/a06.md) | |
| A07 | [Make emotional state change behavior](work-items/a07.md) | |
| A08 | [Retrieve and learn from experienced memory](work-items/a08.md) | |
| A09 | [Unify public voice and capability grounding](work-items/a09.md) | M2: a continuous companion across restart |
| A10 | [Run one sustained Minecraft specialist session](work-items/a10.md) | Developer check: many calls per leaf |
| A11 | [Plan and verify durable Minecraft subtasks](work-items/a11.md) | |
| A12 | [Control and recover interrupted body work](work-items/a12.md) | |
| A13 | [Replan expected gameplay failures](work-items/a13.md) | M3: finished small house and self-crafted iron pickaxe |
| A14 | [Play as a cooperative survival partner](work-items/a14.md) | M4: house, expedition, and changed conditions |
| A15 | [Run selective codelets through the workspace](work-items/a15.md) | |
| A16 | [Deliver a grounded codelet repertoire](work-items/a16.md) | |
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
A failed gate requires a concrete diagnosis before dependent items proceed.

These are delivery work packages.
A complex package may need more than one coding session.
The plan does not promise that all 20 packages fit one weekend.
