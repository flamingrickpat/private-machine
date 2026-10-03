# Minecraft body and world operation

## Summary

One actor controls the shared body.
Interruptions retain actual effects and require fresh evidence.
Expected gameplay failures drive recovery and replanning.
The world configuration determines survival mechanics.
The companion remains self-directed throughout its configured action space.

Parent: [Minecraft architecture](minecraft.md).
Delivery: [A10](work-items/a10.md) through [A14](work-items/a14.md).

## Body ownership and session lifetime

A10 establishes a single body mutation lease before any sustained actor runs.
Do not allow parallel navigation, mining, crafting, or interaction against the same avatar.
Read-only planning can run alongside the actor.
Observation freshness and any tool-induced effects still require explicit handling.

A12 makes ownership durable across disconnects and worker restarts.
Use session epochs and fresh observation to reject stale commands.
Cancellation cannot undo an action already sent to Minecraft.
The next action must respect the latest accepted instructions.

A pm-coder session continues until its leaf succeeds, yields, or needs replanning.
Context compaction retains the goal, evidence, map, and current method.
A new session can resume the same leaf with a durable handoff.
Do not force a new leaf after an arbitrary number of calls.

Instruction changes, hazards, cancellation, and checkpoint opportunities create natural boundaries.
The actor checks these boundaries between actions.
An immediate body hazard can interrupt slower cognitive reasoning.

## Outcomes and recovery

Distinguish these conditions:

| Condition | Response |
| --- | --- |
| Tool transport unavailable | Preserve state and wait for the service |
| Tool reports an unsuccessful game action | Diagnose the concrete precondition or obstruction |
| Leaf makes partial progress | Record the delta and continue |
| Leaf acceptance may be satisfied | Verify through independent observation |
| Current method makes no progress | Change a relevant condition or replan |
| Body is endangered | Recover before resuming the larger goal |
| Goal needs a user decision | Ask once with the retained task context |
| Goal is impossible under current permissions | Report the specific boundary and alternatives |

Failed searches record the local area, method, observation time, and result.
“No ore found here” does not become “this world has no ore.”
Two ineffective repetitions of the same method require diagnosis or a relevant change.
Recovery remains active until its evidence proves the hazard has ended.

Unexpected death creates a recovery goal.
The system tracks lost items, location uncertainty, equipment, and feasible return routes.
A replanner can abandon an unsafe recovery when new evidence justifies that choice.

## Cozy world and cooperation

The confirmed first environment is `pm-agentic-babymode`, with isolated acceptance worlds.
Its survival mechanics differ from vanilla.
Natural regeneration is disabled.
Nutrition and sleep affect state.
Inventory handling can differ during mining.

Inspect real configuration and diagnostic status.
Do not assume full food implies full health.
Fixture `/commit` and `/revert` do not fully restore player state or entities.
Keep test-server controls outside the companion's survival tools.

Cooperation includes task division, rendezvous, progress conversation, and recovery assistance.
These are available behaviors, rather than a mandatory agreeable personality.
The companion can choose other permitted goals based on mood, relationships, beliefs, and intention.
While the user is away, it remains self-directed throughout its action space.
The implementation must not insert an unrequested base-area or cozy-project whitelist.
Actor observations support statements such as having seen a cave.
A screenshot only supports a visual claim when a compatible model actually interpreted it.

Body events should enter cognition as meaningful deltas.
Do not run all cognitive stages after every mining call.
A14 verifies both practical progress and the quality of the partnership.
