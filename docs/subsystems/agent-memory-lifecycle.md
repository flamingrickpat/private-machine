# Managed memory lifecycle

## Summary

Each managed planner group receives a fresh memory snapshot and native session.
Repair turns retain the group's conversation.
Source revisions guard its local commit.
Cancellation and session closure stop pending retrieval without cancelling later work in the enclosing run.

Parent: [direct agent memory](agent-memory.md).
Related: [draft sessions](draft-sessions.md), [managed work](managed-work.md), and [A14B evidence](../work_items/verification/a14b-agent-memory.md).

## Workflow grant and group identity

[`pm.runs.workflow_spec.WorkflowSpec`][] has an optional `memory_config_factory`.
Setting this factory to `None` disables the added tools for that workflow.
The supervisor binds it to the authoritative store and the run's cancellation signal.
[`workflows.minecraft.host.project_memory_config`][] grants accepted in-world history in reality `-1`, including preferences, outcomes, and original sources.
It does not filter history by Minecraft words.
Other realities require another workflow grant.

The real companion assembly enables this factory for `minecraft_project`.
Other managed workflows retain their existing paths.
The planner opens a fresh read session for each decision group.
Each group also gets a fresh native session identity.
Repair turns retain that group's conversation and successful edits.
Later groups rebuild from accepted state and explicit evidence instead of replaying earlier planning conversations.

## Evidence and commit

Search and record inspection collect revisions for all inspected dependencies.
The completed-turn validator rejects changed sources with repair feedback.
The planner records `memory.jsonl` and `memory-revisions.json` beside its other decision artifacts.
[`pm.memory.agent_tools.guard_revisions`][] checks these revisions again through the local specification commit.
Unrelated appends do not invalidate unchanged sources.

Managed control acquires its lock before the memory-owner lock.
This order matches supervisor updates.
No model or network request runs inside the commit boundary.
Read tools cannot commit memory, publish speech, start work, or change the world.

## Cancellation and restart

Cancellation rejects pending retrieval and prevents another accepted turn.
The native runner awaits agent cleanup before listener shutdown.
The caller then revokes the read session.
During retrieval, the search detects a closed session, cancels its provider task, and awaits cleanup.
It does not set the enclosing run's cancellation signal.
Completed embedding cache entries can remain.
Transport retries use the shared embedding client's bounded policy: five retries, with 30-second pauses.
Cancellation interrupts retry waits and request I/O.
Lexical queries perform no network work.

Restart creates a fresh snapshot and listener.
Unaccepted planner edits remain diagnostic artifacts and are not automatically restored.
Old grants cannot authorize another group.

## Diagnose a source conflict

1. Read the group's `memory.jsonl` and `memory-revisions.json`.
2. Inspect the cited original records and their current revisions.
3. Use `memory_refresh` through the current group.
4. Reconsider the affected draft before another attempt.

The [verification record](../work_items/verification/a14b-agent-memory.md) distinguishes engineering checks, actual model decisions, and pending world execution.
