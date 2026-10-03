# Minecraft evidence, memory, and cleanup

Parent: [Minecraft run guide](minecraft.md).
Related: [Ghost](../ghost.md) and [earlier plan](../decisions/plan.md).

## Three different storage areas

| Area | Owner | Lifetime and purpose |
| --- | --- | --- |
| Managed task directory | Spamuel supervisor | Unique run evidence; retained after terminal state |
| MCP character directory | Minecraft MCP | Persistent notes, waypoints, skills, drafts, frames, and body log |
| Minecraft world directory | Minecraft server | Blocks, player state, and other world data |

The supervisor normally selects `<task_root>/<run_id>`.
For a console instance, the application runtime root is instance-specific.
A run therefore gets a fresh task directory without clearing the previous run.
A caller can override `artifact_path`; reusing that path defeats the default isolation.
Kernel workspaces and journals also use run IDs.

The MCP character directory is different.
Its default is `C:/source/pm/pm-minecraft/agent-data`, unless `MCMCP_AGENT_HOME` overrides it.
A new managed run does not change or clear that directory.
Deleting a task folder does not reset the MCP character or the Minecraft world.

## How results enter Ghost

The Minecraft role writes these artifacts:

- `scratch/observations.md`: observations before and after the action.
- `outbox/report.yaml`: goal, action results, evidence references, and acceptance result.
- `outbox/world.yaml`: structured places, objects, and a body episode.
- `screenshots/`: image files from MCP observations.

`main.build_app` connects the supervisor terminal callback to [`runtime.consolidation.Consolidator`][].
This callback also handles failed and cancelled runs when evidence exists.
It inventories recognized top-level text artifacts and image directories.
It computes a run-and-evidence key, then proposes memories through a Ghost batch.

The current mapping is:

| Evidence | Accepted memory |
| --- | --- |
| Report summary | Private internal world-event Feature |
| Observation text | Private internal world-event Feature |
| Structured places | WorldPlaceKnoxel |
| Structured objects | WorldObjectKnoxel |
| Body episode and screenshots | BodyEpisodeKnoxel with image evidence |
| `scratch/result.md`, when present | Reportable inner-monologue Feature for reusable experience |

The Minecraft role does not normally produce `scratch/result.md`.
Consolidation is code-driven. It is not an extra LLM reflection or a learned survival policy.
It does not import all files from the MCP character directory.
MCP notes and installed skills do not automatically become Ghost knowledge.

Ghost validates references and durably commits memories with the consolidation receipt.
Repeated consolidation of the same run and evidence reuses that receipt.
Changed evidence produces a different key.
Images remain evidence metadata, not embedding text.
These memories become available to later Ghost readers.
They do not guarantee an immediate chat reply or that cognition retrieves every new record.

A terminal run receipt and a consolidation receipt are separate evidence.
The supervisor currently catches terminal-callback exceptions to preserve the run receipt.
Thus, **completed** alone does not prove that consolidation succeeded.
Check the Ghost consolidation ledger before you archive task evidence.
This missing visible consolidation-failure state is an implementation gap.

## Cleanup policy

Normal runs do not automatically archive or delete their task folders.
Retain failed and cancelled evidence for inspection.
Archive a successful task only after its durable consolidation receipt exists.
Do not delete task evidence while its worker or consolidation is active.
The first delivery deliberately excludes automatic permanent deletion.

Do not erase the persistent MCP character directory between ordinary runs.
It contains durable waypoints, notes, and installed skills as well as temporary files.
A blanket reset would destroy useful character state.
For independent experiments, configure a separate `MCMCP_AGENT_HOME` before starting a separate MCP process.
That isolates character files; it does not isolate the world or serialize multiple body connections.

The MCP exposes `/test/reset` only in test mode.
That reset removes memory files and drafts, preserves installed skills, and clears selected in-memory test state.
It does not reset the world or clear every frame and log.
It is not a production per-workflow reset API.

## What the original plan required

At commit `6415dd0`, `docs/plan/workflows.md` required terminal evidence consolidation, validated Ghost commits, and reusable experience.
It required archival only after durable memory storage succeeded.
It required failures to preserve evidence and repeated consolidation to avoid duplicate memories.
It explicitly excluded automatic permanent deletion from the first delivery.
W8 delivered a bounded Minecraft body check, not sustained house-building.

That plan does not specify an implemented per-run MCP process or character-directory reset protocol.
The current architecture satisfies task-directory isolation, but not per-run isolation of MCP-owned files.
A future multi-action workflow needs verified goal postconditions, repeated instruction and cancellation boundaries, and bounded action batches.
If it writes temporary MCP files, it also needs a run-scoped staging area and an explicit import receipt before cleanup.
Persistent character memory and run scratch must have separate ownership.
These are proposed next steps, not current automatic behavior.
