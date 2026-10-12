# Sustained Minecraft specialist

## Summary

One pm-coder session performs a meaningful Minecraft leaf through many real tool calls.
A scoped gateway holds the avatar lease and records every dispatched call.
Instruction updates invalidate the next planned call.
The configured text model uses numeric observations and retains screenshots as separate evidence.

Parent: [managed work](managed-work.md).
Operation: [actor controls](minecraft-actor-controls.md).
Evidence: [A10 verification](../work_items/verification/a10-minecraft-actor.md).
Projects: [durable subtasks](minecraft-tasks.md).

## Request and execution

[`main.build_app`][] configures the `minecraft` managed workflow.
[`pm.runs.supervisor.WorkflowSpec`][] supplies its manifest, tool factory, optional agent model, and MCP descriptors.
[`pm.runs.supervisor.ManagedRunSupervisor.start`][] commits the accepted receipt before it starts the worker thread.
The worker creates its environment, cancellation event, and instruction callback.
Its child runtime uses a separate `children/<scope>` directory and inherits cancellation and tools.
Nested scopes retain the agent model and MCP descriptors beneath their parent directory.
Child scope validation rejects paths outside the task directory.
Trusted Python roles retain their existing MCP configuration.

[`workflows.minecraft.schemas.MinecraftRequest`][] defaults to `mode: actor` and `specialist: gatherer`.
The other specialist values are `navigator`, `recovery`, `builder`, and `miner`.
The request contains `instruction`, `goal`, `body_id`, `allowed_tools`, `include_image`, `persona_context`, and `acceptance`.
Project children also carry `task_reference` with the selected leaf, specification revision, and actor attempt.
The supervisor supplies the current companion identity, character, affect, goals, ontology, and selected public context.
This context preserves attribution and the real imported history.
It is a scoped snapshot, not the entire history.
The actor can also read current Minecraft memories through the MCP.

[`workflows.minecraft.steps.run`][] selects the sustained path.
Explicit `action_tool` requests and the old instruction-wait fixtures select the retained W8 boundary check.
`mode: boundary_check` also selects that path.
The request has no `max_actions` field or workflow tool-call ceiling.

[`workflows.minecraft.specialist_session.run_specialist_session`][] acquires the lease, observes the body, and discovers upstream tool schemas.
It starts a temporary loopback MCP gateway and passes its descriptor to `PmCoderDriver.run_session`.
The model comes from the supplied `agent_model` or the lazy `Default` model session.
The driver uses pm-coder's native session, serving context, history, and automatic compaction.
Its existing session deadline remains 7,200 seconds.
This deadline is independent of the number of tools or actions.

## Specialist resources

[`dev.tools.compile_minecraft_skill.compile_skill`][] copies selected upstream sections into project resources.
It leaves `C:/Users/rick/.agents/skills/minecraft-survival-coach/SKILL.md` intact.
`workflows/minecraft/resources/source.json` records its digest and each compiled resource digest.
Local builder guidance lives in `workflows/minecraft/guidance/builder.md`.
The compiler appends it and records its separate source digest.
Recompilation preserves that guidance without changing the installed upstream skill.
Git preserves LF line endings for these resources so their byte digests survive a fresh checkout.

| Resource | Upstream sections | Purpose |
| --- | --- | --- |
| `shared.md` | 1–9, 36–37, 39–42, 49–50 | Evidence, priorities, schemas, memory, and action review |
| `navigator.md` | 10–13, 26 | Geometry, travel, scouting, and return |
| `recovery.md` | 14–18, 33–35 | Hazards, health, food, sleep, and recovery |
| `gatherer.md` | 19–23, 27–29 | Resources, inventory, crafting, smelting, and mining |
| `builder.md` | 24–25 | Shelter and storage |
| `miner.md` | 30–32 | Reversible mining, caves, iron, and diamonds |

The session receives the shared resource, navigation resource, and selected specialist resource.
It receives no example drills or complete upstream prompt.
The specialist belongs to the same companion and sends no public reply.
The public dialogue path owns speech to the user.

## Tools and visual evidence

The gateway retains current pm-minecraft tool names, descriptions, and argument schemas.
An explicit `allowed_tools` list narrows those tools.
The default survival grant excludes `minecraft_suicide` and `minecraft_execute_typescript`.
Scripts can hide multiple actions from the current instruction boundary.
Operator test routes remain outside the actor grant.
The `actor_instructions` tool reports the latest instructions and lease owner without a world call.

[`workflows.minecraft_body.gateway.BodyGateway`][] stores original results and screenshot paths in its journal.
Image bytes never enter text observations or embeddings.
The active actor sets visual interpretation to false.
The gateway can return MCP image blocks through its explicit `visual` capability.
That transport branch has no live multimodal acceptance in A10.
Changing a request cannot establish that its model supports images.

## Completion evidence

[`workflows.minecraft.specialist_session.ActorOutcome`][] contains `status`, `summary`, and `handoff`.
Its status is `complete`, `yield`, `blocked`, or `replan`.
This model output is a candidate, not independent proof of task completion.

The A10 `acceptance.kind: wood_return` verifier uses original tool receipts and a fresh final observation.
It requires distinct positive tree harvests, retained logs, meaningful travel, return, dimension continuity, and a living grounded body.
The default values require five trees, travel of at least 16 blocks, and return within three blocks.
[`workflows.minecraft.specialist_session.verify_wood_return`][] supplies these measured checks.
General task-tree verification belongs to A11.
Other leaves retain their candidate and artifacts without an accepted completion claim.

The final report uses `pm.minecraft-actor-report.v1`.
It includes the goal, body label, outcome, verification, session reference, revision, call count, and both observations.
It also includes the journal path, screenshot paths, and rejection reason.
The report retains its acceptance specification and independent `verified` flag.
Before public task dispatch, [`workflows.public_speech.reply_recovery.task_evidence`][] recomputes wood-return checks from the retained journal.
This check establishes recorded task effects, not a later unchanged world.
The role writes `outbox/handoff.md` and `outbox/world.yaml` beside the report.
If the native session already wrote its handoff, the host preserves that document.
The outcome's `handoff` string can reference the file and must not overwrite its contents.
The supervisor marks a rejected report as failed and retains its evidence.
Read [controls and failure recovery](minecraft-actor-controls.md) before changing ownership or dispatch.
