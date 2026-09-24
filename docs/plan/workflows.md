# Workflow behavior

## Summary

Each workflow owns its prompts, output schemas, Python steps, and tests.
The first cognitive stages are small but respond to real input through real models.
Fixed character responses remain historical demonstration code only.

Parent: [implementation handoff](../../IMPLEMENTATION_PLAN.md).
Related: [runtime](runtime.md), [memory](models-memory.md), [tests](tests.md).

## Common request and result

Each request carries `run_id`, `correlation_id`, `source_message_ids`, a scoped snapshot reference, and workflow-specific typed input.
An action request also carries its action ID and GWT reference.
References resolve through the supplied view, not unrestricted live Ghost access.

Each result carries a status, summary, artifact references, and workflow-specific typed output.
Model responses enter Pydantic schemas at the boundary.
Required fields have no empty defaults that conceal a missing model result.
Prompts stay in YAML. Schemas and deterministic transformations stay in local Python.

Use ordinary Python-role sequences for simple model calls.
Use pm-coder roles only for behavior that needs agentic file or tool work.
Finalizers validate output files and produce typed proposals or accepted reports.

## Main cognitive workflow

MainCogLoop takes one available input batch and starts one parent workflow.
Incoming stimuli enter memory once with their message identities.
The parent runs these child workflows in this order:

| Child | Input | First real behavior | Output |
| --- | --- | --- | --- |
| Triage | New stimuli and source information | Classify relevance, urgency, and input kind | Per-stimulus classifications with evidence IDs |
| Appraisal | Stimuli, triage, persona, needs | Estimate feelings and need effects | Feeling and state-change proposals |
| Context | Stimuli, appraisal, active goals | Retrieve relevant history and run a small set of structural codelets | Referenced memories and contextual candidates |
| Attention | Candidates and appraisal | Nominate and rank what matters | `AttentionKnoxel` proposals and selected candidate IDs |
| Goals | Current goals and selected context | Retain, add, complete, or reprioritize justified goals | Goal-change proposals with reasons |
| Workspace | Selected candidates and active goals | Build one coherent attention frame | `AttentionFrameKnoxel` and draft `GwtTurnRecord` |
| Action selection | Workspace, goals, available capabilities | Choose zero or more action intents | Selected `ActionSchemaKnoxel` proposals and dispatch records |

All children run through the same kernel path that their standalone tests use.
They share the parent's proposal batch.
Only final parent success commits the GWT and publishes action dispatches.
Simple stages can use one direct structured model call and a small deterministic finalizer.
No stage needs a copied pm_lida coordinator.

Use `C:\source\pm\private-machine\pm_lida.py` as a cognitive checklist, not an implementation template.
Start with context codelets for relevant memories, active goals, and capability facts.
Keep their selection and prompts in the context workflow.
The detailed codelet pathways remain later work.

## Action dispatch

Retain the existing `ActionSchemaKnoxel` and its intent fields.
Its `content` states the goal of an action, not a finished companion reply.
Add a typed application dispatch record with `action_id`, `gwt_tick_id`, `workflow_name`, `arguments`, and `depends_on`.
Keep transport routing outside the core action enum.

Existing `Reply` actions can target `dialogue_story` or `dialogue_assistant`.
Existing tool-oriented actions can target `self_management`.
No runtime condition depends on a specific companion or cognitive action name.

`depends_on` names action IDs whose successful receipts are required before dispatch.
The initial scheduling acknowledgement can run without a dependency because it describes intent.
A claim that a task actually started must use a start receipt or depend on the start action.
Rejected dependencies produce a failed action receipt, not fabricated completion.

## DialogueStory

Adapt `plan_context_story_simple` and the existing render helpers from `pm_next/pm_core/subsystems/context`.
Do not copy the old companion-turn coordinator or its Ghost-owned model selection.

The message roles are part of the contract:

1. System: expert storyteller instruction and story constraints.
2. User: character information, background, and temporal context.
3. Assistant: prior dialogue as a story, including both Jesse and Spamuel utterances.
4. User: current feelings, memories, and workspace context.
5. Assistant prefill: the companion name followed by `says: "`.

The final generation continues only the companion's utterance.
The workflow does not request tool calls.
The existing renderer needs adaptation because it currently maps Jesse dialogue to user messages.
Test the exact role sequence and prefix before testing prose quality.

A direct completion call uses the configured model.
The client uses a backend-supported continuation mode for the assistant prefill.
The live test must establish that the backend preserves this format.
Do not silently replace the prefix with an ordinary instruction or change the message roles.
The private-ontology check compares the candidate with the current capabilities and established facts.
It does not reward generic assistant language or punish the established character voice.
The workflow never stores a model-invented next Jesse utterance as user input.
Only accepted dialogue enters canonical memory and public output.

## DialogueAssistant

The system message defines the companion as the assistant, with persona and private ontology.
Normal user, assistant, and tool messages preserve tool-call structure.
This is a separate message builder, not a Story style flag.

Assistant dialogue can read its scoped Ghost tools and call workflow-management tools through MCP.
Those tools share the typed management operations used by SelfManagement.
Tool receipts determine what the companion can claim it did.
This workflow does not directly edit another run's files or bypass Ghost permissions.
Its initial tool-capable backend is the configured `Default` model.

## SelfManagement and managed runs

The workflow supports goal changes, priorities, start, status, instruction updates, and terminate.
The same domain functions serve its Python steps and Assistant MCP tools.
There is one managed-run table, not separate scheduler and status copies.

Each entry holds run ID, originating action, workflow name, parent correlation, state, latest event, and artifact location.
Run states are `accepted`, `running`, `waiting_for_input`, `completed`, `failed`, `cancelled`, and `interrupted`.
Persist the accepted request identity before starting its worker.
Record running or failed startup as a separate receipt.
Terminal runs cannot later publish a successful completion after cancellation.
Starting a run returns an acceptance receipt before background work completes.
The run emits code-generated lifecycle events and optional steward reports.
Duplicate requests with the same action ID return the existing receipt.
An unavailable workflow produces an explicit unavailable result.
The first delivery does not pretend that Minecraft ran before its integration work item exists.

Keep workflow management generic.
A small real local task establishes concurrent execution, status, updates, and termination before Minecraft integration.
Use a real Python workflow with files and declared safe boundaries, not simulated Minecraft behavior.

## Telemetry and consolidation

`pm-workflows` reports transport-neutral events and artifacts.
The application converts them into telemetry knoxels with run ID, event ID, observation time, and evidence references.
Deduplicate repeated events by run ID and event ID.
A prose report remains a reported claim unless evidence supports more specific facts.

On final completion or cancellation, the consolidation workflow reads a stable task-artifact inventory.
It proposes memories for observations, outcomes, spatial information, and reusable experience.
The finalizer validates references and commits those memories with a consolidation receipt.
Use existing `WorldPlaceKnoxel`, `WorldObjectKnoxel`, and `BodyEpisodeKnoxel` where their semantics fit.
Do not revive the obsolete executable-skill model as a runtime dependency.

Image evidence includes media type, base64 content, source artifact, content hash, and observation time.
An image-capable model or structured world observations must support visual claims.
Base64 text never enters ordinary text-embedding requests or dialogue prompts.
A task-folder inventory does not mean every log or secret becomes permanent memory.

Archive the original task folder only after durable memory storage succeeds.
Failures preserve evidence. Repeated consolidation reuses its run-and-evidence receipt instead of duplicating memories.
Automatic permanent deletion is not part of the first delivery.

## Deferred internal thought

The future workflow consumes a GWT shard, generates candidate thoughts, and emits selected thought knoxels.
MainCogLoop evaluates those thoughts like other input.
The thought workflow cannot directly dispatch actions or force a public reply.
Its thought tree, cadence, and selection policy need a later focused design.
Keep it disabled rather than substituting periodic fixed thoughts into the real acceptance run.
