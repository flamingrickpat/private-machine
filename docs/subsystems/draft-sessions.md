# Tool-backed draft sessions

## Summary

A workflow can give a native agent small MCP tools that edit a private draft.
After each completed agent turn, its validator checks that draft against the backend.
Rejection continues the same conversation with precise repair feedback.
The model need not serialize a complete result in its final message.

Parent: [workflow runtime](../runtime.md).
Current caller: [Minecraft projects](minecraft-tasks.md).
Planned rollout: [A14B](../work_items/alpha-plan/work-items/a14b.md).
Planned memory access: [scoped search and graph tools](../work_items/alpha-plan/memory-tools.md).
Judgment policy: [workflow review modes](workflow-reviews.md).
Evidence and remaining limits: [A14 verification](../work_items/verification/a14-cooperation.md#tool-backed-draft-sessions).

## Ownership and execution

[`pm.runs.draft_session.run_draft_session`][] accepts a runtime, native run ID, prompt, MCP server mapping, validator, and artifact directory.
The runtime must contain an activated model connection.
The caller obtains that connection through the normal `pm.llm` configuration path.
No workflow adapts provider HTTP fields.

The workflow constructs its servers and owns draft meaning, storage, validation, and the eventual commit.
[`pm.host.mcp_listeners.serve_services`][] holds OS-assigned local ports for those servers.
The runtime writes their descriptors to `draft-mcp.json` in the artifact directory.
It obtains the environment's native agent target, then uses the installed `pm_coder` public session functions.
The agent has no subagent tool.
File mutation tools stay disabled unless the caller explicitly enables them.
MCP tools retain the permissions that their workflow implementation grants.

One entered agent and its MCP connections remain open through repair turns.
Native transport retries and automatic history compaction remain active.
After `run_turn` completes, the runtime calls the workflow validator with the final assistant text.
The validator returns [`pm.runs.draft_session.DraftValidation`][] containing acceptance, a value, or repair feedback.
Acceptance returns [`pm.runs.draft_session.DraftSessionResult`][] to the caller.
Rejection adds the feedback as the next user turn without replaying successful edits.
Unexpected validator errors reach the caller.
A rejection without feedback is a programming error.

## Minecraft task group

[`workflows.minecraft_project.planning.native_plan.plan_session`][] supplies the planner group.
It constructs native instructions directly. `resources/planner.md` remains available for historical direct-planner captures and comparisons.
Its [`workflows.minecraft_project.planning.draft.TaskDraft`][] starts from the current accepted specification, or a fresh goal if none exists.
Its tools provide these operations:

| Tools | Purpose |
| --- | --- |
| `task_list`, `task_get` | Inspect existing IDs and complete node fields |
| `task_add`, `task_edit` | Allocate a node and its parent link, or patch named fields |
| `condition_set`, `condition_copy` | Set one typed condition, expand an exact block box, or copy exact criteria |
| `plan_outcome`, `plan_check` | Select the planning result and inspect deterministic validation feedback |
| `plan_context` | Read the request, observations, retained history, and prior repair feedback |
| `world_read`, `world_contracts` | Read current world facts through permitted observation tools |
| `evidence_list`, `evidence_read` | Inspect catalogued original evidence with source identity |

The backend supplies generated IDs and reciprocal child links.
An inclusive block box becomes every exact cell between its corners.
This convenience preserves the normal condition semantics.
Partial edits preserve fields that the tool did not name.
Adding descendants to a leaf changes it into a story while retaining its criteria.
Invalid conditions return model validation errors without committing a specification.

Completed-turn validation uses the real task schemas and graph rules.
It also checks accepted specification protection, scoped edits, refinement criteria, and required changes after a failed method.
[`workflows.minecraft_project.tree_store.TreeStore.check_revision`][] checks the candidate against the current store without writing files.
Successful validation produces the existing `PlanProposal` contract.
Later planning code still checks instruction revisions and performs the existing specification commit.
Independent review and world verification remain separate gates before task acceptance.
Draft validation alone cannot establish house construction or completed mining.

## State, cancellation, and recovery

Each successful edit writes the workflow's private `draft.json`.
It does not commit an accepted plan or gameplay proof.
The planner's native session ID combines the managed task ID with its planner role.
Later planning groups retain that conversation while rebuilding editable state from the current accepted specification.
Unaccepted edits from an interrupted group are not automatically restored.
Preserve the rejected draft files when diagnosing such an interruption.

The runtime checks cancellation before inference, while awaiting the turn, and around validation.
Its observation deadline defaults to 7,200 seconds and spans all repair turns in one invocation.
Cancellation or deadline expiry cancels the turn and awaits cleanup before closing listeners.
Synchronous tools and validators must check cancellation beside their own effects.
The shared runner does not make an arbitrary synchronous operation cancellable.

The Minecraft group also checks the managed instruction revision beside draft writes and world reads.
A changed revision raises `InstructionsChanged` so the planning caller can refresh the request.
World reads use the existing command lock and read-only tool whitelist.
They cannot mutate the game world.
The task group does not start another managed worker to repair its draft.

## Inspect a real run

Planner decision artifacts remain below the project's `decisions/NNNN-planner/` directory.
Read `input.json`, `draft.json`, `output.json`, and `session.json` together.
`native/draft-turns.jsonl` records each completed turn, acceptance, feedback, elapsed seconds, and native token statistics.
Its `session_ref` links to native agent logs under the configured prompt-log directory's `agents/` folder.
Failed or cancelled turns may leave no completed-turn row; inspect the native logs and workflow receipt as well.
Keep these artifacts under ignored runtime paths.

The A14 real prototype established same-conversation repair and exact 25-cell expansion without an accepted specification commit.
A separate real session established cancellation after a draft edit, closed listener endpoints, and no late commit.
It is not a controlled old-versus-new latency comparison.
Integrated gameplay, restart, and human acceptance remain separately recorded in the A14 evidence.
A14B requires those lifecycle checks and measured benefit before extending this pattern to other workflows.
