# File relocation map

## Summary

This table lists every application source file and its target location.
Phases 2 and 3 execute it. Phase 5 executes the splits.
A file marked **stays** keeps its location and content.
Phase 0 already moved `scripts/`, `data/`, `demo.py`, fixtures, and work-item docs.

Parent: [structural refactor](index.md).

Actions:

- **move**: same content, new path. Imports follow.
- **rename**: move plus renamed Python identifiers from the [naming rules](../../concepts/naming.md).
- **merge**: the content joins another file.
- **split**: one file becomes several. Phase 5 does this, except `modules.py` in Phase 3.
- **stays**: no change.

No file is deleted. Reserved code moves with its owner.

## Runtime: `pm/`

| Current | Target | Action | Phase | Reason |
| --- | --- | --- | --- | --- |
| — | `pm/paths.py` | new | 2 | One `REPO_ROOT`. Four files compute it with `parent.parent` today. |
| `bus/bus.py` | `pm/bus/bus.py` | move | 2 | Runtime API. |
| `bus/envelopes.py` | `pm/bus/envelopes.py` | move | 2 | `_new_id` becomes the public `new_id`. Every caller changed, so no alias remains. |
| `runtime/shell.py` | `pm/host/application.py` | rename | 2 | `ShellModule` → `BusModule`. "shell" clashes with bash environments. |
| `workflow_adapter.py` | `pm/runs/workflow_module.py` | rename | 2 | `ShellModuleWorkflow` → `WorkflowModule`. It runs workflows, so it belongs to `pm.runs`; in `pm.host` it created a host ↔ runs cycle. |
| `working_target.py` (folder part) | `pm/host/folder_target.py` | split | 2 | `FolderEnvironment`, `FolderTarget`: host directories for workflows. |
| `working_target.py` (memory part) | `pm/memory/bash_view.py` | split | 2 | `GhostBashEnvironment` → `MemoryBashEnvironment`, `GhostBashWrapper` → `MemoryBashTarget`. Removes the store → root import. |
| `runtime/mcp_services.py` | `pm/host/mcp_listeners.py` | move | 2 | Loopback MCP listeners per run. |
| `runtime/cancellation.py` | `pm/host/cancellation.py` | move | 2 | Generic stop events. |
| `runtime/body_lease.py` (generic part) | `pm/host/durable_files.py` | split | 2 | `ProcessLock`, `atomic_json`, `read_json`. `managed_runs` uses them; they are not Minecraft concepts. |
| `runtime/models.py` | `pm/llm/config.py` | move | 2 | LLM connection configuration. |
| `runtime/model_io.py` | `pm/llm/http_client.py` | move | 2 | The single HTTP path to LLM endpoints. |
| `runtime/model_activation.py` | `pm/llm/activation.py` | move | 2 | Endpoint ownership and `ChatClient`. |
| `runtime/model_context.py` | `pm/llm/context_window.py` | move | 2 | Token counting against the serving context. |
| `runtime/cognitive_model.py` | `pm/llm/stage_model.py` | move | 2 | Lazy LLM for cognition stages. |
| `runtime/workflow_models.py` | `pm/llm/workflow_sessions.py` | move | 2 | LLM sessions leased to workflows. |
| `pm_core/model/knoxel_core.py` | `pm/records/base.py` | rename | 2 | `KnoxelBase` → `RecordBase`, `KnoxelType` → `RecordKind`, `KnoxelScope` → `RecordScope`, `KnoxelHaver` → `RecordOwner`. `PERSISTED_TYPE_NAMES` keeps the saved names. |
| `pm_core/utils/datetime_utils.py` | `pm/records/base.py` | merge | 2 | One function, used only by records. |
| `pm_core/model/knoxel_enums.py` | `pm/records/enums.py` | move | 2 | |
| `pm_core/model/knoxel_taxonomy.py` | `pm/records/taxonomy.py` | move | 2 | |
| `pm_core/model/knoxel_common.py` | `pm/records/memory_records.py` | rename | 2 | Main memory record types. |
| `pm_core/consts.py` | `pm/records/memory_records.py` | merge | 2 | Three constants, used only there. |
| `pm_core/model/knoxel_state_change.py` | `pm/records/cognitive_records.py` | rename | 2 | Intention, belief, attention, and ontology records. |
| `pm_core/model/knoxel_embodiment.py` | `pm/records/world_records.py` | rename | 2 | Place, object, and body-episode records. |
| `pm_core/model/knoxel_graph.py` | `pm/records/graph_records.py` | rename | 2 | Reserved. |
| `pm_core/model/knoxel_qualia.py` | `pm/records/qualia_records.py` | rename | 2 | Reserved. |
| `pm_core/model/knoxel_codelets.py` | `pm/records/codelet_records.py` | rename | 2 | Reserved for A15–A16. |
| `pm_core/model/mental_state_vectors.py` | `pm/records/mental_state.py` | move | 2 | Persisted inside records. |
| `pm_core/model/needs_state_vectors.py` | `pm/records/needs_axes.py` | move | 2 | Persisted inside `CycleState`. Reserved second needs model. |
| `pm_core/model/ghost_state.py` | `pm/records/cycle_state.py` | rename | 2 | `GhostConfig` → `PersonaConfig`. |
| `pm_core/model/gwt.py` | `pm/records/global_workspace.py` | move | 2 | Persisted global workspace turns. |
| `pm_core/ghost/service.py` | `pm/memory/store.py` | rename | 2 | `Ghost` → `MemoryStore`. |
| `pm_core/ghost/ghost_base.py` | `pm/memory/record_index.py` | rename | 2 | `BaseGhost` → `RecordIndex`. Phase 5 changes inheritance to composition. |
| `pm_core/ghost/batch.py` | `pm/memory/snapshot.py`, `pm/memory/write_batch.py` | split | 2 | `GhostView` → `Snapshot`, `GhostWriteBatch` → `WriteBatch`, `GhostConflict` and `GhostCancelled` keep their names: they are written into stored rejection reasons. |
| `pm_core/ghost/memory.py` | `pm/memory/retrieval.py` | move | 2 | Ranking over a snapshot. `distinctive_terms` moves in from `dialogue_story/context.py`. |
| `pm_core/ghost/references.py` | `pm/memory/references.py` | move | 2 | |
| `pm_core/embeddings.py` | `pm/memory/embeddings.py` | move | 2 | Depends on `pm.llm`, which is now allowed. |
| `pm_core/persist/dataprovider_yaml.py` | `pm/memory/yaml_codec.py` | rename | 2 | Gains the fixed `__type__` name table. |
| `pm_core/persist/snapshot.py` | `pm/memory/checkpoint.py` | rename | 2 | `GhostCheckpoint` → `Checkpoint`. Avoids two "snapshot" files. |
| `ghost_mcp.py` | `pm/memory/store_tools.py` | move | 2 | Scoped memory tools. Tool names stay `ghost_*`. |
| `runtime/workflow_runner.py` | `pm/runs/workflow_runner.py` | move | 2 | One kernel invocation. |
| `runtime/managed_runs.py` | `pm/runs/supervisor.py` | move | 2 | Split in Phase 5 into `run_receipt.py`, `control_store.py`, `event_outbox.py`, `instruction_revisions.py`, `supervisor.py`. |
| `runtime/management_tools.py` | `pm/runs/management_api.py` | move | 2 | Tool wrappers over the supervisor. |
| `runtime/operator_services.py` | `pm/app/operator_config.py` | move | 2 | `.env` parsing and health probes. Minecraft defaults leave in Phase 4. |
| `runtime/lifecycle.py` | `pm/app/lifecycle_records.py` | move | 2 | Startup and shutdown records. |
| `console/server.py` | `pm/console/server.py` | move | 2 | |
| `console/worker.py` | `pm/console/worker.py` | move | 2 | |
| `console/static/*` | `pm/console/static/*` | move | 2 | |
| `main.py` (CLI loop) | `pm/app/cli.py` | split | 3 | Terminal entry point. |
| `main.py` | `main.py` | stays | 3 | Becomes a three-line launcher, so `python main.py` still works. |

## Behavior: `workflows/`

| Current | Target | Action | Phase | Reason |
| --- | --- | --- | --- | --- |
| `main.py` (`build_app`, workflow specs) | `workflows/companion.py` | split | 3 | The assembly. Same construction order. |
| `modules.py` `MainCogLoop`, `InnerThoughtLoop`, `CognitiveWorkflow` | `workflows/main_cognition/host.py` | split | 3 | Bus modules of cognition. Phase 5 extracts `input_preparation.py` and `cycle_commit.py`. |
| `modules.py` Ghost query helpers, `_feature`, `_projection_from_report` | `workflows/main_cognition/host.py` | split | 3 | Used only by cognition. |
| `modules.py` `PublicDialogue` | `workflows/public_speech/public_dialogue.py` | split | 3 | Base bus module of both dialogue paths. |
| `modules.py` `StoryDialogue` | `workflows/dialogue_story/host.py` | split | 3 | |
| `modules.py` `AssistantDialogue` | `workflows/dialogue_assistant/host.py` | split | 3 | |
| `modules.py` `SelfManagement` | `workflows/self_management/host.py` | split | 3 | |
| `modules.py` `StewardModule` | `workflows/steward/host.py` | split | 3 | |
| `modules.py` `UserInputProxy` | `pm/app/user_input.py` | split | 3 | Terminal input is an entry point. |
| `modules.py` `Dummy*`, `TaskTable`, `ScheduleSubprocesses`, `CheckSubprocess`, `ToolPlusDialogue` | `workflows/demo/host.py` | split | 3 | `--cognition dummy` mode. |
| `*/module_tools.py` (six workflows) | the same folder's `host.py` | merge | 3 | One host binding per workflow. |
| `runtime/experience.py` | `workflows/main_cognition/experience.py` | move | 3 | Cycle clusters and beliefs. |
| `runtime/intents.py` | `workflows/main_cognition/dispatch_recovery.py` | rename | 3 | Replays pending dispatches. `replay_intents` → `replay_dispatches`. |
| `pm_core/model/affect_dynamics.py` | `workflows/main_cognition/affect_dynamics.py` | move | 3 | Cognition policy, not a record. |
| `pm_core/model/mental_state_verbalization.py` | `workflows/main_cognition/verbalization.py` | move | 3 | Reserved. |
| `pm_core/model/meta_action.py`, `meta_narrative.py` | `workflows/main_cognition/meta_prompts.py` | merge | 3 | Reserved prompt text. |
| `workflows/main_cognition/engine.py` | `workflows/main_cognition/stage_runner.py` | move | 3 | Host-side runner for the stage child workflows. |
| `workflows/main_cognition/state_changes.py` | `workflows/main_cognition/stage_commits.py` | move | 3 | Applies accepted appraisal and goal proposals. |
| `workflows/main_cognition/{steps, pipeline, dispatch, schemas, stage_support}.py`, `stages/**` | same | stays | — | Already well-bounded. |
| `workflows/main_cognition.workflow.md`, `dialogue`, `inner_thought`, `schedule`, `status`, `tool_dialogue` manifests | `workflows/demo/` | move | 3 | Dummy manifests. Their names stay. |
| `workflows/roles.py`, `workflows/roles/*.py` | `workflows/demo/roles.py`, `workflows/demo/role_skills/*.py` | move | 3 | Dummy role behavior. |
| `workflows/classification_demo/` | `workflows/demo/classification/` | move | 3 | Demonstration workflow. |
| `workflows/public_context.py` | `workflows/public_speech/public_context.py` | move | 3 | |
| `workflows/public_claims.py`, `public_claims.yaml` | `workflows/public_speech/claim_review.py`, `claim_review.yaml` | move | 3 | |
| `workflows/public_recovery.py` | `workflows/public_speech/grounding_recovery.py` | move | 3 | |
| `runtime/publication.py` | `workflows/public_speech/publication.py` | move | 3 | Commit before publication. |
| `runtime/reply_recovery.py` | `workflows/public_speech/reply_recovery.py` | move | 3 | Phase 4 replaces its per-schema verifiers with workflow declarations. |
| `workflows/dialogue_story/engine.py` | `workflows/dialogue_story/standalone.py` | move | 3 | Standalone Story runner used by `run.py`. |
| `workflows/dialogue_story/context.py` | same | stays | 3 | `distinctive_terms` leaves to `pm.memory.retrieval`. |
| `workflows/dialogue_story/{steps, schemas, run}.py`, YAML files | same | stays | — | |
| `workflows/dialogue_assistant/{steps, schemas, tools}.py`, `prompts.yaml` | same | stays | — | |
| `runtime/steward.py` | `workflows/steward/gate.py` | move | 3 | Dormant and active gate over the supervisor. |
| `runtime/consolidation.py` | `workflows/memory_consolidation/artifact_consolidation.py` | move | 3 | Turns run artifacts into records. |
| `workflows/{local_task, note_task, self_management, steward, memory_consolidation}/` other files | same | stays | — | |
| `runtime/body_lease.py` (lease part) | `workflows/minecraft_body/lease.py` | split | 3 | `BodyLeases`, `BODY_LEASES`. |
| `runtime/minecraft_actor.py` | `workflows/minecraft_body/gateway.py`, `journal.py` | split | 3 | `ActorGateway` → `BodyGateway`, `ActorTool` → `GatewayTool`. Journal reading moves to `journal.py`. |
| `runtime/minecraft_outcomes.py` | `workflows/minecraft_body/call_outcomes.py` | move | 3 | `classify`, `GameplayEvidence`. |
| `runtime/minecraft_tools.py` | `workflows/minecraft_body/mcp_client.py` | move | 3 | MCP transport. `_build_plan_tool` moves to `workflows/minecraft/` in Phase 5. |
| `runtime/capabilities.py` | `workflows/minecraft_body/capabilities.py` | move | 3 | Minecraft capability snapshot. |
| `workflows/minecraft/actor.py` | `workflows/minecraft/specialist_session.py` | rename | 3 | `run_actor` → `run_specialist_session`. |
| `workflows/minecraft/{steps, schemas}.py`, `prompts.yaml`, `resources/`, `guidance/` | same | stays | — | `.gitattributes` keeps the resource bytes. |
| `workflows/minecraft_tasks/` | `workflows/minecraft_project/` | move | 3 | Folder and manifest file follow the workflow name `minecraft_project`. |
| `workflows/minecraft_tasks/steps.py` | `workflows/minecraft_project/planning/*.py` | split | 5 | See [Minecraft project split](#minecraft-project-split). |
| `workflows/minecraft_tasks/tree.py` | `workflows/minecraft_project/tree_store.py` | move | 3 | Tree validation and the certificate lifecycle. |
| `workflows/minecraft_tasks/schemas.py` | `workflows/minecraft_project/schemas.py` | move | 3 | |
| `workflows/minecraft_tasks/verification.py` | `workflows/minecraft_project/acceptance.py` | move | 3 | Independent acceptance of world effects. |
| `workflows/minecraft_tasks/evidence.py` | `workflows/minecraft_project/evidence_selection.py` | move | 3 | Chooses evidence for model inputs. Phase 5 moves the retention operations to `planning/retention.py`. |
| `workflows/minecraft_tasks/resources/*` | `workflows/minecraft_project/resources/*` | move | 3 | |

## Minecraft project split

Phase 5 splits `steps.py` by role. Function bodies move unchanged.

| Lines today | Target | Content |
| --- | --- | --- |
| 1–53, 1536–1565 | `planning/__init__.py` (`run`) | Router and request helpers. |
| 56–177 | `planning/decisions.py` | `reason`, `reason_once`, decision schemas. |
| 215–342 | `history_encoding.py` | `gameplay_history`, `scoped_gameplay`. |
| 180–212, 345–467, 566–752 | `planning/context.py` | Instruction boundary, scoped reads, rejection feedback, context compaction. |
| 470–563 | `planning/proposals.py` | Applying node updates and refinement. |
| 755–793 | `planning/retention.py` | Retention role and retention tree operations. |
| 796–1027 | `planning/plan.py` | Planner role. |
| 1030–1066 | `planning/survey.py` | Survey role and recipe reads. |
| 1135–1220 | `planning/review.py` | Review role. |
| 1223–1287 | `planning/selection.py` | Assumption checks and leaf selection. |
| 1290–1426 | `planning/dispatch.py` | Act role. |
| 1429–1473 | `planning/verify.py` | Verifier role. |
| 1069–1132, 1476–1533 | `planning/integration.py` | Caller constraints, project check, finish. |

The manifest keeps one skill path per role, now pointing at the role file.

## Outside the application

| Current | Target | Action | Phase |
| --- | --- | --- | --- |
| `tests/test_*.py` | `tests/unit/<package>/test_<file>.py` | split and rename | 6 |
| `workflows/**/tests/` | `tests/contract/<workflow>/` | move | 6 |
| `tests/live/*` | `tests/live/*` | stays | — |
| `tests/support.py`, shared helpers in test files | `tests/support/` | split | 6 |
| Gate functions that tests import | `tests/support/` or the owning workflow | move | 6 |
| `dev/`, `test_data/`, `docs/work_items/` | same | stays | — |
