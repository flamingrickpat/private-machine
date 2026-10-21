# Naming rules

## Summary

A name tells the reader what a file or symbol owns, without opening it.
Use the terms of the [glossary](glossary.md). One concept has one name.
Rename only when the current name misleads or uses a deprecated term.
Persisted names and names that a model reads never change.

Parent: [manual](../index.md).

## Names that never change

A rename must not change observable behavior.
Keep these strings exactly as they are, even when the Python name changes:

- The `__type__` values in seeds and checkpoints, for example `ActionSchemaKnoxel`. A fixed table in `pm.memory.yaml_codec` maps them to the renamed classes.
- Tool names that a model sees, for example `ghost_get`, `ghost_search`, `ghost_gwt`, `minecraft_call`.
- Virtual paths that a model sees, for example `ghost/knoxels/*.yaml`.
- Workflow names in manifests, for example `minecraft_project`. Run receipts store them.
- Bus topics, for example `action.request`, and persisted field names, for example `tick_id`, `task_id`.
- Report schema identifiers, for example `pm.minecraft-project-report.v1`.
- Runtime locations, for example `.runtime/ghost.json`.
- Prompt text.

## Packages and folders

- The runtime lives in `pm/`. Behavior lives in `workflows/`. Development tools live in `dev/`. Test data lives in `test_data/`.
- Name a runtime package after the API it provides: `pm.memory`, `pm.llm`, `pm.runs`.
- Name a workflow folder after its manifest name: `workflows/minecraft_project/` holds `minecraft_project.workflow.md`.
- Name a workflow library after its shared concept: `public_speech`, `minecraft_body`.
- Do not create `utils`, `common`, `helpers`, or `misc` folders. Put a helper in the narrowest package that owns its concept.
- Create a folder only for a real boundary. A folder with one file is acceptable only for a workflow or a stage.

Acceptable: `workflows/main_cognition/phases/appraise.py`.
Not acceptable: `pm_core/utils/datetime_utils.py` (one function, generic folder name).

## Python files

- Name a file after the concept or operation that it owns, as a noun: `tree_store.py`, `claim_review.py`, `retrieval.py`.
- Let the folder give context. `minecraft_body/lease.py`, not `minecraft_body/minecraft_body_lease.py`.
- Keep enough context for stack traces and search. `pm/memory/store.py` is acceptable because its import path is `pm.memory.store`.
- These names have a fixed meaning in every workflow folder:

| File | Meaning |
| --- | --- |
| `<name>.workflow.md` | The manifest. |
| `steps.py` | The kernel-side skill with `run(context)`. A large one becomes a `steps/` package. |
| `schemas.py` | Pydantic request, report, and decision contracts. |
| `host.py` | The host binding: bus module and host-side tools. |
| `prompts.yaml` or `prompts/` | Prompt text. |
| `resources/` | Skill text that an agent role reads. |

- Avoid `engine.py`, `manager.py`, `handler.py`, `core.py`, `base.py`, `support.py`, `model.py` unless the package gives them one exact meaning.

| Current | Problem | Target |
| --- | --- | --- |
| `modules.py` | Every bus module in one file. | One `host.py` per workflow. |
| `working_target.py` | "target" means three things. | `pm/memory/bash_view.py`, `pm/host/folder_target.py`. |
| `runtime/models.py` | "model" means the LLM, a schema, or the mental state. | `pm/llm/config.py`. |
| `runtime/workflow_models.py` | It holds LLM sessions, not workflow data. | `pm/llm/workflow_sessions.py`. |
| `runtime/shell.py` | It clashes with the bash environments. | `pm/host/application.py`. |
| `runtime/minecraft_actor.py` | Two different "actors". | `workflows/minecraft_body/gateway.py`. |
| `pm_core/model/knoxel_state_change.py` | It holds record types, not state changes. | `pm/records/cognitive_records.py`. |
| `workflows/main_cognition/engine.py` | "engine" says nothing. | `workflows/main_cognition/stage_runner.py`. |
| `workflows/main_cognition/state_changes.py` | "state" has four meanings. | `workflows/main_cognition/cycle_commits.py` (A19F renamed it again when the stages went away). |

## Classes and interfaces

- Use nouns in `PascalCase` from the glossary: `MemoryStore`, `WriteBatch`, `BodyGateway`.
- Name a record class `<Concept>Record`: `ActionSchemaRecord`, `BeliefRecord`. The old `__type__` stays.
- Use `Protocol` for a boundary that has more than one implementation. Name it after the capability: `WorkingTarget`, not `IWorkingTarget`.
- Do not prefix a class with its package name. `pm.memory.Snapshot`, not `MemorySnapshot`.
- Name exceptions `<Condition>Error`. Keep an exception's existing name when the name is written into saved data. `GhostConflict` and `GhostCancelled` keep their names for this reason: `type(error).__name__` goes into stored rejection reasons.

## Functions and methods

- Use a verb for an operation: `classify`, `commit`, `replay_dispatches`.
- Use a noun for a pure query result: `eligible_leaves`, `dialogue_history`.
- Prefix a function that performs external I/O with its effect when it is not obvious: `read_journal`, `observe_conditions`.
- Do not encode the caller in the name. `scoped_gameplay` describes the result, not the user.

## Constants, enumerations, and configuration

- Constants use `UPPER_SNAKE_CASE`: `ACTIVE_STATES`, `READ_TOOLS`.
- Define one set of run states in `pm.runs`. Other packages import it.
- Enumerations use `PascalCase` class names and the existing member values.
- Configuration classes end in `Config`: `LlmConfig`, `PersonaConfig`.

## Tests and fixtures

- A test file mirrors the file it tests: `tests/unit/minecraft_project/test_tree_store.py`.
- Name a test after the behavior: `test_stale_epoch_cannot_dispatch`.
- Do not name a test file after a work item. Record the work item in the docstring.
- Captured receipts live in `test_data/captures/<origin>/` and keep their source digest.

## Infrastructure adapters

- Name an adapter after the external system and its role: `mcp_client.py`, `http_client.py`, `mcp_listeners.py`.
- Keep the external system's own names in arguments and payloads.
