# Structural refactor

## Summary

This refactor makes pm readable for human developers without changing its behavior.
The runtime moves to `pm/` and provides only APIs and entry points.
All companion behavior moves to `workflows/`: cognition, speech, managed work, and the Minecraft body.
Each workflow folder holds its manifest, scripts, prompts, skills, and host binding together.

Parent: [work-item process](../process.md).
Related: [glossary](../../concepts/glossary.md), [naming rules](../../concepts/naming.md), [relocation map](relocation-map.md).

## Owner decisions

These decisions were made on 2026-10-12. They are not open.

- The product is **private-machine (pm)**. Spamuel is a synthetic test persona.
- Spamuel stays the default seed for now. Later, users supply their own persona.
- `pm/` is the runtime. `workflows/` holds all behavior. `pm` never imports `workflows`.
- One explicit assembly, `workflows/companion.py`, wires the application. Auto-discovery comes much later.
- Development-only material stays in `dev/`, `test_data/`, and `docs/work_items/`.
- Reserved code stays. A15–A19 need it.
- Known defects stay unfixed. The refactor keeps exact behavior.
- The test runner stays `unittest`.
- Minecraft moves early, so that A13 can be investigated by hand afterward.

## Target layout

```text
pm_next_v2/
  main.py                    three-line launcher for pm.app.cli
  pm/                        RUNTIME: APIs and entry points
    paths.py                 REPO_ROOT
    app/                     CLI, operator configuration, lifecycle records, terminal input
    console/                 operator HTTP server, worker process, static UI
    bus/                     envelopes and BusSystem
    host/                    BusModule, Application, folder target,
                             MCP listeners, durable files, cancellation
    llm/                     config, HTTP client, activation, context window,
                             stage model, workflow sessions
    records/                 record schema, mental state, cycle state, global workspace
    memory/                  MemoryStore, snapshot, write batch, retrieval, references,
                             embeddings, checkpoint, YAML codec, bash view, store tools
    runs/                    WorkflowModule, workflow runner, managed-run supervisor, management API
  workflows/                 BEHAVIOR: manifests, scripts, skills, host bindings
    companion.py             the assembly
    main_cognition/          cognition cycle, stages/, host.py, affect dynamics, dispatch recovery
    public_speech/           library: public context, claim review, publication, reply recovery
    dialogue_story/          Story dialogue
    dialogue_assistant/      Assistant dialogue
    self_management/         SelfManagement
    steward/                 Steward
    local_task/  note_task/  managed local workflows
    memory_consolidation/    run artifacts to records
    minecraft_body/          library: lease, gateway, journal, call outcomes, MCP client
    minecraft/               specialist session
    minecraft_project/       project tree, acceptance, planning/
    demo/                    dummy cognition mode, classification demo
  tests/   test_data/   dev/   docs/
```

The [relocation map](relocation-map.md) gives the target of every current file.

## Dependency rules

Imports may only point in these directions:

| Package | May import |
| --- | --- |
| `pm.records` | external libraries only |
| `pm.llm` | `pm.records` |
| `pm.memory` | `pm.records`, `pm.llm` |
| `pm.bus` | external libraries only |
| `pm.host` | `pm.bus`, `pm.memory`, `pm.records` |
| `pm.runs` | `pm.host`, `pm.memory`, `pm.llm`, `pm.records`, `pm.bus` |
| `pm.app`, `pm.console` | all of `pm` |
| workflow libraries | `pm`, other libraries, `main_cognition` |
| workflows | `pm`, workflow libraries |
| `workflows/companion.py` | everything in `pm` and `workflows` |
| `tests` | `pm`, `workflows`, `test_data` |
| `dev` | `pm`, `workflows`, `tests/support`, `test_data` |

`tests/test_architecture.py` checks these rules. It parses source only, so it runs without dependencies.
`public_speech` builds on accepted cognition state, so libraries may import `main_cognition`.
Only `main.py`, tests, and gates import `workflows.companion`. The console loads it by name through `pm/console/assembly.py`.

Phase 4 removed every runtime violation:

| Violation today | Removal in Phase 4 |
| --- | --- |
| `managed_runs` names `minecraft` and `minecraft_project` and injects a persona for them. | Done: `WorkflowSpec.resumable` and `WorkflowSpec.request_context`. |
| `managed_runs` imports `workflows.public_context`. | Done: `workflows.minecraft.host.persona_context`. |
| `reply_recovery` imports Minecraft verifiers per report schema. | Open. The test allowlist holds these two imports. |
| `operator_services` and the console worker hold Minecraft endpoint defaults. | Done: `workflows/minecraft_body/operator_hooks.py`. |
| `pm/console/worker.py` imports `workflows.companion.build_app`. | Done: `pm/console/assembly.py` loads it by name. |
| `workflows/companion.py` holds workflow descriptions as strings. | Kept: the assembly is workflow-side code. |

## Three walkthroughs

These read only the target structure.

### A cognition decision reaches the Minecraft body

1. `workflows/main_cognition/host.py` receives `user.input` and starts one cognition cycle.
2. `workflows/main_cognition/steps.py` runs the seven `stages/`. `action_selection` proposes an action intent.
3. `dispatch.py` admits it. `host.py` commits the dispatch record, then publishes `action.request`.
4. `workflows/self_management/host.py` receives it. Its workflow calls the management API in `pm.runs`, which starts a `minecraft_project` run.
5. `workflows/minecraft_project/planning/` plans the tree. `planning/dispatch.py` starts a specialist session from `workflows/minecraft/`.
6. Every body call passes `workflows/minecraft_body/gateway.py`, which checks the lease epoch and writes the journal.

### Evidence returns to cognition

1. `minecraft_body/journal.py` holds the call rows. `call_outcomes.py` classifies them.
2. `minecraft_project/planning/verify.py` reads the world through read-only tools. `acceptance.py` evaluates the conditions. `tree_store.py` records the certificate.
3. `planning/integration.py` writes the project report. The supervisor records a terminal run receipt and emits an event.
4. `memory_consolidation/artifact_consolidation.py` turns the run artifacts into records.
5. `public_speech/reply_recovery.py` verifies the report and starts the task reply. The next cognition cycle retrieves the records through `pm.memory.retrieval`.

### A developer changes a planning policy

1. Open `workflows/minecraft_project/`. The planner prompt is `resources/planner.md`.
2. The planner role is `planning/plan.py`. Its model call is in `planning/decisions.py`. Its inputs come from `planning/context.py`.
3. Change the policy there. The runtime is not involved.
4. Run `tests/unit/minecraft_project/` and the captured-receipt replays.

## Phases

Each phase is one branch. Each commit is a move, a rename, or an extraction, never a mix.

| Phase | Content | State |
| --- | --- | --- |
| 0 | Baseline and repository hygiene: `dev/`, `test_data/`, `docs/work_items/`, `AGENTS.md`. | Done, branch `refactor/phase-0`. Test baseline not yet run. |
| 1 | This page, the glossary, the naming rules, and the relocation map. | Done, branch `refactor/phase-1`. Awaits owner review. |
| 2 | Build `pm/`: move the runtime packages. One commit per package. Then renames. | Done, branch `refactor/phase-2`. Static checks only. |
| 3 | Move behavior into `workflows/`: Minecraft first, then `modules.py`, then the assembly. | Done, branch `refactor/phase-3`. Static checks only. |
| 4 | Remove domain knowledge from the runtime. Empty the allowlist. | Done, branch `refactor/phase-4`. One library coupling stays open. |
| 5 | Split large files: `minecraft_project/steps.py`, `supervisor.py`, `main_cognition/host.py`. | Pending. |
| 6 | Reorganize tests into `unit/`, `contract/`, `live/`. | Pending. |
| 7 | Walkthrough review. Then the A13 investigation. | Pending. |

## Verification

Every commit must keep these results equal to the Phase 0 baseline:

- The full `unittest` run over `tests/test_*.py`.
- The topology dry-run in `tests/test_topology.py`.
- `mkdocs build --strict`.

Extra checks for risky steps:

- Record renames: load the Spamuel canon, save it, and compare the bytes with the original.
- Phase 4: the produced request payloads must be identical before and after.
- Phase 5: replay the captured receipts in `test_data/captures/alpha/`.

Live gates need the real model endpoints and the real Minecraft server.
They are optional checkpoints that the owner runs on a suitable machine.

## Risks

- **Skill paths.** Manifests reference skills by repository-relative paths. A managed run that started before a move may not resume after it. Finish or cancel runs first.
- **Lock ordering in the supervisor.** Code moves verbatim. Phase 5 must keep every lock scope.
- **History encoding.** `history_encoding.py` and the specialist prompt describe the same format. Change them together.
- **Replay guard.** `PermissionError` must still become `RuntimeError` before it reaches the kernel.
- **Persisted names.** See [names that never change](../../concepts/naming.md#names-that-never-change).
