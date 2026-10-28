# Typing and docstring pass

## Summary

This pass makes the existing code easier to read without changing companion behavior.
It adds precise annotations, replaces anonymous internal dictionaries with named contracts, removes reflection where a real type exists, and rewrites module and function documentation to the manual's standard.

Parent: [structural refactor](index.md).
Code conventions: [development loop](../../workflows/development.md#typed-contracts).

## Scope

The pass covers the runtime, the workflows, and the simulation harness.

| Area | Files | Main work |
| --- | --- | --- |
| `pm/` | 75 | Annotations, `TypedDict` and `Protocol` boundaries, module overviews, legacy typing, deprecated `datetime.utcnow()` calls. |
| `workflows/` | 106 | Annotations, fixed-shape receipt and report mappings, host overviews, one invalid library import removed. |
| `simulation/` | 14 | Annotations and a package overview; the controller state stays open JSON by design. |

The five `pm/records` files that still used `typing.Dict/List/Optional/Tuple` were converted in the same pass.
No test, prompt, workflow manifest, Pydantic field name, JSON key, or control-flow branch changed on purpose.

## Contracts that replaced anonymous dictionaries

- `pm.bus.envelopes`: `EnvelopeJson`, `JsonObject`, `JsonValue`, and `EnvelopeKind`.
- `pm.records.memory_records`: `MentalStateInput` plus `FeatureStore` and `FeatureBatch` protocols for `new_feature`.
- `pm.memory.retrieval`: `ActivityEvent` and `ActivitySummary`.
- `pm.memory.store`: `DialogueTurn`.
- `pm.memory.bash_view`: `GhostIndex` and `RecordIndexEntry`.
- `pm.app.control`: `ModuleActivity`, `RuntimeState`, `DeliveryState`, and three receipt mappings.
- `pm.app.lifecycle_records`: `StartupEvidence`.
- `pm.app.embed_cli`: `VectorSampleStats` and `EmbeddingRebuildReport`.
- `pm.app.operator_config`: `ConnectionRow` and `ConnectionReport`.
- `pm.console.server`: `InstanceSettings`.
- `pm.runs.supervisor`: `ParkedRunPointer` and `BoundaryInstructions`.
- `pm.llm.activation`: the model-discovery entry mappings.
- `pm.llm.prompt_encoding`: the OpenAI wire-part contracts.
- `pm.introspection.capabilities`: three capability rows.
- `workflows.main_cognition`: `GoalChangeReceipt`, `BeliefUpdateReceipt`, `RejectedBeliefUpdate`, `UtteranceMetadata`, and `NarrativeDefinition`.
- `workflows.companion`: `CognitionModules` and `DummyWorkflowPaths`.
- `workflows.demo`: `TaskRecord`.
- `simulation.review.record_format`: `RecordLink`.

Mappings that stay open JSON keep a JSON mapping type and a docstring that names the expected keys.
Examples are provider request bodies, tool results, and the simulation controller document.

## Deliberate behavior differences

These are the only intended differences. Review them before accepting the pass.

1. `pm.memory.record_index`: a duplicated `@property` on `all_ontology_terms` was removed. The old definition returned a property object instead of a list. No current caller used it.
2. `pm.records.needs_axes`: the `__mul__` helper no longer swallows every exception per axis. A non-numeric axis value would now raise instead of being silently dropped. Current models declare float axes only.
3. `pm.app.character_home`: `unsupported_capabilities` and `capability_notes` are now `list[str]`. A malformed `character.yaml` with non-string entries now fails validation. The repository's example personas already use strings.

`pm.app.character_cli` now reads `vclock.utcnow()` for import receipts instead of real UTC.
When no virtual clock is installed the value is identical. Inside a simulation the receipt follows world time, which matches the records it describes.

## Verification

All commands ran on Windows with the repository virtual environment.
The model endpoints on ports 8080 and 8081 were running.

- `python -m compileall -q pm workflows simulation`: exit 0.
- Import sweep: every module imported. `pm` 134 modules, `workflows` 174 modules, `simulation` 14 modules, zero failures.
- Architecture and topology tests: 15 tests, OK. This caught and fixed one library import from `workflows.dialogue_assistant` into `public_speech`.
- Targeted unit suites were compared with a clean `HEAD` worktree.
  The failing test names were byte-identical for every package below, so the pass added no test regression.

| Package | Result |
| --- | --- |
| `tests/unit/pm/bus` | 8 OK |
| `tests/unit/pm/records` | 50 OK |
| `tests/unit/pm/memory` | 152 tests, 1 failure + 1 error, identical to `HEAD` |
| `tests/unit/pm/llm` | 121 tests, 5 failures, identical to `HEAD` |
| `tests/unit/pm/introspection` | 36 tests, 3 failures, identical to `HEAD` |
| `tests/unit/pm/host` | 28 tests, 1 failure, identical to `HEAD` |
| `tests/unit/pm/console` | 2 tests, 1 error, identical to `HEAD` |
| `tests/unit/pm/runs` | 47 OK |
| `tests/unit/pm/context_render` | OK |
| `tests/unit/pm/app` | 46 tests, 4 failures + 1 error, identical to `HEAD` |
| `tests/unit/workflows/main_cognition` | 136 tests, 46 errors, identical to `HEAD` |
| `tests/unit/workflows/public_speech` | 81 tests, 1 failure, identical to `HEAD` |
| `tests/unit/workflows/demo` | 14 tests, 2 failures + 1 error, identical to `HEAD` |
| `tests/unit/workflows/inner_thought` | 76 tests, 1 failure + 3 errors, identical to `HEAD` |
| `tests/unit/workflows/cognitive_control` | 36 tests, 6 errors, identical to `HEAD` |
| `tests/unit/workflows/focus_management` | 8 tests, 1 failure, identical to `HEAD` |
| `tests/unit/workflows/memory_consolidation` | 8 tests, 1 failure, identical to `HEAD` |
| `tests/unit/workflows/tool_call` | 15 tests, 1 error, identical to `HEAD` |
| `tests/unit/workflows/dialogue_assistant` | 20 OK |
| `tests/unit/workflows/dialogue_story` | 12 OK |
| `tests/unit/workflows/introspection` | 5 OK |
| `tests/unit/workflows/minecraft` | 8 OK |
| `tests/unit/workflows/minecraft_body` | 39 tests, 1 failure + 2 errors, identical to `HEAD` |
| `tests/unit/workflows/minecraft_project` | 170 OK |
| `tests/unit/workflows/reflection` | 22 OK |
| `tests/unit/workflows/self_management` | 8 OK |
| `tests/unit/workflows/steward` | 6 OK |
| `tests/unit/dev` | 78 OK |
| `tests/unit/simulation` | 6 OK |
| `tests/unit/test_design_contracts` | 7 tests, 3 errors, identical to `HEAD` |

- The strict pyright configuration in `pyrightconfig.json` (11 files): 0 errors.
- `mkdocs build --strict`: success. The site rebuilt in 155 seconds.

### Pre-existing failures

The failures above already exist at commit `8480b73`.
They include the memory repair exhaustion and repeated-consolidation assertions, the prompt-log JSON sidecar assertions, introspection revision ids, and the demo/scheduler and cognitive-control lifecycle assertions.
They are not evidence for or against this pass.

## Not verified

- Minecraft workflow unit suites were not run: they need a live Minecraft body endpoint.
- No live conversation or cognitive-evaluation gate was run. Model quality cannot be judged from these static changes.
- The simulation control API was not exercised against a running participant process. The six unit tests cover plan validation and checkpoint restore only.
- The docs build proves the site builds. It does not prove every prose claim against the changed code.

## Follow-ups for the owner

1. `workflows/demo/host.py` and the other `CognitiveWorkflow` subclasses still accept `*args: Any, **kwargs: Any` and forward them to the base constructor. The base signature is now explicit; typing each subclass would remove the last broad argument bags.
2. `pm.app.web_chat.Submit` and its control callback remain broad callables. A real protocol would make the participant submit path statically checkable.
3. `workflows/minecraft_project/tree_store.py` still stores an open JSON state document. A shared TypedDict would require coordinated validation in the planning and evidence modules.
4. `pm.records.needs_axes` has no importer in the repository. Decide whether to delete it or finish the second needs model.
5. Several pre-existing test failures listed above need owner decisions; they are not typing issues.
