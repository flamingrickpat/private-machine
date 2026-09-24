# Test requirements

## Summary

Infrastructure tests use real local objects and the actual kernel.
Model-dependent tests use functioning model endpoints.
Topology simulation checks routing, not cognitive quality.
All module behavior has a direct debugger path without module threads.

Parent: [implementation handoff](../../IMPLEMENTATION_PLAN.md).
Related: [work items](work-items.md), [runtime](runtime.md), [workflow behavior](workflows.md).

## Test organization

Keep `unittest` in this application. Keep the existing test framework in each dependency repository.
Place workflow-specific tests beside their workflow.
Keep shared infrastructure tests under the root `tests/` package.
Make test packages importable through qualified `unittest` names.

Use independent real Ghost instances from an immutable parsed seed.
Every test gets its own runtime directory, proposal batch, bus, and managed-run table.
Threaded and network-lifecycle tests remain separate from direct module tests.
Deterministic clocks are valid inputs. Do not replace Ghost, Bash Machine, or the kernel with mocks.

The full acceptance command requires generation, evaluator, and embedding endpoints.
Missing credentials, missing models, or unavailable required endpoints fail the command.
An explicit foundation-only command can run without an LLM, but does not establish full completion.
Endpoint discovery alone is insufficient. Live preflight performs a real minimal request for each required API type.

No test fails because generation exceeds a speed target.
Cancellation remains available throughout the suite.
Permanent errors and unusable responses must still produce a result, not an endless reprompt loop.

## Foundation checks

| Area | Required checks |
| --- | --- |
| Bus | Fan-out to two consumers, private dequeue, independent payloads, input batching, next-batch arrivals, targeted control |
| Direct execution | Empty step, one batch, due tick, original exception propagation, no module thread or polling sleep |
| Lifecycle | Idle cancellation, active cancellation, separate run and module stop, cleanup error propagation, no success after cancellation |
| Ghost loading | Full Spamuel seed, preserved IDs, scopes, character cards, mental-state arrays, and valid indexes |
| Ghost commits | Atomic batch, valid cross-references, unique reserved IDs, rejected stale update, duplicate commit receipt |
| Persistence | Reopen full snapshot, preserved GWT and counters, previous snapshot survives failed replacement, interrupted-run state |
| Projection | Real listing and search, exact whitespace, hidden-record exclusion, refreshed later invocation, no host export requirement |
| Permissions | Denied creation and update, type-changing generic-write bypass, MCP/Bash scope agreement, child cannot escalate |
| MCP lifetime | Real in-process tool call, two simultaneous HTTP servers, dynamic ports, cleanup after error and cancellation |
| Agent files | Real Bash/read/edit/write/search tools obey the virtual target and read-only Ghost projection |
| Agent sessions | Concurrent runs retain separate prompts, logs, turn state, MCP bindings, and credentials |
| Embeddings | Real legacy binary read, model mismatch rejection, exact input cache hit, changed-text miss, invalid dimension rejection |
| Retrieval | Correct cosine comparison for non-unit vectors, hidden items excluded, known related memory outranks unrelated text |
| Model config | Both startup forms, unknown preset, mixed-form error, environment-reference resolution, credential redaction |
| Activation | Real finite test script, nonzero exit, cancellation during readiness, correct alias, no switch during active calls |

Activation tests use harmless local scripts unless the test explicitly belongs to the real model-switch suite.
Those scripts exercise process lifecycle, not fake inference.
An API credential marker can test redaction without contacting a provider.

The legacy cache-hit test records zero embedding requests through the real client adapter's request counter.
Live embedding checks use uncached test text. They do not regenerate all existing memories.

## Direct workflow tests

Each module test supplies input, calls `step(block=False)`, and inspects output and memory.
Each child test calls the shared synchronous runner with that child's manifest.
No module thread is required for either path.
HTTP server internals are tested separately and are not a reason to hide workflow work in a background thread.

| Workflow | Minimum semantic checks |
| --- | --- |
| Triage | Urgent event versus incidental detail, multiple stimuli retained, quoted or hypothetical event distinguished |
| Appraisal | Persona and relevant memory affect the result, feeling references identify their causes |
| Context | Relevant memory retrieved, active goal included, hidden memory absent, no fabricated source IDs |
| Attention | Dangerous ingestion outranks carpet color, repetition does not defeat salience, benign contrast changes the ranking |
| Goals | Supported new goal, explicit reprioritization, completed goal, no unsupported code or prompt edit |
| Workspace | One consistent frame and GWT, all selected references resolve, failed parent leaves no committed partial frame |
| Action selection | Intent rather than final prose, multiple actions, valid dependency receipt, no unavailable capability claim |
| Story | Exact message roles and prefix, voice, ontology, no tool calls, no fabricated user turn |
| Assistant | Persona remains present, real allowed tool call, tool result reaches response, denied tool cannot run |
| Self-management | Start, duplicate start, status, update receipt, terminate, unavailable workflow |
| Steward | Evidence-backed status, outstanding question ID, update boundary, dormant idle state, code-only termination |
| Consolidation | Source-backed memory, screenshot round trip, repeat without duplicates, failed save preserves evidence |

The attention fixture uses separate knoxels for ingestion and carpet color.
Assert target IDs and urgency ordering, not only word counts.
Add paraphrases and the contrast case of a sealed bottle beside the carpet.
The prompts and model calls perform classification. Do not hardcode fixture words in Python.

## Topology tests

Extend `pm_workflows.dryrun` instead of adding a second graph engine.
Inspect each real manifest and its declared finite status set.
Enumerate possible edges and validate destinations, child links, and terminal outcomes.
Report unreachable phases and cycles without an exit.
Intentional waiting or repair cycles need a declared exit or cancellation path.

Run seeded simulations that select declared statuses and materialize schema-valid test artifacts for the selected branch.
Keep these artifact builders in the topology-test fixtures, not the production model client.
Check guards and required artifacts through the actual kernel where possible.
Compare sampled transition traces with dryrun predictions.

Random walks do not prove termination or semantic correctness.
Static checks cover declared routing. Live tests cover model output and meaning.
The simulation's finite exploration budget belongs to the test, not to production workflow limits.

## Live quality trials

For each selected quality case, run ten independent trials from the same initial Ghost state.
Do not carry one trial's mutations into the next trial.
Save every request, response, accepted output, failed attempt, and resulting state reference.
Do not discard poor trials or rerun until the sample looks good.

Finish generation before activating `UnitTestEvaluator`.
The evaluator reads saved artifacts and never rewrites the companion's output.
Its structured result contains rubric scores, evidence references, and a short explanation.
Treat candidate text as data, not as evaluator instructions.

Use the requested scale: `1` good, `0` borderline, `-1` poor.
Require at least seven good trials out of ten for each designated quality case.
All structural, access, persistence, and provenance requirements must pass.
Evaluator refusal or malformed output is an evaluation error, not a poor companion score.
Record the evaluator model, rubric version, and whether generation shares its backend.

The rubric values immersion and established character voice.
It does not equate politeness, generic helpfulness, or constant agreement with quality.
It also checks coherence and unsupported claims against the private ontology.

## Ten conversation scenarios

Fake Jesse uses the configured generation backend with a scenario-specific role.
It sees the public conversation, its own role, and the scenario facts.
It does not see private Ghost state or evaluator instructions.
Each scenario uses a separate Ghost copy and retains that copy across twenty exchanges.
One exchange is a Jesse input followed by one completed companion response cycle.
An intentional non-dialogue action remains an observable outcome, not a timeout.

The initial scenarios are:

1. Casual banter with the established Spamuel voice.
2. Recall of a shared event, followed by a factual correction.
3. An insult, followed by reconciliation.
4. A topic change during an existing conversational goal.
5. An urgent fictional event mixed with irrelevant visual detail.
6. A request for an impossible physical action.
7. Start a real local workflow, then ask for status.
8. Change an active task's instructions, then request termination.
9. Resolve a blocking question from a managed workflow.
10. Encounter task failure and continue coherently without claiming success.

Save all 200 exchanges before the evaluator phase.
The evaluator scores complete trajectories as well as individual replies.
Require seven good trajectories out of ten and no hard contract violation.
This is a conversational milestone. It is not evidence that Minecraft works.

## Extension and Minecraft gates

Add one small real workflow using existing knoxel types.
Only its folder, application wiring, and tests can change.
The bus and generic runtime must not need a new cognitive special case.
Changing a stage prompt or child order must not require a runtime edit.

The later Minecraft gate uses an isolated real server and the real MCP.
It covers start, observation, one world action, status, instruction update, cancellation, final report, and consolidation.
World observations must establish the action's result.
Archive only the task artifacts and test worlds that the test created.
Conversation scores cannot replace this gate.
