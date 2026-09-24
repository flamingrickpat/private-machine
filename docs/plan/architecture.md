# Architecture and wiring

## Summary

The shell connects independent modules through an in-process bus.
Each workflow receives a working target and scoped services.
Ghost stores shared memory. Workflow folders contain cognitive behavior.

Parent: [implementation handoff](../../IMPLEMENTATION_PLAN.md).
Details: [runtime](runtime.md), [memory](models-memory.md), and [workflows](workflows.md).

## Three separate lifetimes

| Term | Meaning | Lifetime |
| --- | --- | --- |
| Shell module | Inbox, routing predicate, and common execution loop | Application session |
| Workflow run | One request through a workflow | Request to terminal result |
| Child workflow | A synchronous stage inside another workflow | One stage |

`run()` starts a shell module thread and returns.
`step(block=False)` performs one synchronous iteration.
The thread calls `step(block=True)`.
Both paths use the same workflow runner and output handling.

Triage, appraisal, attention, and goal management are child workflows.
They do not need separate threads or bus subscriptions.
An external Minecraft task is a separate workflow run that can overlap conversation.

## Application path

```text
UserInputProxy -> MainCogLoop -> action intents
                                  |
                   +--------------+---------------+
                   v              v               v
             DialogueStory  DialogueAssistant  SelfManagement
                   |              |               |
                   +-- replies ---+        managed workflow runs
                          |                       |
                     user output             telemetry
                                                  |
                                             MainCogLoop

All accepted memory changes -> Ghost
```

User messages and external telemetry enter MainCogLoop before action selection.
Completed dialogue also becomes memory.
Its own completion notification does not automatically request another reply.
MainCogLoop can select several actions from one batch.

A Minecraft request can produce a start-task action and a dialogue action.
The first reply can describe the scheduling intent.
Only an actual acceptance receipt permits a later claim that the task started.

## Target layout

```text
main.py                       CLI and explicit application wiring
pm_core/
  model/                      existing knoxel and GWT models
  ghost/                      Ghost service, scopes, mutation batches
  persist/                    seed loading and snapshot persistence
  embeddings.py               one shared embedding client and cache adapter
bus/
  envelopes.py                message data
  bus.py                      fan-out and private mailboxes
runtime/
  shell.py                    common module lifecycle
  workflow.py                 pm-workflows adapter
  working_target.py           real-folder and virtual adapters
  models.py                   chat presets and connection resolution
  model_activation.py         explicit script activation
  managed_runs.py              active-run table and lifecycle control
workflows/
  main_cognition/
    workflow.md               ordered child phases
    module.py                 bus routing only
    stages/
      triage/                 workflow.md, prompts.yaml, steps.py, tests/
      appraisal/
      context/
      attention/
      goals/
      workspace/
      action_selection/
  dialogue_story/              workflow.md, module.py, prompts.yaml, steps.py, tests/
  dialogue_assistant/
  self_management/
  steward/
  memory_consolidation/
data/                         immutable seed fixtures
tests/                        shared infrastructure and acceptance tests
```

Create folders only with their implementation work item.
Do not create empty future frameworks or one-line forwarding classes.
Shared context functions belong in a small shared context package only where multiple workflows actually use them.
Internal thought and Minecraft receive their own workflow folders when their work items begin.

The layout is the target, not a demand for a repository-wide move before the first test.
Move one responsibility at a time. Keep imports and tests working after each move.

## Ownership rules

| Owner | Responsibility | Excluded responsibility |
| --- | --- | --- |
| Application wiring | Construct modules, presets, scopes, and subscriptions | Cognitive decisions |
| Bus | Route serialized envelopes to matching inboxes | Interpret knoxels or invoke LLMs |
| Shell runtime | Lifecycle, invocation, files, cancellation | Character policy |
| Ghost | State, IDs, access scopes, commits, persistence | Run cognitive stages |
| Workflow folder | Prompts, schemas, scripts, cognitive choices | Copy queue or thread loops |
| `pm-workflows` | Generic phases, agents, children, reports | Import Ghost or know attention types |

Keep the `ShellModule`, `ShellModuleWorkflow`, and `WorkingTarget` ABCs.
Stateless transformations remain functions.
Do not build a plugin registry, dependency-injection framework, or message broker for this delivery.
`main.py` visibly constructs and connects the modules.

## Later extensions

HTTP transport can later deliver the same envelopes to the same inbox boundary.
The in-process bus remains the only implemented transport now.
Independent processes must contact the Ghost owner. They must not mutate separate copies of supposedly shared memory.

The future internal thought workflow consumes a scoped GWT snapshot and emits thought knoxels.
Those knoxels enter MainCogLoop as input, not as direct dialogue or actions.
Its tree search and novel-thought policy remain deferred.
