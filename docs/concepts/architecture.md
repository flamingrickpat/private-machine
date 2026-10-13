# System map

## Summary

The system has three distinct lifetimes: application modules, workflow runs, and synchronous child stages.
Ghost owns shared state within one instance.
The console owns worker processes and trace files.

Parent: [manual](../index.md).

## Control and data flow

```text
Browser -> loopback HTTP service -> worker commands
                                   |
                    UserInputProxy -> private bus mailboxes
                                         |
                                    MainCogLoop
                                         |
          triage -> appraisal -> context -> attention
                                         |
                         goals -> workspace -> action_selection
                                         |
                              accepted Ghost commit
                                         |
                  Story / Assistant / SelfManagement
                      |                         |
                public reply               managed worker
                      |                         |
                   browser                receipts -> cognition

All accepted records -> instance checkpoint
Runtime evidence -> events.jsonl -> browser graph and details
```

The arrows describe causality.
They do not imply that all work runs on one thread.
Child stages run inline inside the cognitive module.
Managed tasks can overlap dialogue.

## Ownership

| Owner | State | Boundary |
| --- | --- | --- |
| HTTP service | Instance definitions, process handles, event cursor | [`pm.console.server.Console`][] |
| Worker | Application, trace hooks, pending exchange | [`pm.console.worker.run`][] |
| Application | Module lifecycle and first error | [`pm.host.application.Application`][] |
| Bus | Subscriptions and serialized fan-out | [`pm.bus.bus.BusSystem`][] |
| Bus module | Private inbox and active invocation | [`pm.host.application.BusModule`][] |
| Memory store | Records, revisions, GWT, receipts | [`pm.memory.store.MemoryStore`][] |
| Kernel runner | One workflow invocation and artifacts | [`pm.runs.workflow_runner.WorkflowRunner`][] |
| Supervisor | Managed-run lifecycle | [`pm.runs.supervisor.ManagedRunSupervisor`][] |
| Assembly | Which bus modules run and how they are wired | [`workflows.companion.build_app`][] |

## Code layout

The runtime in `pm/` provides APIs and entry points. It never imports `workflows/`.
All companion behavior lives in `workflows/`.

| Folder | Content |
| --- | --- |
| `pm/bus/` | Envelopes and the bus. |
| `pm/host/` | Bus modules, the application, host folders, MCP listeners, durable files, cancellation. |
| `pm/llm/` | LLM configuration, HTTP client, activation, context window, stage model, workflow sessions. |
| `pm/records/` | The record schema, mental state, and cycle state. |
| `pm/memory/` | The memory store, snapshots, write batches, retrieval, embeddings, checkpoints. |
| `pm/runs/` | Workflow-backed bus modules, the kernel runner, and the managed-run supervisor. |
| `pm/app/`, `pm/console/` | The terminal CLI and the operator console. |
| `workflows/<name>/` | One workflow: manifest, `steps.py`, prompts, schemas, and `host.py`. |
| `workflows/public_speech/`, `workflows/minecraft_body/` | Libraries that several workflows share. |
| `workflows/companion.py` | The assembly. |

`tests/unit/test_architecture.py` enforces these import directions.
The [structural refactor](../work_items/refactor/index.md) explains the layout and three walkthroughs.
The [glossary](glossary.md) defines the terms.

## Traps

Separate instances do not share accepted memory.
Separate chat sessions within one instance do share accepted memory.
Bus delivery has no durable replay mechanism.
The console trace records observation, not a transaction log for bus recovery.

## Go deeper

Read [one exchange](../workflows/exchange.md), [cognition](../subsystems/cognition.md), and [operator service](../subsystems/console.md).

