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
| HTTP service | Instance definitions, process handles, event cursor | [`console.server.Console`][] |
| Worker | Application, trace hooks, pending exchange | [`console.worker.run`][] |
| Application | Module lifecycle and first error | [`runtime.shell.Application`][] |
| Bus | Subscriptions and serialized fan-out | [`bus.bus.BusSystem`][] |
| Module | Private inbox and active invocation | [`runtime.shell.ShellModule`][] |
| Ghost | Memory, revisions, GWT, receipts | [`pm_core.ghost.service.Ghost`][] |
| Kernel runner | One workflow invocation and artifacts | [`runtime.workflow_runner.WorkflowRunner`][] |
| Supervisor | Managed-run lifecycle | [`runtime.managed_runs.ManagedRunSupervisor`][] |

## Traps

Separate instances do not share accepted memory.
Separate chat sessions within one instance do share accepted memory.
Bus delivery has no durable replay mechanism.
The console trace records observation, not a transaction log for bus recovery.

## Go deeper

Read [one exchange](../workflows/exchange.md), [cognition](../subsystems/cognition.md), and [operator service](../subsystems/console.md).

