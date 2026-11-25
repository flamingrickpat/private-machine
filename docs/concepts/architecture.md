# System map

PM gives a companion a persistent viewpoint. It keeps track of what happened,
what matters now, and what the companion intends to do. These are computational
models of cognition.

## Data

```mermaid
flowchart LR
    A["Chat or world event"] --> B["Stimulus"]
    B --> C["Triage"]
    D["Memory records"] --> E["Selected evidence and workspace"]
    C --> E
    E --> F["Reply or action request"]
    F --> G["Public output or tool outcome"]
    G --> D
```

[Memory](../subsystems/experienced-memory.md) stores events and experiences.
[Context planning](../subsystems/context-planning.md) prepares evidence for a prompt.

## Execution

```mermaid
flowchart LR
    A["Triage this input"] -->|"Consequential"| B["Main cognition"]
    A -->|"Routine exchange"| C["Ordinary interaction"]
    B --> D["Action request"]
    C --> E["Public dialogue"]
    D --> E
    D --> F["Managed workflow"]
```

The bus passes messages to the modules that own them.
[Input triage][workflows.input_triage.host.InputTriage] judges the new event.
[Ordinary interaction](../subsystems/ordinary-interaction.md) can select a local reply.
[Main cognition](../subsystems/cognition.md) can revise the workspace and release
[managed work](../subsystems/managed-work.md).

## Evidence

```mermaid
flowchart LR
    A["Accepted memory and cycle IDs"] --> D["Debugger"]
    B["Workflow receipts and prompt logs"] --> D
    C["Profiler spans and chat"] --> D
```

Use the [overnight Spamuel recording](../workflows/spamuel-baseline.md) to see
how a visible reply connects to prompts and workflow decisions.
