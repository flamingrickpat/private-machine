# private-machine engineering manual

## Summary

private-machine connects cognitive workflows through an in-memory bus.
The memory store owns accepted memory and checkpoints.
The operator console controls isolated instances and records their execution evidence.
This manual explains current behavior, test procedures, and the earlier implementation plan.

## Start here

| Task | Guide |
| --- | --- |
| Create a character and optionally import a sample history | [Example characters](workflows/example-characters.md) |
| Run your own test Spamuel with a local LLM and Minecraft | [Jesse quickstart](workflows/jesse-quickstart.md) |
| Start an instance and send a message | [Operator console](workflows/console.md) |
| Change code and compare results | [Development loop](workflows/development.md) |
| Follow a message through cognition | [One exchange](workflows/exchange.md) |
| Inject or change a memory | [Memory experiment](workflows/memory.md) |
| Understand runtime ownership | [System map](concepts/architecture.md) |
| Compare the plan with current code | [Plan and implementation](work_items/plan.md) |
| Develop the companion alpha in ordered sessions | [Companion alpha plan](work_items/alpha-plan/index.md) |
| Prepare a continuing character for daily use | [A19 pre-trial sequence](work_items/alpha-plan/index.md#current-development-priority) and [personal trial](work_items/alpha-plan/personal-trial.md) |
| Plan faster cognition experiments | [Next experiments](work_items/experiments.md) |

## Main boundaries

The application constructs modules in [`workflows.companion.build_app`][].
Each module owns a mailbox and a thread.
Cognition runs one workflow with twelve typed phases before it dispatches actions.
Story dialogue and Assistant dialogue use separate prompts and tools.

The console deploys local process definitions.
It does not install remote services or containers.
Each instance owns a separate Ghost checkpoint.
Chat sessions within one instance share that Ghost.

## Read deeper

- [Cognition](subsystems/cognition.md) explains the stage order and action boundary.
- [Ghost and bus](ghost.md) explains accepted state and message isolation.
- [Models](models.md) explains activation and configuration.
- [Workflow runtime](runtime.md) explains direct debugger calls and journals.
- [Managed work](subsystems/managed-work.md) explains updates, questions, cancellation, and consolidation.
- [Evidence and quality](concepts/evidence.md) explains what the tests establish.
- [Glossary](concepts/glossary.md) defines the canonical terms. [Naming rules](concepts/naming.md) apply them.
- [Structural refactor](work_items/refactor/index.md) describes the target layout and its phases.

## Documentation system

The build generates API pages from public Python modules.
The reference excludes tests, private modules, and command-only demonstration scripts.
Concept pages link to exact objects in that reference.

The site uses direct HTML links for local file access.
This follows the [MkDocs configuration](https://www.mkdocs.org/user-guide/configuration/) for `use_directory_urls`.
The reference uses [mkdocstrings-python](https://mkdocstrings.github.io/python/usage/).

Build instructions appear in the [development loop](workflows/development.md).
