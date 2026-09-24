# Private Machine: implementation handoff

## Purpose

Build the first real companion in this repository, not another prototype repository.
Immersion, agency, freedom, and novelty are the product goals.
The immediate goal is a dependable foundation with small, real cognitive workflows.
This document specifies future implementation. It does not claim that these features already work.

Plan date: September 24, 2026.

## Start here

1. Read this page.
2. Read [architecture and wiring](docs/plan/architecture.md).
3. Read [runtime contracts](docs/plan/runtime.md).
4. Read [models and memory](docs/plan/models-memory.md).
5. Read [workflow behavior](docs/plan/workflows.md).
6. Read [test requirements](docs/plan/tests.md).
7. Read [commits and GitHub delivery](docs/plan/delivery.md).
8. Implement [work items](docs/plan/work-items.md) in order.

The detailed contracts are deliberate. Do not replace them with another framework.
Keep application wiring separate from execution details and cognitive behavior.
Use the `hackable` and `simple-english` skills.
The user's explicit requirements for ABCs, test suites, and topology simulation override conflicting skill defaults.

## Decisions that are fixed

- Keep one shared Ghost and independent shell modules with private bus inboxes.
- Use real knoxel models and the existing Spamuel seed.
- Use the real `pm-workflows` kernel and `pm-coder` Bash Machine.
- Support real folders and virtual working targets through the same interfaces.
- Keep Ghost projections virtual. Do not require a complete filesystem export for each run.
- Give every module a synchronous debugger path that also serves its thread loop.
- Run the cognitive stages as serial child workflows, not separate module threads.
- Use Python scripts for direct generation. Not every workflow needs a coding agent.
- Keep Story and Assistant dialogue as separate workflows with different message construction.
- Include a limited communication steward for active workflows.
- Limit self-management to goals, priorities, and active workflows.
- Use one shared embedding model: `perplexity/pplx-embed-v1-0.6b` through OpenRouter.
- Preserve compatible embeddings from `C:\Users\rick\.pm\embeddings`.
- Keep separate `Default` and `UnitTestEvaluator` chat presets.
- Treat missing required model endpoints as test failures, never successful skips.
- Do not use inference latency as a quality score, deadline, or reason to remove a cognitive stage.

The earlier local EmbeddingGemma proposal is superseded.
No embedding launch script is necessary for the selected remote service.
No API key belongs in these documents, prompts, snapshots, or committed config.

## First delivery

The first delivery contains the foundation, a real cognitive loop, both dialogue modes, self-management, and limited stewardship.
It includes standalone workflow tests and ten multi-turn conversation scenarios.
External workflow management first uses a small real local workflow.
Minecraft integration follows as its own work item.

Deferred features are the internal thought generator, full doctor repair, prompt editing, coding integration, HTTP bus transport, containers, and optimization passes.
The design leaves extension points for these features. It does not implement placeholders that pretend to perform them.

## Authority and scope

This plan supersedes the dummy-only requirements in `HANDOFF.md`, `IMPLEMENTATION.md`, `ADAPTER_NOTES.md`, and `DIRECT_TESTS.md`.
Those files remain useful historical evidence.
The current README describes the existing dummy implementation, not the target architecture.

Work in `C:\source\pm\pm_next_v2`.
Make required generic integration changes in the local `pm-workflows` checkout on branch `main`.
Make required Bash Machine and session-isolation changes in the local `pm-coder` checkout on branch `main`.
Preserve existing changes in all repositories.
Do not change Hadal task folders, workflow state, or its running environment.
The user authorizes tested commits and ordinary pushes in the required PM repositories.
Publish dependencies first, then update this application from GitHub through uv.
Commit the application lockfile and push application changes to its existing `master` branch.
Follow the [delivery procedure](docs/plan/delivery.md) for versions, tests, and commit evidence.
Do not replace seed data or remove task evidence without a separate request.

## Completion evidence

A feature is complete only after its specified real path passes its tests.
A dummy result does not establish live model behavior.
A conversation test does not establish Minecraft behavior.
Record the tested revisions, commands, artifacts, failures, and outstanding work.

The original cognitive-planning session performed source inspection and documentation checks only.
The later delivery setup also publishes existing dependency support and creates a GitHub-backed uv environment.
Its prototype tests do not replace the proposed live cognitive acceptance suite.
