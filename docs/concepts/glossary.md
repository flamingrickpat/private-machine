# Glossary

## Summary

This page defines the canonical terms of private-machine (pm).
One concept has one name in code, documentation, and tests.
Each entry gives the current name if the code still uses an older one.
The [naming rules](naming.md) and the [refactor plan](../work_items/refactor/index.md) apply these terms.

Parent: [manual](../index.md).

## How to read an entry

- **Definition**: what the term means in this project.
- **Not**: the boundary. What the term does not mean.
- **Relations**: the related terms.
- **Owner**: the package that owns the concept after the refactor.
- **Old names**: names that the code still uses or used before. Do not use them in new code.

Persisted names and names that a model reads keep their old spelling.
Examples are the `__type__` value in a checkpoint, the tool name `ghost_get`, and the virtual path `ghost/knoxels/`.
A rename changes Python identifiers only.

## Product and test data

### private-machine (pm)

- Definition: the product. A persistent companion with cognition, memory, and embodied work.
- Not: a persona. The product contains no persona.
- Owner: the whole repository.

### persona

- Definition: the identity, history, and configuration that the user supplies as a seed.
- Not: product code.
- Relations: a persona is loaded from a **seed** into the **memory store**.
- Owner: `test_data/personas/` holds the test personas.
- Note: Spamuel and Sunny (cheerful) are synthetic test personas. The default seed is still Spamuel until users supply their own persona.

### seed

- Definition: a YAML file with a persona configuration and its initial records.
- Not: a checkpoint. An existing checkpoint always takes precedence over a seed.

### checkpoint

- Definition: the durable, versioned file of one memory store, for example `.runtime/ghost.json`.
- Relations: written after each accepted **commit** and before public output.
- Owner: `pm.memory.checkpoint`. Old names: `GhostCheckpoint`.

## Runtime and workflows

### runtime

- Definition: the `pm/` package. It provides APIs and entry points: bus, bus modules, workflow execution, LLM access, memory store, managed runs, CLI, and console.
- Not: companion behavior. The runtime contains no cognition, speech, or Minecraft rules.
- Relations: workflows depend on the runtime. The runtime never imports a workflow.

### workflow

- Definition: one folder under `workflows/` with a manifest, its scripts, its prompts or skills, and optionally a host binding.
- Not: a single Python function.
- Relations: the pm-workflows **kernel** runs a workflow. A **managed run** can run one in the background.
- Old names: "task" for managed workflows.

### manifest

- Definition: the `<name>.workflow.md` file. Its YAML front matter declares the name, roles, tools, phases, and result contracts.
- Relations: each role names a **skill** path.

### role

- Definition: one named callable entry in a manifest. The kernel calls its skill.
- Relations: a cognition **stage** is a child workflow with one role.

### skill

- Definition: the file that a role points to. Usually `steps.py` with a `run(context)` function, or a prompt file for an agent role.
- Old names: none. Do not call it "step" or "script".

### host binding

- Definition: the `host.py` file of a workflow. It connects the workflow to the runtime: its bus module and the host-side tools that its manifest lists.
- Not: kernel-side code. The kernel never imports a host binding.
- Old names: the module classes in `modules.py`; `module_tools.py`.

### workflow library

- Definition: a folder under `workflows/` without a manifest. Several workflows share it.
- Examples: `public_speech`, `minecraft_body`.
- Relations: a workflow library imports only the runtime.

### assembly

- Definition: `workflows/companion.py`. One explicit list of bus modules, managed workflows, and their wiring order.
- Relations: the runtime entry point `pm.app` runs an assembly.
- Old names: `build_app` in `main.py`.

### kernel

- Definition: the external pm-workflows package that runs manifests, journals runs, and enforces role grants.
- Not: part of this repository.

### bus

- Definition: the in-memory publish and subscribe system. It copies every payload through JSON.
- Not: durable. Bus delivery has no replay.
- Owner: `pm.bus`.

### envelope

- Definition: one bus message with a topic, a correlation identifier, and a causation identifier.
- Owner: `pm.bus.envelopes`.

### bus module

- Definition: a long-lived component with a private mailbox and a thread. It reacts to envelopes.
- Not: a Python module. Always write "bus module".
- Owner: `pm.host`. Old names: `ShellModule`; a workflow-backed one was `ShellModuleWorkflow`.

### application

- Definition: the set of running bus modules and their shared lifecycle and first error.
- Owner: `pm.host.application`.

## Memory

### memory store

- Definition: the owner of all accepted records of one instance. It holds the lock, revisions, snapshots, commits, and the checkpoint.
- Not: a retrieval algorithm. Not the persona.
- Owner: `pm.memory.store`. Old names: `Ghost`, "ghost". The tool names `ghost_*` stay.

### record

- Definition: one persisted memory item with an identifier, a type, scope, provenance, and content.
- Owner: `pm.records`. Old names: knoxel. `KnoxelBase` becomes `RecordBase`, `ActionSchemaKnoxel` becomes `ActionSchemaRecord`.

### snapshot

- Definition: an immutable, serialized view of selected records at known revisions.
- Owner: `pm.memory.snapshot`. Old names: `GhostView`.

### write batch

- Definition: staged record proposals that commit atomically if their expected revisions still hold.
- Relations: a failed revision check raises a **conflict**.
- Owner: `pm.memory.write_batch`. Old names: `GhostWriteBatch`.

### commit

- Definition: the atomic acceptance of a write batch into the memory store and its checkpoint.
- Relations: a commit happens before any **publication** of its result.

### retrieval

- Definition: ranking records of a snapshot for a query by text, recency, importance, and embeddings.
- Not: policy about which evidence is acceptable. That policy belongs to the workflow that uses the result.
- Owner: `pm.memory.retrieval`. Old names: `pm_core.ghost.memory`.

### persona configuration

- Definition: names, character card, and supported and unsupported capabilities of a persona.
- Owner: `pm.records.cycle_state`. Old names: `GhostConfig`.

## Cognition

### cognition cycle

- Definition: one pass of the main cognition workflow over new stimuli, from triage to action selection, followed by one commit.
- Old names: "tick" in some fields. `tick_id` stays as a persisted field.

### stage

- Definition: one of the eight ordered child workflows of a cognition cycle: triage, appraisal, context, codelets, attention, goals, workspace, action selection.
- Not: a role or a skill file.

### mental state

- Definition: the fixed-position state vector with emotions, needs, cognition, relationship, and expression groups.
- Not: a cognition cycle's full state. That is the **cycle state**.
- Owner: `pm.records.mental_state`. Old names: "state", `FullMentalState` stays.

### affect

- Definition: the projected change of mental state from appraisal, decay, and personality.
- Owner: `workflows/main_cognition/affect_dynamics.py`.

### cycle state

- Definition: the persisted state of one cognition cycle.
- Owner: `pm.records.cycle_state`. Old names: `CycleState`.

### global workspace turn

- Definition: the committed record of what entered the global workspace in one cycle.
- Owner: `pm.records.global_workspace`. Old names: GWT turn. The tool name `ghost_gwt` stays.

### action intent

- Definition: an action that the action-selection stage proposes.
- Relations: an intent becomes a **dispatch** after dependency checks.
- Not: an `Intention` record. That record is a goal.

### dispatch

- Definition: a committed action record with a pending envelope that targets one bus module.
- Relations: dispatch recovery replays pending dispatches after an interruption.
- Owner: `workflows/main_cognition/dispatch.py`, `dispatch_recovery.py`. Old names: "intent" in `runtime/intents.py`.

### reserved

- Definition: code that is kept for later work items but is not wired into any path.
- Examples: mental-state verbalization, meta prompts, graph, qualia and codelet records, the needs axes model.

## Speech

### public reply

- Definition: text that the companion says to the user.
- Relations: a public reply is committed, then **published**.

### publication

- Definition: putting a committed reply on the bus. Commit before publication is an invariant.
- Owner: `workflows/public_speech/publication.py`.

### violation

- Definition: one hallucination that the speech review reports in a draft reply, with a kind, an exact quote, and a policy weight. Lies and exaggeration are never violations.
- Owner: `workflows/public_speech/speech_review.py`.

### public context

- Definition: the evidence contract that a public reply may use.
- Owner: `workflows/public_speech/public_context.py`.

## Managed work

### managed run

- Definition: one background execution of a managed workflow, with a durable **run receipt**.
- Old names: task, task folder, `task_root`. The `task_id` field of a project stays.
- Owner: `pm.runs`.

### run receipt

- Definition: the durable row of one managed run: state, revisions, epochs, events, and artifact path.
- Old names: `ManagedRunReceipt`.

### instruction revision

- Definition: the counter of instruction updates to a managed run. `applied_revision` is the last one that a worker consumed.

### execution epoch

- Definition: the counter that fences stale workers. A worker from an older epoch cannot publish or dispatch.

### supervisor

- Definition: the owner of managed-run lifecycle: start, update, answer, cancel, resume.
- Owner: `pm.runs.supervisor`. Old names: `ManagedRunSupervisor`.

### receipt

- Definition: any persisted record of a completed step, cited by path or identifier.
- Note: always qualify it: run receipt, commit receipt, answer receipt, consolidation receipt.

## Minecraft

### body

- Definition: the Minecraft avatar, reached through one MCP endpoint.
- Owner: `workflows/minecraft_body/`.

### body lease

- Definition: the cross-process lock and epoch that gives one run control of the body.
- Owner: `workflows/minecraft_body/lease.py`.

### body gateway

- Definition: the per-call enforcer between a specialist session and the body. It checks the epoch, instruction revision, and schema, and journals each call.
- Owner: `workflows/minecraft_body/gateway.py`. Old names: `ActorGateway`, `runtime/minecraft_actor.py`.

### journal

- Definition: the append-only file of dispatch and result rows of every body call.
- Relations: the dispatch row is written before the call.

### call outcome

- Definition: the classification of one body call result, for example partial progress.
- Owner: `workflows/minecraft_body/call_outcomes.py`. Old names: `runtime/minecraft_outcomes.py`.

### specialist session

- Definition: one model session that controls the body for one leaf, with a specialist prompt.
- Owner: `workflows/minecraft/`. Old names: actor, `run_actor`.

### project

- Definition: a durable Minecraft goal with a task tree, its planning state, and its acceptances.
- Old names: task tree, task, plan (as the whole tree). The workflow name `minecraft_project` stays.
- Owner: `workflows/minecraft_project/`. Old names: `workflows/minecraft_tasks/`.

### plan

- Definition: one tree revision that a reviewer checks.
- Not: the whole project.

### node, goal, story, leaf

- Definition: a node is any tree element. The goal is the root. A story groups nodes. A leaf is executable.

### condition

- Definition: a checkable world fact that a node requires.

### read

- Definition: one recorded read-only world query with its arguments and result.

### certificate

- Definition: the acceptance record of a node. It carries the specification digest, dependencies, and file hashes.
- Relations: only the tree store creates and validates certificates.

### evidence

- Definition: retained observations and receipts used to evaluate a condition.
- Not: an acceptance decision.
