# Current implementation audit

## Summary

The current repository has useful durable state and workflow boundaries.
Several cognitive outputs never affect accepted state.
Minecraft is a one-action adapter.
Conversation prompts describe capabilities beyond the active runtime.
These gaps explain much of the prototype's shallow behavior.
This page preserves the planning audit, rather than a new runtime snapshot.
Subsequent evidence appears in the [A01 receipts](baseline/receipts.md) and [A02 verification](../../subsystems/decisions/verification.md).
The current [model policy](model-policy.md) uses the LLM by default after A02's poor Laya quality results.

Parent: [alpha plan](index.md).
Design response: [cognition](cognition.md) and [Minecraft](minecraft.md).

## What already works

Ghost owns accepted state, detached snapshots, optimistic revisions, and checkpoints.
Its commit precedes normal publication.
The console exposes operation, traces, memory inspection, and task controls.
Managed runs have durable receipts, instruction revisions, cancellation, and terminal evidence consolidation.

Main cognition runs seven child workflows.
The workflow engine supports finite routing, scoped agent sessions, child workflows, journals, and external state policies.
The coding workflow demonstrates reviewed task trees.
These boundaries should remain.

Current construction lives in [`main.build_app`][].
Managed-run configuration uses [`runtime.managed_runs.WorkflowSpec`][].
The cognitive parent contract is [`workflows.main_cognition.schemas.CognitiveReport`][].

## Observed gaps

| Area | Current source evidence | Practical consequence | Delivery |
| --- | --- | --- | --- |
| Affect and goals | Appraisal and goal stages return proposals. `pipeline.run_cycle` omits them from `CognitiveReport`. | Later state cannot reflect those proposals. | A06–A07 |
| Event identity | `modules.MainCogLoop._stimulus_from` labels ordinary knoxel inputs as `user.input`. Task events lose detailed progress. | Private thoughts and world evidence can receive the wrong interpretation. | A06 |
| Active goals | `prepare_input` takes intention text without resolving active lifecycle state. | Completed or stale intentions can continue steering work. | A06 |
| Memory | `modules._memory_candidates` selects recent dialogue, important facts, and recent narratives. | Relevant older experiences often remain outside cognition. | A08 |
| Internal thought | Real-mode scheduling disables `InnerThoughtLoop`; its compatibility role emits a fixed thought. | There is no real independent thought process. | A17 |
| Public voice | Story prompts use story-writer framing, third-person speech, and thoughts as feelings. | The voice does not reliably express durable affect. | A09 |
| Capabilities | Static text-only descriptions and global physical-verb restrictions coexist with Minecraft tools. | Legitimate game experience and invented embodiment are difficult to distinguish. | A09 |
| Dependencies | `resolve_same_cycle` releases dependencies based on earlier dispatch, rather than task success. | Work can start before its prerequisite actually finishes. | A06, A11 |
| Managed agents | `WorkflowSpec` and the worker expose Python tool wiring, without the full agent/MCP child configuration. | A pm-coder Minecraft actor needs explicit runtime integration. | A10 |
| Minecraft | `workflows/minecraft/schemas.py` fixes `max_actions` to one. The manifest stops after one Python role. | There is no sustained skill session or subtask tree. | A10–A14 |
| Visual evidence | The body adapter preserves screenshots. The current text planner does not interpret them. | An image artifact does not establish visual perception. | A10 |
| Reflection | Prompts describe meta capabilities without a live source-inspection and belief-update loop. | Self-description can outrun implementation. | A18–A19 |
| Publication recovery | The bus is in memory. Accepted state can precede a crash before publication. | Restart can lose an accepted action or public reply. | A06, A12 |
| Reply failure | An ontology rejection produces telemetry that main cognition does not fully consume. | A user turn can end without a useful reply or recovery. | A09 |

A06 closed the main state path for the affect, goal, event identity, active
goal, and dependency rows. Appraisal and goal proposals now commit with the
attention frame. Lifecycle startup and shutdown evidence is durable. See
[accepted cognition state](../../subsystems/cognition-state.md).
The inner-loop and recovery parts of those rows remain open.

Inferred rationale: the audit table stays as a planning record. The delivery
column names the item that owns the repair, not the current status.

A10 closed the sustained-session and managed-agent configuration gaps.
The [actor](../../subsystems/minecraft-actor.md) now uses a native pm-coder session and a process-local body lease.
The real five-tree return gate passed with concurrent inventory dialogue, instruction changes, and cancellation.
Generic task trees and durable project verification are delivered by A11.
Durable body recovery across processes remains A12 work.
Screenshots remain artifacts; A10 does not establish visual interpretation.
See the [retained verification](../../subsystems/minecraft-actor-verification.md).

A11 adds generic goal/story/leaf trees, independent scope review and world
surveys, prerequisite certificates, fresh specialist sessions, revision and
replanning routes, and independent evidence recomputation. The real birch-house
run accepted nine construction and usability leaves before a later entry actor
returned invalid native output. A separate 83-cell inspection then passed every
house structure, door, air, support, and alive check without changing the world.
The retained [A11 record](../../subsystems/minecraft-task-verification.md) keeps
the failed attempts beside the accepted evidence.

The rich mental-state models already contain emotions, needs, personality, and social appraisal.
Their existence does not establish that every field participates in live behavior.
Repair the active state path before adding more state dimensions.

## Verification on 2026-10-03

The root foundation suite passed: **257 tests in 363.527 seconds**.
The existing documentation passed a strict MkDocs build before this plan.
The suite excludes nested live model and Minecraft gates.
These results establish a useful engineering baseline, not companion quality.

Historical conversation acceptance covered ten scenarios and twenty exchanges per scenario.
Historical Minecraft acceptance demonstrated narrow action and evidence handling.
Neither result establishes sustained survival, current emotional continuity, or the proposed alpha.

No Laya inference request or live Minecraft session was run during this planning audit.
Previous repositories were reviewed through source and history.
Their runtime health remains unverified.

## Snapshot boundaries

Current HEAD: `54149fc`.
Existing console, memory, Minecraft, and documentation changes were uncommitted during review.
The audit includes that working tree.
A01 must capture the implementation snapshot before coding sessions begin.

Locked dependencies point to pm-workflows `d76c5aa` and pm-coder `0efcafb`.
Their local repositories matched those revisions during review.
Package upgrades must preserve this reproducibility.

A01 defines the repeatable [baseline procedure](work-items/a01.md).
The [A01 receipts](baseline/receipts.md) record subsequent runs, live failures, and the implementation snapshot.
