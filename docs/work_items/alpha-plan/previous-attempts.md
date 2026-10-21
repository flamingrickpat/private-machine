# Lessons from previous PM attempts

## Summary

Four generations contain useful ideas with different levels of implementation.
The current Ghost and workflow boundaries provide the best integration base.
The previous `pm_next` contains valuable cognitive and Minecraft mechanisms.
Port specific mechanisms with acceptance evidence.
Avoid restoring a second competing runtime.

Parent: [alpha plan](index.md).
Current evidence: [implementation audit](current-state.md).

This page is a historical review.
The [curated cognitive sources](cognitive-sources.md) define the current reuse scope.
A16 through A19 contain extracted requirements, so developers need not search the old summaries for architectural decisions.
Current owner direction takes precedence over discarded or superseded historical choices.

## Reviewed history

The original brief is `C:\Temp\pm.md`.
Its W0–W8 implementation is recorded in the [earlier plan](../plan.md).
The following revisions identify the source review.
They do not prove operational health.

| Attempt | Reviewed revision | Useful material | Main limitation |
| --- | --- | --- | --- |
| `private-machine` | `7f9bd04` | Need decay, self-origin stimuli, causal cognitive phases, persistence | `pm_lida.py` combines thousands of lines of state, tools, models, and control. |
| `private-machine-head` | `5fa78b0` | Modular procedures, explicit cognitive stages, thought-tree concepts | README reports a broken application. Several procedure/codelet abstractions remain drafts. |
| `pm_next` | `d5ddda7` | Continuous personality cycle, appraisal math, source kinds, Minecraft actor/replanner, rich codelet catalogue | Legacy and newer runtimes coexist. Some prompts invent embodied experiences. |
| `pm_next_v2` | `54149fc` plus working changes | Durable Ghost, workflow boundaries, scoped tools, console, accepted evidence | Important cognitive proposals and independent thought remain disconnected. |

## Mechanisms to adapt

### Causal personality and appraisal

In `pm_next`, inspect `pm_loop_engine/agentic/personality_cycle.py` and `appraisal.py`.
They preserve stimulus origins and connect structured appraisal to state changes.
Reuse suitable state mathematics and event contracts.
Do not inherit a silent neutral appraisal when a real model request fails.

Keep one canonical state owner in the current Ghost.
The older runtime must not run beside it as another personality authority.

### Minecraft actor and replanner

The older Minecraft workflow distinguishes progress, completion, complication, failure, and waiting.
Its replanner uses observations to revise work.
That is closer to the desired behavior than the current one-action workflow.

Its actor still restricts an individual session to a small action.
The alpha must allow a meaningful subtask to require many calls.
Transfer the outcome taxonomy and recovery ideas, then verify longer sessions.

### Codelets and internal thought

`pm_core/subsystems/codelet/codelet_definitions.py` contains **342** `SimpleCodelet` subclasses.
The catalogue covers ten families, including appraisal, memory, imagination, regulation, identity, attention, and action.
`pathways_detailed.py` describes longer cognitive sequences.

This catalogue provides design material.
It does not demonstrate 342 effective live processes.
Many prompts use a narrator, bodily cues, or invented past experiences.
A16 rewrites a selected repertoire around accepted state and attributed evidence.

The older thought graph provides generation, evaluation, and branching concepts.
Start with a small graph that produces measurable behavioral consequences.
Unbounded branching would spend resources without establishing a richer companion.

### Source reflection

Inspect `pm_core/subsystems/architecture_reflection/architecture_snapshot.py`.
Its root follows the subsystem directory.
Its path categories target a different package layout.
A generated architecture description can therefore describe incomplete or stale structure.

A18 uses the actual repository root, active wiring, revisions, and source hashes.
Descriptions of intended architecture remain separate from observations of active implementation.

## Supporting repositories

| Repository | Retain or adapt | Boundary |
| --- | --- | --- |
| `coding-workflow`, `c32e6bd` | Native task tree, independent planning review, stable selections, specification fingerprints, dependency order | Minecraft cannot undo world changes through Git checkpoints. |
| `pm-workflows`, `d76c5aa` | Finite routes, role result JSON, journals, child contracts, scoped environments, external state policy | Add explicit inputs and decisions inside this engine. |
| `pm-coder`, `0efcafb` | Fresh sessions, scoped MCP, multimodal attachments, context compaction, cancellation | A session can use many calls without an arbitrary action ceiling. |
| `pm-minecraft`, `2afee33` | Mineflayer body, authoritative MCP schemas, observations, screenshots, game actions | It supplies a body, rather than companion cognition. |
| `pm-agentic-babymode` | Cozy survival configuration, nutrition, sleep, diagnostic state | Tests must use its actual mechanics. A world checkpoint is not a complete player rollback. |

The adjacent `deep-swe-agentic` and `model_skill_compiler` repositories were inventoried.
They were not treated as additional demonstrated PM cognitive runtimes.
This review covers the identifiable PM generations and the relevant workflow/body implementations.
