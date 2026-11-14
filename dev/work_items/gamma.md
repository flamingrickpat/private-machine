# Gamma: work after the continuous-character milestone

Status: opened by the owner on 2026-11-14. Only the items below are named. No gamma design is accepted yet.

Gamma is the milestone after [beta](beta.md). Beta is still in progress: B05 is in close-out, and B06 and B07 have not started.
This file is a development record. It describes no implemented behaviour.

| Item | Status | Subject |
| --- | --- | --- |
| G01 | Accepted as the first gamma work item | Reflective inquiry work deferred from B05 |
| G02 | Proposed by the owner, not designed | Memory reconsolidation and sleep |

## G01: reflective inquiry work deferred from B05

The owner moved this work out of B05 on 2026-11-14, so that B05 can close on behaviour a person can review.
The reasons and the audit are in [the B05 record](beta-b05.md#audit-and-close-out-decisions-2026-11-14).
The four parts concern how the companion examines and adjusts its own way of working.
They are not needed for the companion to notice and report its own recent thinking; B05 close-out covers that.

### 1. A graph step that commits a reflection decision

Today a `reflection_review` step in the inquiry graph reads the recorded experiments, the retained episodes, and their sources.
It cannot change the experiment ledger. The reflection child that produces a `ReflectionDecision` runs only in the retained linear path.

The blocker is a persistence boundary, observed in the source during B05:
every graph commit is an append-only recovery root, and `stage_decision` replaces an existing experiment record.
A recovery-root commit rejects replacements.

Owner decisions this part needs before any design:

- Where an experiment transition is accepted: a derived, non-recovery-root commit like the one `ensure_inner_turn` makes, or another boundary.
- What the operation record retains, so a repair can replay the decision without paying for the model calls again.
- Whether three extra model calls on each reflection step are acceptable.

### 2. Capability doubt in the graph

`triggers._capability_doubt` fires only in the linear path, because the graph evaluates its trigger policy without inspection evidence.
Giving the graph that evidence creates new private opportunities and new paid calls on private ticks.
That is a behaviour change, so it needs its own measured decision.

### 3. A step proposes the next recall query (decision 3B)

The owner accepted this direction during B05 as a separate pass.
An inquiry step would propose the query of the next `memory_search` step.
It changes the step contract of every inquiry, so every recording must be recorded again.

### 4. A sample that uses the extended reflection citation grant

A reflection step extends its citation grant with the reflection records.
No recorded sample has cited a record that only the extension permits, so the extension is present but unobserved.
This part is evidence work. It needs a scenario whose reflection material holds a record outside the origin grant.

### Starting G01

G01 needs an architecture proposal first, as `AGENTS.md` section 5 requires.
Inspect the current source at that time. The observations above describe `3c2640b` and can be stale.
No part of G01 has run, and no part is implemented.

## G02: memory reconsolidation and sleep

The owner said on 2026-11-14 that this will probably be the second gamma item.
It has no scope, no design, and no accepted requirements yet.
Record the owner's intent here before any proposal is written.
