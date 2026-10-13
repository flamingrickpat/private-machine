# Context planning

## Summary

The context planner decides which memory records go into one public reply, within a token budget.
It divides the budget into lanes: verbatim dialogue, workspace focus, earlier times, facts, and narratives.
The current workspace decides what the scored lanes look for.
Story and Assistant use the same planner with different plans.

Parent: [public dialogue](public-dialogue.md).
Related: [Story speech](story-speech.md) and [codelets and attention](codelets.md).
Entry points: [`pm.memory.context_plan.plan_context`][] and [`workflows.public_speech.context_planning.plan_reply_context`][].

## Origin

This is the v2 port of the `ContextPlanner` in `pm_next/pm_core/subsystems/context/`.
The old planner had lanes, budget shares, time boundaries between lanes, and redistribution of unused budget.
Its story preset used a recent window, historical summaries, facts, rules, and narratives.
It did not choose context from the current workspace.
The v2 planner keeps the lane design and adds the workspace focus.

## Where the code lives

| Part | Location | Responsibility |
| --- | --- | --- |
| Planner | `pm/memory/context_plan.py` | Pure lane selection, scoring, boundaries, and redistribution. No store, model, or network. |
| Plans | `workflows/public_speech/context_plans.yaml` | The `story` and `assistant` plans: budgets, lanes, weights. |
| Integration | `workflows/public_speech/context_planning.py` | Scope, embedding cache, focus queries, bounded new embeddings, and conversion to dialogue turns and evidence. |
| Callers | [`PublicDialogue.plan_context`][workflows.public_speech.public_dialogue.PublicDialogue.plan_context] | Builds the focus from the action payload and plans before the public context is frozen. |
| Index tool | `dev/tools/prepare_context_index.py` | Embeds all score-lane records once into the shared cache. |

## The focus

Main cognition sends the winning attention nominations with each dispatched action, under `attention`.
[`PublicDialogue.plan_context`][workflows.public_speech.public_dialogue.PublicDialogue.plan_context] turns the payload into a [`FocusInput`][workflows.public_speech.context_planning.FocusInput]:

| Part | Weight | Source |
| --- | --- | --- |
| Workspace experience | 1.0 | `workspace` in the action payload |
| Current message | 1.0 | The last stimulus |
| Winning nomination texts | 0.7 | `attention[].text` |
| Active goals | 0.3 | The active intentions |

Each part becomes a weighted query: an embedding (through the shared cache) and distinctive terms for lexical scoring.
`attention[].target_ids` and the cognition-selected `memory_evidence` become required records of the `focus` lane.
The current stimuli are excluded from every lane. The prompt shows them as the current line.

## Lanes

A [`LaneSpec`][pm.memory.context_plan.LaneSpec] has a name, a budget share, a selection mode, record filters, and weights.

| Mode | Behavior |
| --- | --- |
| `recent` | The newest contiguous records that fit. The window never skips a record. |
| `focus` | Only the required records of the focus. |
| `score` | Candidates ranked by `semantic * cosine + lexical * term match + importance * importance`, with optional decay per day. The lane fills greedily by score. Records that did not fit are kept in `competing`. |

Further rules:

- A record fills at most one lane. `recent` lanes run first, the others in plan order.
- `before_lane` keeps a lane strictly older than the oldest record of the named lane. Summaries of earlier times therefore never repeat the verbatim window.
- `latest_per` keeps the newest record per key, for example the latest refinement of each narrative kind and person.
- After a pass, lanes that ran out of candidates give back their unused budget. The first lane in `priority` that still had candidates receives it, and the plan runs again, up to `redistribution_rounds` times.

### The story plan (9,000 tokens)

| Lane | Share | Content | Goes to |
| --- | --- | --- | --- |
| `dialogue` | 45% | Dialogue, thoughts, and inner monologue, newest first | Assistant turn, verbatim |
| `focus` | 12% | Records that the workspace points at | User turn |
| `history` | 20% | Topical and temporal summary clusters older than the dialogue window | User turn |
| `facts` | 10% | Declarative facts and beliefs, with an importance weight | User turn |
| `narratives` | 13% | The latest narrative of each kind and person | User turn |

### The assistant plan (6,000 tokens)

The assistant plan uses the same lanes with 35% dialogue (speech only), 15% focus, 20% history, 20% facts, and 10% narratives.
Prior speech becomes ordinary user and assistant chat messages.
Private thoughts stay out of the tool transcript.
Assistant mode is the companion talking while it can also call tools: managed work, Minecraft, or a user-added MCP server such as mail.

## Embeddings

Records score semantically only when their first index chunk has a cached vector.
Without a vector, a record still competes lexically and by importance.
After the first pass, at most `max_new_embeddings` (32) uncached records near each score lane's cut are embedded, and the plan runs again.
If the embedding service fails, the plan scores lexically and records `semantic.status: unavailable`. The reply continues.

Run the index tool once per seed so that every candidate has a vector:

```powershell
uv run --locked python dev/tools/prepare_context_index.py --seed test_data/personas/spamuel/canon.yaml
```

The cache uses the configured `embedding.cache_dir` and the exact record text as its key.
A checkpoint and its seed therefore share entries.
On 2026-10-13 the Spamuel seed had 2,941 score-lane records, and 354 were cached before the first run of the tool.

## Public context and commit guard

[`build_public_context`][workflows.public_speech.public_context.build_public_context] accepts the plan as `planned`.
The lane-tagged records then replace the cognition-selected evidence and the old 48-record source-chain expansion.
Each evidence item keeps its `lane`.
The revisions of the planned records and of their source records still guard the commit, as A08 requires.
`PublicContext.plan` keeps the plan trace.

## Inspect a plan

1. Run `python -m workflows.dialogue_story.run show` to print the planned Story prompt for the fixed situation.
2. In a workflow report, read `public_context.plan`. For each lane it gives the budget, the tokens used, the number of selected records, the number of candidates, whether the lane was exhausted, and the first competing records.
3. `rounds` lists each redistribution: the spare tokens and the lane that received them.

## Limits

- Token counts are `tiktoken` (`o200k_base`) estimates of the record text plus 12 tokens.
- Facts are many and short. Without cached vectors, lexical scoring selects them weakly.
- Mood-similarity scoring from the old planner (`weighted_mental_state`) is not ported.
- The plans are fixed per workflow. A per-persona plan is not configurable yet.
