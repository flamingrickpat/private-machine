# Context planning

## Summary

The context planner decides which memory records go into one standalone public reply, within a token budget.
It divides the budget into lanes: verbatim dialogue, focus, earlier times, facts, and narratives.
Since A19F step 10, a reply that main cognition chose does not use the planner.
It reads the workspace shards of its cycle ([shard context](#shard-context)).
Only the standalone Story situation (`python -m workflows.dialogue_story.run`) still plans its context.

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
| Plans | `workflows/public_speech/context_plans.yaml` | The `story` plan: budgets, lanes, weights. |
| Integration | `workflows/public_speech/context_planning.py` | Scope, stored record vectors, focus queries, and conversion to dialogue turns and evidence. |
| Caller | [`plan_story_request`][workflows.dialogue_story.context.plan_story_request] | Plans the standalone Story situation. |
| Shard context | [`shard_context`][workflows.public_speech.shard_context.shard_context] | Builds the context of a reply that a cycle chose, from the shards of its action request. |
| Index tool | `dev/tools/prepare_context_index.py` | Embeds every missing record and saves a pickle checkpoint. |

## Shard context

[`PublicDialogue.reply_context`][workflows.public_speech.public_dialogue.PublicDialogue.reply_context] calls [`shard_context`][workflows.public_speech.shard_context.shard_context] with the typed [`ActionRequest`][workflows.main_cognition.schemas.ActionRequest]:

| Part | Source | Goes to |
| --- | --- | --- |
| Transcript | The dialogue in the tail of the partner shard, and the private thoughts of the self tail that belong to that exchange ([`exchange_thoughts`][workflows.public_speech.shard_context.exchange_thoughts]), in world-time order | Story: assistant turn, verbatim. Assistant: chat messages, speech only. |
| Shard story | Each shard, rendered with [`shard_block`][workflows.main_cognition.story_prefix.shard_block] as in the story prefix of the cycle, without the verbatim lines | User turn |
| Evidence | Every item and tail record of the shards, with the lane `shard:<name>` | Speech review and the commit guard |

A thought belongs to the exchange when it is not older than the oldest dialogue line of the transcript.
Of these thoughts, the newest `DIALOGUE_THOUGHT_LIMIT` (12) stay.
Older thoughts outside this selection do not appear in the reply context.
The self tail holds 6k tokens of thoughts. In the A19F step 10 replay, about 88 thoughts from earlier days filled the transcript and brought back their old wording, for example "she" for Jesse.

### Empty history records

The shard context omits empty dialogue, thoughts, and inner monologue from the transcript.
Whitespace-only content, including Unicode whitespace, also supplies no turn.
The filter runs before exchange selection, so empty records cannot displace useful thoughts or extend the exchange boundary.
The context retains each empty record in evidence with its original ID, revision, and provenance.
Publication guards therefore still detect changes to those records.
The filter does not change accepted records or invent replacement text.
Missing shard records still raise `KeyError`.

`PlannedContext.trace.empty_transcript_ids` lists omitted record IDs.
`trace.transcript` counts actual turns after selection.
The standalone planner already excludes empty records through [`scoped_records`][pm.memory.retrieval.scoped_records].

The focused live checks use the full Spamuel seed and the real Story host, kernel, review, commit, and bus delivery.
The normal case runs separately from the case with added empty history records.
The same module checks captured syntax repair through the cognitive tool grant and asynchronous schema repair.
It also reproduces an empty local prefill and an invalid review span with real inference, then checks their recovery.

```powershell
uv run --locked python -m unittest tests.live.test_runtime_stability -v -b
```

The default connection is `http://127.0.0.1:8080/v1`, model alias `qwen`.
`PM_STABILITY_LLM_URL` and `PM_STABILITY_LLM_MODEL` select another running model.
No startup script runs, and an unavailable endpoint fails the checks.
Completion caching stays disabled during these checks.
Prompts, rejected outputs, repaired outputs, public contexts, reports, and transport counts stay under a fresh `.runtime/stability/` directory.
`PM_STABILITY_EVIDENCE` selects a new output directory explicitly.
These checks establish execution boundaries. They supply no human quality rating or full-application acceptance.

The shards already have token budgets (`WORKSPACE_SHARD_TOKENS`, `WORKSPACE_TAIL_TOKENS`) and content requirements.
So the reply reads the same conscious content as the decision, and no record is retrieved again.
The current stimuli are not in any tail. The prompt shows them as the current line.

## The focus of a standalone plan

[`FocusInput`][workflows.public_speech.context_planning.FocusInput] holds the parts of the focus:

| Part | Weight | Source |
| --- | --- | --- |
| Workspace text | 1.0 | `workspace` of the situation |
| Current message | 1.0 | The situation message |
| Attention texts | 0.7 | Optional texts |
| Active goals | 0.3 | The active intentions |

Each part becomes a weighted query: an embedding from the model and distinctive terms for lexical scoring.
Required ids fill the `focus` lane.
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

## Embeddings

Records score semantically from the vectors stored on their knoxels.
Startup prepares every missing vector and saves the full state with pickle.
The planner never reads vector cache files or embeds a bounded candidate subset.
It embeds its temporary focus queries directly.
If the query embedding fails, the plan uses lexical scoring and records `semantic.status: unavailable`.

Each invocation normalizes the supplied vectors for its lane calculations.
It reuses these local calculations across lanes and budget rounds.
The durable vector owner remains the knoxel. A later invocation reads the current record vector.
Read [model connections](../models.md#embeddings-and-scoped-retrieval) for startup, FAISS, and persistence.

## Public context and commit guard

[`build_public_context`][workflows.public_speech.public_context.build_public_context] accepts the shard context or the plan as `planned`.
Its records are the evidence ([`EvidenceItem`][workflows.public_speech.public_context.EvidenceItem]).
Each evidence item keeps its `lane`. `PublicContext.shard_story` keeps the shard story.
The revisions of the planned records and of their source records still guard the commit, as A08 requires.
`PublicContext.plan` keeps the plan trace.

## Inspect a plan

1. Run `python -m workflows.dialogue_story.run show` to print the planned Story prompt for the fixed situation.
2. In a workflow report, read `public_context.plan`. For each lane it gives the budget, the tokens used, the number of selected records, the number of candidates, whether the lane was exhausted, and the first competing records.
3. `rounds` lists each redistribution: the spare tokens and the lane that received them.

## Limits

- Token counts are `tiktoken` (`o200k_base`) estimates of the record text plus 12 tokens.
- Facts are many and short. Without record vectors, lexical scoring selects them weakly.
- Mood-similarity scoring from the old planner (`weighted_mental_state`) is not ported.
- The plans are fixed per workflow. A per-persona plan is not configurable yet.
- A shard context has no separate budget. A reply reads up to two full shards and their tails.
