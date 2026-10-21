# Memory consolidation before the personal trial

## Summary

Consolidation turns detailed experience into compact memory that supports later choices.
It preserves what mattered, what changed, and what remains unresolved.
The companion can manage this background work through ordinary cognition.
The design uses one durable cursor and complete atomic windows.
This behavior is planned in A19B.

Parent: [alpha plan](index.md).
Work item: [A19B](work-items/a19b.md).
Transaction design: [atomic consolidation windows](consolidation-commits.md).
Current behavior: [experience and artifact consolidation](../../subsystems/memory-consolidation.md).

## Existing mechanisms and reuse

[`workflows.main_cognition.experience.consolidate_cycle`][] creates experience clusters from accepted inputs and affect.
Those clusters largely copy the input text.
[`workflows.memory_consolidation.artifact_consolidation.Consolidator`][] imports task artifacts into accepted memory.
Keep both paths functional during A19B.
Neither path establishes complete hierarchical consolidation.

The older `pm_next` consolidator supplies source material for topical clustering, temporal summaries, fact extraction, and narrative refinement.
Its path is `C:/source/pm/pm_next/pm_core/subsystems/memory/memory_consolidation.py`.
It directly mutates the older store and lacks cooperative cancellation boundaries.
Review and adapt its mechanisms to the current write-batch API.
Source review does not establish that the older runtime works.

## Source channels

The first partition separates reality scope and channel identity.
Topical clustering then operates within each partition.

| Channel | Membership |
| --- | --- |
| Conversation | Attributed utterances within one conversation, including both sides |
| Internal thought | Executed private thoughts and their relevant decisions, expectations, and consequences |
| Container agent | Companion instructions, subordinate-agent replies, and supported task results within one work session |
| Other experience | Supported introspection, reflection, and world or workflow events under explicit source rules |

Channel labels describe sources, rather than model-selected topics.
Use existing conversation identity and episode lineage where they exist.
The console currently writes `session`, while `Feature.session_id` reads `session_id`.
Reconcile that contract before relying on it for conversation separation.
Keep missing historical identity explicit instead of inventing conversations.

Internal and dialogue clusters can cover overlapping times and record-ID ranges.
Actual source IDs identify membership.
An ID range alone cannot distinguish the interleaved channels.
Related topics can connect through graph edges without merging their attribution.

## Full processing of a window

One bounded input window passes through channel partitioning and embedding-based agglomerative clustering.
Each topical group receives a compact summary.
Focused calls extract supported facts and relationships from the relevant sources.
The same window updates affected temporal summaries and the record graph.
All accepted results retain the source links needed for inspection.

Temporal summaries cover the input available so far.
Later windows can revise an open day, week, or month.
The cursor does not wait for every calendar period to close.
Existing ancestor revisions participate in the write batch's conflict checks.

The workflow must support small inputs that cannot form multiple clusters.
A singleton can still carry important experience.
An empty extraction can be a valid result when the sources contain no supported fact or relationship.
Missing stages and unusable model output cannot masquerade as empty successful results.

Keep the code direct.
Do not optimize clustering cost without evidence that it blocks use.
Respect prompt capacity and cancellation needs when selecting input windows.
Do not reduce necessary reasoning to fit one large prompt.

## Meaning and normal recall

A summary retains the gist and result of thought without copying every intermediate sentence.
Preserve commitments, unresolved concerns, uncertainty, corrections, chronology, and emotional significance.
Preserve the difference between an executed worry and the event it predicts.
One source repeated through several summaries remains one source of support.

Consolidation does not own active goal state or an A19 experiment's lifecycle.
Its derived records summarize those mechanisms and link their authoritative records.
Graph relationships retain the same attribution and uncertainty as their supporting memory.

Ordinary context uses compact summaries and relevant facts.
Deliberate retrieval can inspect original sources or descend the hierarchy.
[`pm.records.memory_records.MemoryClusterRecord.get_story_element`][] currently expands topical clusters into source events.
A19B must inspect consumers instead of assuming that a stored summary saves context.
Keep original records for alpha diagnosis and detailed recall.

## One mental state and one measured need

[`pm.records.mental_state.StateNeeds`][] is the canonical needs group in the mental-state vector.
A19B removes duplicate live needs and cognition holders from `CycleState`.
Existing seeds and checkpoints retain an explicit import path.
Compatibility conversion must preserve already valid vector values.

`memory_integrity` measures integration coverage on a documented scale.
Eligible uncovered source material lowers the value.
Committed complete windows raise it.
Ordinary decay or an appraisal's arbitrary delta cannot invent completed work.
Recall correctness remains a separate measurement.

The companion can appraise the measured need and start consolidation itself.
Conversation or other immediate work can interrupt consolidation.
The [atomic-window contract](consolidation-commits.md) defines discard, restart, and cursor behavior.
The wider needs model remains beta work.
