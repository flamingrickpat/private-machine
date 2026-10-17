# Deliberate mental control

## Summary

A18C separates a personal decision to stay quiet from an actual suspension of routine inference.
It also develops agentic recall, temporary focus advice, and bounded private thought.
Waiting, deliberate mental work, and bounded private episodes are now wired.
Focused full-canon checks now cover these mechanisms and their recovery boundaries.
The owner accepted A18C for alpha on 17 October 2026.
Detailed human ratings for recall usefulness, voice, and immersion remain unassigned.

Parent: [A18C](work-items/a18c.md).
Runtime: [inference waits](../../subsystems/inference-wait.md).
Related: [private loop](../../subsystems/private-loop.md), [reflection](reflection.md), and [cognitive sources](cognitive-sources.md).

## Mental actions and accepted state

The intended mental workflow uses a real pm-coder session with scoped memory and goal reads.
A natural-language instruction selects the mental work.
The agent can refine a query, inspect originals, compare activities, and propose goals or focus advice.
Its result remains a private proposal.

The proposed path is:

```text
selected mental action -> scoped agent session -> attributed private proposal
  -> ordinary appraisal -> codelets -> attention -> goal and focus acceptance
```

Structural checks enforce grants, source revisions, cancellation, and bounded influence.
They do not choose a hobby, desired mood, or compliant social response.
The accepted workspace remains the result of ordinary competition.
Focus advice cannot remove an outside stimulus from its reserved place.
An urgent concern can defeat reduced-focus advice.

The assembly advertises and instantiates `focus_management`.
It serves scoped memory reads and a paged goal read over a real bound pm-coder session.
Final validation reads the inspected set instead of granting a new inspection.
Boundary tests cover the report write, host commit, and stale-source rejection.
The first real session completed recall but failed final validation because the manifest omitted `mental_inspected`.
The repaired manifest grants that tool; the failed receipt remains visible.
Provider errors and session deadlines now retain failed action results without applying a plan.
The action receipt and private result commit together, and restart replays unappraised results without another agent session.
Real timeout and authentication-error runs now retain failed results and replay them after checkpoint reopen.
The normal full-canon recall run committed a proposal with inspected original sources and revised queries.
Main appraisal accepted the plan and bounded focus advice.
Real attention nominations received the bias while the direct stimulus kept its reserved place.
The private loop then suspended a concern and retained its parked sources in the next request.
Separate real gates passed continued episodes, urgent competition, and focus expiry.

## Private episodes

The intended episode has a stable identity, a source frontier, operation history, and explicit resource limits.
Each consequential step returns through main appraisal before the next private operation reads accepted state.
An acknowledgment must name the accepted thought and survive arrival before the waiter starts.
No model call or acknowledgment wait holds the memory-store lock.

Continuation can request evidence, compare another option, change direction, develop an artifact, or yield.
Step, elapsed-time, and stalled-progress limits end an episode explicitly.
A model's progress rating alone cannot establish new work.
Rephrasing a decision is not new evidence or a new option.

The private host now runs bounded episodes.
Each accepted step commits a durable episode identity, a sequence, an operation history, and an evidence frontier.
A `continue` step waits for main appraisal before the next step reads accepted state.
Step, elapsed-time, and stalled-progress limits close an episode explicitly.
A suspension or a stall parks the concern's sources for a bounded review time.
New outside input, a superseded generation, or an expired acknowledgment wait ends an episode.
The pipeline rejects an unknown continuation, an unknown next operation, and a missing next operation.
It downgrades self-rated progress that names no concrete difference from earlier episode work.
The main pipeline always reaches action selection for an admitted cycle.
Suppressing that stage is still inactive, because no owner defers the external step.
An explicit main `Sleep` still enforces a wait.

## Record links and future consolidation

[`pm.records.lineage.RecordLineage`][] adds optional episode links to records.
It retains operation, sequence, parents, sources, goals, topic hints, and reality attribution.
[`pm.memory.write_batch.WriteBatch.annotate_lineage`][] can annotate only records created in that batch.
It cannot assign a new history to an existing source.

Main records currently retain appraisal and action links.
Wait and timeout records retain their scheduling causes.
Private steps retain their episode identity, sequence, parents, source frontier, goals, and reality label.
Lineage supplies no visibility grant and no truth label.
The accepted memory tools exclude a derived record whose required sources are hidden.

The older memory subsystem contains useful topical and temporal consolidation ideas.
Its source is `C:/source/pm/pm_next/pm_core/subsystems/memory/`.
That source review does not establish runtime health.
A18C does not port its clustering or overnight maintenance.

## Future direction

On 17 October 2026, the owner identified a possible later design with an ad hoc MCP for cognitive control.
That design also uses a workflow of real agents for deliberate work.
This is a future direction, not an implemented interface or an A18C acceptance requirement.

## Research constraints

These sources motivate mechanisms, not acceptance claims:

| Source | Selected physical PDF pages | Use |
| --- | --- | --- |
| *Modeling Long-Term Intentions and Narratives in Autonomous Agents*, DOI `10.1142/S2705078521500107` | 13, 20–22, 30–33 | Interruptible intentions, bounded deliberation, and limited attention influence |
| *How Metacognitive Architectures Remember Their Own Thoughts*, arXiv `2503.13467v1` | 29–30, 40, 47–48 | Inputs, alternatives, causal traces, and limits of architecture comparisons |
| [MIDCA](https://ojs.aaai.org/index.php/AAAI/article/view/9886), DOI `10.1609/aaai.v30i1.9886` | 1–3 | Object-level action and meta-level processing control |
| *Towards an Ontology for Robot Introspection and Metacognition*, DOI `10.3233/FAIA231137` | 3, 6, 9, 11–12 | Explicit inspection questions and information lineage |

Local PDFs remain under `C:/source/pm/pm_next/documents/pdfs/`.
The older thought source remains under `C:/source/pm/pm_next/pm_core/subsystems/thought/`.
The local handoff retains filenames, extracted research, and more precise source landmarks.
No historical code is copied wholesale, and no new ontology dependency is required.
