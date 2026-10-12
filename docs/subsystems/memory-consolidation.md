# Experience consolidation and reported beliefs

## Summary

Cognition stores evidence-linked episodes and accepted emotional consequences in its own commit.
User reports can create or correct attributed beliefs.
Task consolidation retains artifact hashes and deduplicates unchanged material across progress snapshots.
Generated companion claims cannot establish factual memory.

Parent: [experienced memory](experienced-memory.md).
Related: [managed work](managed-work.md), [private ontology](../work_items/alpha-plan/ontology.md), and [Ghost commits](../ghost.md).
Evidence: [A08 verification](../work_items/verification/a08-experienced-memory.md).

## Cognitive experiences

[`runtime.experience.consolidate_cycle`][] groups source events by reality/frame scope.
It creates a topical `MemoryClusterKnoxel` with the original event IDs and their timestamps.
The summary quotes the source records with attribution.
It adds the accepted affect summary and links the appraisal Feature.
The source event remains separate from the internal emotional consequence.

A sad imagined possibility therefore remains imagined.
The accepted sadness remains an actual state change.
Unsupported descriptive emotion terms do not create another state mechanism.
[`pm.memory.retrieval.record_text`][] indexes and recalls the accepted affect summary for internal-state records.
Generated appraisal descriptions remain in the original record for inspection.
They cannot supply factual sensory experience through memory retrieval.
Repeated cognition commit IDs reuse accepted receipts instead of adding another episode.
No separate memory authority or database exists.

## Evidence categories

| Category | Meaning and permitted interpretation |
| --- | --- |
| `background` | Imported canon. The source states this content. |
| `utterance` | Actual attributed speech. Its content is not independently verified. |
| `executed_event` | A host-recorded lifecycle or workflow event. |
| `internal_state` | Accepted affect with its implemented vector and causal trace. |
| `speculation` / `prediction` | Imagined or anticipated events. No completed outside action follows from them. |
| `artifact_report` | A real task file reports this content. The report does not independently prove task success. |
| `reported_belief` | A verbatim live user claim with its original evidence ID. |
| `experience_summary` | An evidence-linked abstraction. Its source categories remain explicit. |
| `interpretation` / `attributed_record` | Other supported records with limited epistemic authority. |

[`pm.memory.retrieval.memory_kind`][] supplies the retrieval interpretation.
Explicit metadata retains existing speculation and prediction attribution.
Body observations remain historical during recall, regardless of their confidence.
Screenshot bytes remain evidence metadata and never enter text embeddings.

## Belief proposals and corrections

The Context LLM can propose [`workflows.main_cognition.schemas.BeliefUpdate`][] objects.
A proposal contains an original `source_id`, verbatim `quote`, stable `belief_key`, and explicit `correction` flag.
[`runtime.experience.update_beliefs`][] accepts only live user dialogue with matching provenance and an exact source quote.
It rejects companion speech, background, speculation, missing sources, and unsupported paraphrases.
Main cognition also restricts belief sources to the current stimulus ID grant.
Rejections remain inspectable in the action's belief-update receipt.

An accepted `BeliefKnoxel` remains `reported_by_user`.
It cites its supporting user utterance and preserves reality/frame scope.
It does not become an independently observed outside fact.
An identical active quote reuses its belief ID.

A different claim with the same scoped key adds conflict evidence.
Without an explicit correction, both claims remain active.
An explicit correction deactivates earlier beliefs and sets `superseded_by_id` to the new belief.
A `BeliefShiftKnoxel` retains the original cause and accepted change.
Neither operation rewrites the original utterance.

The model still chooses the semantic key and correction intent.
Different keys cannot automatically establish a contradiction.
The correction gate therefore verifies key reuse and the resulting evidence links.
Human interpretation remains necessary for ambiguous claims.

## Utterances and intentional deception

[`runtime.experience.utterance_metadata`][] records a prior action schema when Story accepts public speech.
An explicit deception intent requires `presentation.mode=deceive`, a purpose, and visible `known_fact_ids`.
Its action schema must exist before the utterance commit.
The utterance retains the intent ID, purpose, and known evidence.
The false content cannot become a factual belief or a task-completion receipt.

Unsupported sensory or emotional output remains an utterance if another pathway accepts it.
It cannot acquire an invented deception justification afterward.
A08 preserves this memory boundary.
A09 owns consistent public generation and capability grounding.

## Task progress and artifact identity

[`runtime.consolidation.Consolidator.consolidate`][] reads a text and image inventory before proposing memory.
Receipts retain relative artifact paths and content hashes.
Text hashes use decoded UTF-8 content with normalized newlines.
Image hashes use the original bytes.
The run and inventory hash identify each receipt.
Each proposal also receives a `memory_key` from its type, content, and relevant artifact hashes.
The run's earlier receipts supply the deduplication ledger after restart.

A changed observation file adds its new observation report.
It does not recreate an unchanged outcome or result file.
A changed screenshot produces new image evidence.
The existing terminal hook still consolidates completed, failed, or cancelled runs.
Main cognition also consolidates artifact snapshots when it accepts workflow progress events.

[`pm.memory.write_batch.GhostWriteBatch.guard_consolidation_run`][] protects concurrent inventory decisions.
A changed run ledger rejects the stale consolidation batch.
The caller can repeat the operation against the new ledger.
Failed reads or commits retain the artifact folder and accept no partial memory.
Malformed report text remains attributed report text, without promotion to a verified event.

## Temporal activity

[`pm.memory.store.Ghost.activity_summary`][] returns atomic activity within a half-open UTC interval: `[start, end)`.
Naive legacy timestamps mean UTC.
The result contains original IDs, timestamps, categories, provenance, and an evidence-linked textual summary.
It excludes background, predictions, simulations, and artifact summaries from executed activity.
An internal-state entry uses the accepted affect summary instead of unsupported descriptive prose.

The query does not distribute a summary's contents across its whole time span.
A stopped process has no executed work in its offline interval.
Startup and shutdown remain events at their actual recorded times.
The API can answer an interval query without inventing private activity before A17 implements the second loop.

```python
activity = ghost.activity_summary("2026-10-04T01:00:00+00:00", "2026-10-04T11:00:00+00:00", visible_ids=allowed_ids)
```

## Remaining boundaries

Task reports establish artifact contents, rather than verified world effects.
Legacy receipts without proposal keys retain their earlier consolidation semantics.
World observations require fresh tool evidence before reuse as current state.
The code does not infer a past sense from imported prose.
Public immersion ratings remain with the operator.
