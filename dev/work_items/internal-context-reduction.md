# Internal evidence reduction

## Owner-approved scope

The owner requests evidence batching before one final appraisal, and the same repair for other internal workflows with this gap.
Public speech generation, review, fitting, and batching stay unchanged.
If the complete prompt fits, its content and model-call count stay unchanged.

## Current execution and failure

`phases.appraise.run` builds a `StoryPrefix`, adds triage and interval evidence, and calls `phases.model.complete`.
`model_json.complete_json` inserts grounding, then calls `fit_turns` before `pm.llm.structured` invokes the model tool.
The failed real run needed 30,789 mandatory tokens against a 27,852-token input ceiling.
The fitter can omit optional recall. It cannot reduce mandatory evidence.
`batch_prompt` covers independent tasks, but mandatory shared context and one oversized task can still exceed its ceiling.
Main-cognition phases, private inquiry, and reflection share this risk. Attention already has a local structure-digest implementation.

## Target contracts and ownership

- Add `ContextTag.reducible: bool = False`. Evidence owners opt in; instructions, output schemas, media, and repair feedback remain fixed.
- Add internal `evidence_text(text, name, source_ids=()) -> Content` annotations for mandatory evidence.
- Add `fit_internal_prompt(context, turns, budget) -> PromptSelection` in workflow model plumbing.
- Add a workflow-owned reduction module with named digest and coverage contracts. It uses the existing `batch_prompt` planner.
- Retain original source text in immutable request/checkpoint data. Write original prompts, digest outputs, and coverage receipts to kernel scratch files.
- Run one final cognitive judgment after reduction. Never merge appraisal deltas from separate batches.
- Preserve optional selection policy. Reduce only required evidence and required supporting evidence when ordinary fitting fails.
- Split an oversized evidence text into complete, ordered spans for review. Every character enters a review; no clipping or omitted spans.
- Keep source IDs attached to derived evidence. A coverage receipt proves prompt inclusion, not semantic equivalence of a lossy digest.

No durable record, state owner, public workflow result, or persistence schema changes.
Model failures use the real completion boundary and its visible retry/cancellation policy.
Unreducible mandatory instructions, impossible metadata size, and non-convergent reduction remain explicit failures.

## Implementation order

1. Add annotations and shared internal reduction, with an unchanged ordinary-fit path.
2. Annotate appraisal and other growing internal evidence at their builders; preserve annotations through composition.
3. Connect shared-context preparation to existing independent-task batching and reflection selection.
4. Run deterministic planner checks, scoped strict typing, and real configured-model stress calls with captured Spamuel evidence.
5. Resume the beta with frozen repaired source; inspect real appraisal and retained receipts.
6. Update current-system context documentation and build the strict manual.

## Risks and limits

Digests lose detail. Explicit provenance, coverage, contradiction instructions, and unchanged originals make that loss inspectable.
Extra model calls occur only after ordinary fitting fails. A larger context avoids them when it holds the complete prompt.
The separate delivered-speech recovery defect remains outside this repair.
Live prerequisites are `config.yaml`, its configured model and embeddings, the saved beta checkpoint, and the real workflow kernel.
