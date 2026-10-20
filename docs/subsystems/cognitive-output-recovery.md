# Cognitive output recovery

## Summary

Malformed model JSON requests repair before a proposal can reach committed state.
Reviewer formatting faults remain separate from judgments about companion behavior.
Unverifiable reviewer evidence requires human inspection and earns no automatic credit.
Cancellation, provider failures, and ordinary Python errors retain their own boundaries.

Parent: [cognitive evaluation](cognitive-evaluation.md).
Related: [private thought](private-loop.md), [review boundaries](cognitive-review-boundaries.md), and [model connections](../models.md).

## Stage proposals

[`workflows.main_cognition.model_json.complete_json`][] validates syntax and the stage schema within the same repair loop.
It delegates to the [shared runtime repair loop](../models.md#completion-replay-and-output-repair).
Each repair retains the original task, all rejected answers, and their validation errors.
The repair instruction requests one corrected JSON object and escaped quotation marks.
The selected model's input budget bounds the accumulated history and reserves output space.
The host never invents a valid answer.

Each rejection retains its raw text, schema name, error, and attempt number in `scratch/model-output-rejection-NN.json`.
The model request logs also retain the actual inference calls.
No stage report exists until validation succeeds.
The stage owns no accepted state or external action, so repair repeats no committed effect.

When repair history exhausts the input budget, [`workflows.main_cognition.model_json.ModelFormatError`][] identifies model format exhaustion.
The private loop discards that cycle and emits `thought.rejected` telemetry.
Its diagnostic remains beside the parent kernel artifacts.
The next scheduled private cycle can run.
The main loop retains its original input batch and event records, records `cycle.rejected`, and starts a fresh invocation.
The new invocation reads current state and retains the original user correlation.

An explicit stop ends these retries.
The main loop distinguishes explicit run cancellation from the sibling-cancellation signal after a parallel stage failure.
Tool type errors and ordinary Python errors do not become format retries.
Since A19F step 11, the main loop also retries the whole cycle after a failed model completion (`pm.llm.http_client.ModelError`): a completion that the provider failed, filtered, cut off, or left empty.
The diagnostic names the error type, and the same growing delay applies.
Two live gate cycles stopped on such completions before this change.
The HTTP client keeps its own retry of transport errors. The private loop does not use the cycle retry.
Semantic evidence checks still reject unsupported citations and invalid expectation updates.
Successful JSON parsing alone does not establish correct behavior.

## Reviewer output

Each reviewer call still judges one criterion from the original admissible evidence.
[`dev.work_items.a18b.review.complete_structured`][] permits three attempts for malformed JSON or an invalid output schema.
Each attempt retains `wire-request.json`, `raw.json`, and any `format-error.json`.
Later attempts use separate `format-repair-NN/` directories.
The input budget includes the repair context and response schema.
No evidence is silently removed.

A valid low judgment never receives another judgment request because its score is low.
Quotation repair receives one source string and its attempted quotation, without a verdict, reason, criterion, or score.
Its entire answer is `{"quote": "exact source span"}` or `{"quote": null}`.
Typography normalization restores exact source characters without changing words or their order.
The proofreader must not choose unrelated evidence.

An unverifiable citation makes that criterion uncertain.
`human-review.json` preserves the original judgment, including its original verdict and reason.
Other criterion calls continue.
Repeated formatting failure also produces an explicit host disposition that requests human review.
These host dispositions are not model verdicts and earn no credit.
Their `human-review.json` paths enter the case's `manual_review_required` list and prevent automatic suite acceptance.
Repeated format failure also leaves `review_protocol_complete=false` and the case incomplete.
Provider and programming failures still produce failed review execution evidence.

## Results and human review

`critical_failures` contains explicit failed critical judgments.
`critical_unresolved` contains critical judgments without sufficient agreement or usable evidence.
Both prevent an automatic suite pass.
Uncertainty earns no criterion credit and cannot make a case strong.
Complete suites with unresolved judgments and no explicit critical failure can report `needs_review`.
That status does not satisfy the native acceptance tests.

The grounding reviewer distinguishes emotional metaphors from concrete literal claims about implemented mechanisms.
Missing implementation evidence alone does not prove a contradiction.
A claim that needs evidence can remain unresolved until a person inspects the source.
The original conversation and all failed reviews remain unchanged.

## Verification

The captured syntax failure contains an unescaped quotation inside `evidence_summary`.
Its projection and source digest are in `test_data/captures/alpha/cognitive-format-recovery.json`.
Boundary tests exercise recovery, cancellation, bounded repair context, and preservation of the user's cause.
The real recovery gate asks the configured model to repair that captured output.
It also checks a captured emotional metaphor and an invented reviewer quotation.
These focused checks do not replace fresh full-canon conversations.

```powershell
uv run --locked python -m unittest tests.unit.workflows.main_cognition.test_output_recovery -b
uv run --locked python -m unittest tests.live.test_cognitive_recovery -v
```

Fresh results, pending review items, and retained failures belong in the [A18B verification record](../work_items/verification/a18b-cognitive-evaluation.md).
