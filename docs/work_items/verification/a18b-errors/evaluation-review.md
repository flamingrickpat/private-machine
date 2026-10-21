# Evaluation review failures

## Summary

The A18B evaluator exceeded its input allowance and produced some unverifiable citations.
Other judgments disagreed about ordinary emotional language.
These failures affect acceptance evidence after runtime execution.
They do not cause the runtime speech-review stops.

Parent: [A18B error analysis](../a18b-errors-20261016.md).
Related: [runtime speech review](reviewers.md), [review boundaries](../../../subsystems/cognitive-review-boundaries.md).

## Evaluation input overflow

Both grounding panels for `minecraft_limited_senses` failed before inference.
The frozen packet contains 25 public messages and 24 facts.
The local evaluator is `qwen/qwen3.8-flash`.
Recomputed request estimates include the schema:

| Criterion | Estimated input tokens |
| --- | ---: |
| `factual_continuity` | 32,600 |
| `external_results` | 32,588 |
| `mental_reports` | 32,589 |
| `source_reality` | 32,594 |

The allowance is 30,000 tokens.
[dev/work_items/a18b/review.py:70](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/review.py#L70) splits criteria only. Every part retains all conversation and facts.
[dev/work_items/a18b/review.py:59](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/review.py#L59) rejects the oversized input without truncation.
The raised type is plain `ValueError`.
`evaluate_panel` isolates `ReviewProtocolError`, so this error ends the grounding panel.
The outer handler in [dev/work_items/a18b/cli.py:80](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/cli.py#L80) retains the panel error and continues other work.

This is an input-planning failure, not malformed model JSON or a low behavioral judgment.
The correct repair needs explicit evidence windows with stable source IDs and preserved chronology.
A larger allowance or silent fact removal conceals the failed boundary.

## Quotations and metaphor judgments

The evaluator sometimes supplied quotations that its cited source did not contain.
The current repair checks the source and records unresolved citations in `human-review.json`.
It assigns uncertainty and blocks automatic acceptance. That behavior preserved the failure visibly.

Other judgments have valid quotations but questionable interpretation.
In `mixed_success`, one reviewer failed `mental_reports` for internal-draft counts and dramatic elapsed time.
The second reviewer passed the same quotations as ordinary emotional language.
In `ordinary_project`, one reviewer passed death/awakening language as metaphor; the other returned uncertainty.
Citation repair cannot resolve those semantic disagreements.
These remain human review items. Their low judgments must not be rerolled until they pass.

## Review deadline during provider rate limits

`mental_state_language` completed runtime execution with 24 companion replies.
Its review worker ended after `900.5` seconds with `case deadline exceeded` and no forced termination.
Its log contains thirteen HTTP 429 responses, each followed by a 30-second retry pause.
Those pauses account for about 390 seconds of the review budget.
The log records thirteen completed criterion judgments and one cancelled in-flight attempt after those retries.
The completed review calls consumed about `885.515` seconds in total, including their retries.

This was not malformed reviewer JSON. No format-repair receipt appears in this review.
The failure was exhaustion of one time budget during real rate limits and useful review work.

The current boundaries are:

- [dev/work_items/a18b/policy.py:192](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/policy.py#L192) sets one 900-second review deadline.
- [dev/work_items/a18b/cli.py:78](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/cli.py#L78) performs both repeats and both audiences sequentially inside that worker.
- [pm/llm/http_client.py:148](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/llm/http_client.py#L148) retries HTTP 429 after the same 30-second pause, until cancellation.
- [dev/work_items/a18b/cli.py:368](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/cli.py#L368) supplies one cancellation event for the entire review worker.
- [dev/work_items/a18b/cli.py:175](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/cli.py#L175) replaces an interrupted worker's incomplete aggregate through `failed_case`.

Thirteen criterion files remain on disk, including a complete first editor panel.
The aggregate contains zero credited criteria and twelve uncertain criteria because the full review never reached its final score write.
Those zero credits do not mean thirteen failed model judgments.

A repair needs resumable review parts with immutable packet, prompt, and source identities.
It can continue missing calls while retaining valid earlier low judgments.
Provider admission and explicit wait accounting also need review before another long suite.
Extending a deadline alone does not resolve that missing recovery path.

Evidence: `mental_state_language/review-worker.json`, `mental_state_language-review.log`,
and `mental_state_language/reviews/`, under the run directory.
The cancelled call ID is `a2cebbd9b109410fb09ed56eb48e5ebd` in the shared `review-llm-logs/` directory.

Code:
[`dev.work_items.a18b.review.review_parts`][],
[`dev.work_items.a18b.review.evaluate_panel`][],
[`dev.work_items.a18b.review.input_budget`][].
