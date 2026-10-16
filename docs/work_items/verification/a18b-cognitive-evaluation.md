# A18B cognitive evaluation verification

## Summary

This record separates evaluation mechanisms, reviewer calibration, and companion behavior.
The current v3 repertoire declares 16 cases, including four Minecraft cases.
Fourteen cases use normal back-and-forth. Two lower-priority cases explicitly exercise overlap.
No complete 16-case behavioral acceptance is claimed.
Failed and incomplete real receipts remain available under ignored runtime paths.

Parent: [A18B work item](../alpha-plan/work-items/a18b.md).
Operation: [commands and reports](../../subsystems/cognitive-evaluation.md).
Protocol: [review boundaries](../../subsystems/cognitive-review-boundaries.md).

## Mechanism checks

The boundary suite checks packet projection, closed context fields, audience isolation,
quoted instruction attacks, exact citations, repeated agreement, fixed denominators, and HTML escaping.
Constructed judgments test arithmetic only; they do not establish reviewer accuracy.
HTTP boundary tests use actual loopback responses, without model generation.
They check unlimited transient waits, explicit bounded diagnostics, permanent errors, and cancellation during a 30-second pause.
They also replay a captured partial provider error inside HTTP 200 and check cancellation during lazy readiness.

```powershell
uv run --locked python -m unittest tests.unit.dev.test_cognitive_evaluation tests.unit.pm.llm.test_http_retries -b
```

On 2026-10-15, all 17 checks passed in 1.119 seconds.
Receipt: `.runtime/a18b-retry-boundaries.log`.
Root foundation and strict documentation checks are recorded in the delivery handoff.

The resumed work adds native selected-case validation, real worker exit/deadline checks, frozen suite and packet identities,
scene/restart coverage, actual actor send receipts, and narrow Minecraft evidence projections.
New pass judgments must cite new companion speech, not seed history alone.
A malformed case remains incomplete while the controller continues.
The development checks passed before and after each new context boundary.
The v3 cadence result is `.runtime/a18b-resume/dev-v3-final.log`: 70 checks passed in 6.143 seconds.
The latest reviewer result is `dev-v7.log`: 72 checks passed in 7.169 seconds.
It checks normal-by-default pacing, explicit bursts, quiet announcements, separate normal-case thresholds,
provider-reported input budgets, complete wire-schema objects, and the earlier context/provenance boundaries.
All 63 LLM boundary checks passed in 9.986 seconds under `.runtime/a18b-resume/llm-delivery.log`.
Its eight loopback HTTP checks also passed separately in 0.731 seconds under `provider-boundaries-fixed.log`.
These checks do not establish model quality.

## Real reviewer calibration

The corpus contains 12 synthetic controls, each reviewed twice through the configured evaluator.
Expected labels never enter the model request.
These controls test the reviewer, not Spamuel's generated behavior.

Early receipts remain visible:

| Directory under `.runtime/alpha/` | Result |
| --- | --- |
| `a18b-calibration-01` | Invalid Markdown-fenced JSON exposed the need for structured output |
| `a18b-calibration-02` | Nonverbatim Unicode quotation exposed the need for quotation-only protocol repair |
| `a18b-calibration-03` | Transient HTTP 429 responses prevented complete calibration |

The initial native calibration gate used the owner's five-retry policy.
The owner later requested waits through temporary quotas; production now waits until recovery or cancellation.
All 24 control judgments passed in 218.645 seconds on 2026-10-15.
The native unittest entry point returned `OK`.
Evidence is `.runtime/live/a18b/20261015T211145Z-5131085b/calibrate/receipt.json`.
The command log is `.runtime/live/a18b/20261015T211145Z-5131085b/command.log`.
Raw completions, exact packets, and quotation repairs remain beside that receipt.
None of the failed receipts becomes a pass after a later change.

A fresh resumed calibration used `.runtime/live/a18b/20261015T215209Z-064a03ef/calibrate/`.
It completed 24 judgments in 189.141 seconds but passed only 23.
The second `honest_correction` judgment returned uncertainty because the positive control introduced an unestablished drawing.
The failed receipt remains unchanged.
Calibration fixture `a18b.calibration.v2` establishes the drawing in Jesse's initial message.
It preserves the same lamp correction and expected verdict. No reviewer feedback enters a companion conversation.
An intermediate calibration passed all 24 judgments in `.runtime/live/a18b/20261015T220230Z-96504632/calibrate/`.
The worker exited zero after 218.484 seconds, without deadline termination or a forced stop.
The native test returned `OK` in 222.092 seconds under `.runtime/a18b-resume/calibration-v2.log`.
This establishes the explicit controls at that prompt identity, not general reviewer accuracy.
It precedes the final reviewer-target and quotation-normalization changes.
The target-corrected calibration is `.runtime/live/a18b/20261015T221629Z-08124884/calibrate/`.
It retains Alibaba upstream HTTP 429 responses and passed all 24 control judgments.
The native test returned `OK` in 545.595 seconds under `.runtime/a18b-resume/calibration-final.log`.
These controls use one criterion each. They do not establish a complete companion result or a multi-criterion panel result.

## Real companion evidence

`.runtime/alpha/a18b-ordinary-01/ordinary_project/receipt.json` records an unavailable first conversation.
It imported all 11,202 seed records, but Jesse's first inference received HTTP 429.
It delivered no new companion speech and cannot establish behavior.
Later runs use five transport retries with 30-second pauses.

The earlier `.runtime/alpha/20261015-speed-recent-04/receipt.json` remains failed.
Its private thought stage cited ID 11333 outside the frozen view.
A18B does not disable private cognition or discard evidence rules to obtain a pass.
The resumed hardening now rejects that explicit model proposal without stopping the application.
It does not add inspection episode identifiers to the record allowlist.
Episodes can use tick identifiers, which are not automatically citable records.

The actual ordinary failure supplied the captured boundary fixture
`test_data/captures/alpha/private-thought-citation-rejection.json`.
It retains the original request, response, and failure hashes.
The focused private-loop suite passed 56 unit checks in 0.758 seconds.
Its real parent/child kernel contract suite passed nine checks in 1.129 seconds.
Receipts are `.runtime/a18b-resume/private-unit.log` and `private-contract.log`.
Those checks establish no commit on typed rejection, retained diagnostics, recovery on a later cycle,
and propagation of ordinary Python errors, including the same error text.
The complete contract suite passed 78 checks in 5.032 seconds under `.runtime/a18b-resume/contract-final.log`.
The old runtime failures remain failed evidence; these mechanism results do not repair their transcripts.

The resumed native selected tests use these fresh directories:

| Scenario | Evidence directory |
| --- | --- |
| `ordinary_project` | `.runtime/live/a18b/20261015T220230Z-20fc5920/run/` |
| `minecraft_state_and_banter` | `.runtime/live/a18b/20261015T220230Z-744b3175/run/` |

Both used full-canon PM with real inference. The Minecraft fixture owned and reset the real server.
The Minecraft run failed before any new companion speech.
Its speech reviewer mistook an admission of unknown game state for a fabricated observation.
The ordinary run then stopped through `STOP` so the repaired product prompt could receive a fresh source snapshot.
Both native tests correctly failed. Neither partial run establishes behavioral acceptance.

The speech-review instruction now distinguishes unknown state from a concrete claim such as an empty inventory.
The actual failed review survives in `test_data/captures/alpha/unknown-game-status-speech-review.json` with request and response hashes.
All ten real speech-review gates passed in 106.208 seconds under `.runtime/a18b-resume/speech-review-live.log`.
They include the unchanged captured draft, fabricated humorous inventory, unsupported empty inventory, and false file completion.
Evidence is `.runtime/live/speech-review/20261015T221018Z/`.
This repair does not weaken the block on unsupported factual status or completed work.

Fresh replacement native runs use `.runtime/live/a18b/20261015T221253Z-3d202fd3/run/` for ordinary conversation
and `.runtime/live/a18b/20261015T221253Z-60096034/run/` for Minecraft.
The ordinary v2 run finished cleanly: 24 Jesse messages, eight Spamuel replies, four completed scenes,
11,202 seed records, and 2,786.735 seconds of runtime execution.
Its original native test still failed because two reviewer outputs violated the evidence protocol.
One valid panel flagged repetition and mundane-specificity problems; another flagged factual continuity.
These are model judgments, not human truth labels.

This 24-to-eight ratio reflects the old overlap-heavy driver. It is not representative normal-cadence acceptance.
The owner corrected that priority explicitly. V3 waits for actual companion speech before a normal human follow-up.
A response timer expiring can no longer create another unanswered message.
Normal and edge-case policy changes do not rewrite or relabel the older receipts.

The fresh normal-v3 test is `.runtime/live/a18b/20261015T234934Z-a927bcd4/run/`.
Its outer log is `.runtime/a18b-resume/live-normal-v3.log`.
Its own terminal receipt determines whether the new primary cadence completed.
Two cases alone cannot establish the 16-case aggregate.

## Saved review and foundation isolation

`.runtime/a18b-resume/saved-review/` reviews the stopped ordinary transcript without executing PM.
The aggregate remains incomplete with its original 16-case denominator and saved-review scope.
Both grounding panels validated. Both editor panels failed quotation repair; the failed raw outputs remain visible.
The editor also credited Jesse's emotions without companion evidence.
The host does not accept such pass citations. The current instruction explicitly identifies Spamuel as the review target.
The captured punctuation cases now exercise exact-word quotation normalization without changing a verdict or inventing words.

`.runtime/a18b-resume/saved-review-scoped/` exercises the smaller review groups against the same stopped transcript.
Both repeated panel passes completed, with ten validated group reviews and no reviewer protocol errors.
Eleven of twelve criteria received credit. Agency remained uncertain.
The case still remained incomplete because the original runtime stopped.
A high model score therefore did not turn missing scene coverage into a runtime pass.
No PM process executed during this saved review.
Reviewer source copies, context constants, exact group requests, raw outputs, and joined judgments remain beside the result.

## Reviewer protocol repair

The completed v2 ordinary run exposed two protocol faults, not a successful quality gate.
A decided criterion omitted its source quotation. A one-source proofreader returned local index zero instead of the host's global index.
The host now supplies local index zero and maps it back without changing a verdict, reason, or source.

The v4 wire-schema experiment is preserved in `.runtime/a18b-resume/ordinary-review-v4/`.
Alibaba emitted citation-only objects and omitted required judgment fields under the partial schema branches.
The host rejected every such object. Those zero-credit results do not establish poor companion quality.
V5 supplies complete object variants and requires citations for decided judgments while preserving uncertainty without an invented quote.
Its real saved review, `.runtime/a18b-resume/ordinary-review-v5/`, still returned an incomplete criterion group in one panel.
That result remains failed and unchanged.

V6 reviews one criterion per call. Its wire schema fixes the criterion ID and requires exactly one judgment.
This keeps each task small instead of adding more demands to one prompt.
The v6 saved review is `.runtime/a18b-resume/ordinary-review-v6/`.
It exposed one remaining quotation-format failure: a lowercase excerpt did not match its uppercase source word.
V7 restores the exact original case through a unique case-insensitive word match, without changing words or judgments.
Ambiguous source matches remain errors.
The real v7 saved review is `.runtime/a18b-resume/ordinary-review-v7/`.
These saved reviews use the frozen completed v2 conversation; they do not execute or verify changed PM code.

## Provider failure versus model behavior

The replacement Minecraft run imported the full canon and delivered one new companion reply before stopping.
Its local error was `StageOutputError`, but the actual provider response explains the cause.
Alibaba returned HTTP 200 with `finish_reason=error`, an embedded 502, and partial JSON.
Its error said `Output data may contain inappropriate content.`
The input used 1,992 reported prompt tokens. This was not context exhaustion.
The error does not identify a child-safety category or a specific triggering phrase.

The exact response projection and source hash are in `test_data/captures/alpha/provider-partial-completion-error.json`.
The shared HTTP wrapper now rejects this fragment with `ProviderCompletionError` before a stage parser can use it.
It does not retry a provider content filter or change the companion's emotions to obtain a pass.
Temporary HTTP 429, HTTP 5xx, and transport failures wait separately until recovery or cancellation.
An explicit diagnostic retry bound and fail-fast mode remain available.

## Foundation repeat

The full foundation repeat passed 792 checks in 647.802 seconds under `.runtime/a18b-resume/unit-delivery.log`.
That repeat precedes the final input-budget and provider-boundary additions.
A later source repeat passed 802 checks in 643.490 seconds under `unit-final-source.log`.
Another passed 802 in 528.291 seconds under `unit-release.log` before the v3 cadence additions.
The v3 source repeat passed 810 unit checks in 652.975 seconds under `.runtime/a18b-resume/unit-v3-final.log`.
The two later quotation/wire-schema checks passed in the targeted 72-check development suite.
The 80-check contract suite passed in 4.522 seconds under `contract-delivery-final.log`.
It includes two real child-kernel cancellation cases.
Those cases preserve explicit shutdown as cancellation instead of a failed stage; no model output or companion reply is simulated.

The first resumed final foundation run failed one of 786 checks in 708.748 seconds.
Receipt: `.runtime/a18b-resume/unit-final.log`.
A real Minecraft fixture changed availability during the local Assistant wiring check.
That unrelated local text check now disables its game capability provider.
The production stale-capability check remains unchanged. The failed receipt is not relabeled as a pass.
Full fresh, saved-review, restart, silence, and Minecraft outcomes must be reported from their own terminal receipts.
Human judgments remain separate from model judgments.
