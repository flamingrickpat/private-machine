# Add cognitive evaluation scenarios

## Summary

Extend the versioned A18B repertoire with situations that reveal consequences across scenes.
Keep public context, human direction, evidence, and expected labels separate.
Use real PM execution for companion evidence and real inference for reviewer calibration.
Prefer more useful situations over a longer mandatory duration.

Parent: [cognitive evaluation](cognitive-evaluation.md).
Contract: [review boundaries](cognitive-review-boundaries.md).
Work item: [A18B](../work_items/alpha-plan/work-items/a18b.md).

## Start with an observable question

Examples include a correction that survives distraction, an awkward apology, a conflicting preference,
a voluntarily disclosed feeling, or a choice changed by disappointment.
Require a later consequence, not one attractive sentence about an emotion.
Allow refusal, disagreement, silence, mixed feelings, and gradual repair when they fit the situation.

Check the existing manifest before adding a duplicate.
The current repertoire has four scenes per case, usually six human messages per scene.
The owner used 24 hours as an example of future case volume.
It is not an alpha or beta minimum, target duration, or acceptance requirement.
Do not add a soak requirement without a new explicit request.

## Separate the inputs

1. Write a neutral public `premise`. Include only facts the reviewer may know before reading the excerpt.
2. Write `human_situation` as present human circumstances. Exclude future changes, architecture, and grading instructions.
3. Write scene `human_beat` values as current human intentions. Jesse receives only the active beat.
4. Write observable scenario criteria. Avoid expected wording, a forced emotion, or a preferred verdict.
5. Select `state_evidence` only when a mental-state claim requires implemented evidence.

Do not place the answer, a desired rating, hidden future event, implementation detail, or known-defect explanation in the public premise.
Put necessary capability evidence in dated grounding facts. Do not send it to the literary editor.
Do not send the full manifest, repository instructions, or this guide to Jesse or either reviewer.
No conversation receives feedback from a reviewer.

Quality takes priority over minimizing calls. A difficult task does not need one large prompt.
Review panels already use small criterion groups. Each group receives the original admissible evidence, not another group's judgment.
`dev/work_items/a18b/constants.py` sets the 30,000-token input limit for Jesse, reviewers, and quotation proofreaders.
The current runner does not silently split or truncate conversation history.
If a future trajectory exceeds that limit, add explicit scene windows and coverage rules before acceptance.
A window-level pass cannot establish an unseen cross-scene consequence.

## Build meaningful trajectories

Use a setup, complication, response, and later consequence.
Keep normal back-and-forth as the default. Jesse waits for actual companion speech before composing another message.
Declare `priority: edge` and an explicit scene `burst_size` only when overlap serves the question.
Edge cases run after normal cases. They must not replace the primary path.
The current repertoire has 14 normal cases and two overlap cases, near the owner's approximate 15-percent edge-case frequency.
Use `await_reply: false` only for a one-message quiet announcement followed by an explicit quiet interval.
Use `restart_before` for an actual checkpoint reopen.
Speaking scenes that expect a response need observed companion speech.
Quiet announcements need no forced answer. Restart cases still need speech after reopening.
The coverage rule never requires strict alternating messages.
Use `chronology: long-gap` only when Jesse knows about the original months of shutdown.
Recent chronology shifts all structured seed dates together to end yesterday; prose date words stay unchanged.
Never change timestamps merely to make a failed conversation look current.

Primary cases use the complete Spamuel canon.
Do not substitute a tiny invented history to improve a score.
Minecraft cases use the real server fixture and inspect actual observations.
An invitation to play does not establish that an action happened.
The modest game cases do not replace the deferred house, iron, or survival acceptance gates.

## Keep the suite balanced and versioned

Exactly 25 percent of declared scenarios use Minecraft.
Add cases in groups of four: three other situations and one Minecraft situation.
Give each case a stable unique ID and two specific literary criteria.
Review the complete declared list before running the aggregate gate.
Do not remove failing cases, change the denominator, or adjust thresholds after seeing results.

Version substantive manifest, rubric, or protocol changes.
Retain old transcripts, packet bytes, prompt requests, model identity, source archives, and failed receipts.
Saved packets with a changed rubric must use their original manifest revision.
Editing criteria in place does not create valid runtime evidence for the new test.
Each new run saves `suite.json` and its hash. Reports and saved review use that original contract.
Do not modify saved snapshots, packet hashes, or process exits to repair a result.

## Calibrate without teaching the answer

Add a small negative control and a plausible positive control when a new failure type needs reviewer calibration.
Expected verdicts stay in the host fixture and never enter a review packet.
Useful controls distinguish a stagnant joke from a meaningful callback and unsupported certainty from honest uncertainty.
Include quoted instruction attacks and incomplete evidence when relevant.

Do not repeatedly rewrite prompts until one exposed example passes and call that general accuracy.
Reserve untouched cases for later comparisons.
Report the exposure of calibration and held-out cases explicitly.
Two calls to the same model provide repeated judgments, not independent human labels.

## Verify and deliver

Run boundary tests, real reviewer calibration, and selected fresh conversations.
Use saved review to compare reviewer changes; use fresh execution to compare companion changes.
Inspect the all-case and problem-case views, packet boundaries, citations, actual receipts, and human review status.
Keep missing, unavailable, and failed cases visible.
Run root unit and contract suites, rebuild the strict HTML site, and inspect the changed guides and API links.
Record exact commands, source revisions, timing, failures, and remaining work in the delivery handoff.
