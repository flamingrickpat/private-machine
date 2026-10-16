# Review materiality

## Summary

Reviewers classify a discrepancy by its practical consequence for the actual request.
A minor discrepancy can receive `good_enough` and pass with its reason intact.
Material errors still require correction.
Prompt examples guide the model.
Python does not define an error percentage or a universal distance tolerance.

Parent: [workflow review modes](workflow-reviews.md).
Related: [public dialogue](public-dialogue.md), [blinded review](cognitive-review-boundaries.md), and [Minecraft evidence](minecraft-task-evidence.md).
Evidence: [materiality verification](../work_items/verification/a18b-review-materiality.md).

## Classifications

| Path | Passing judgments | Other judgments |
| --- | --- | --- |
| Public speech | `accept`, `good_enough`, `equivalent_or_better` | `block` |
| Flexible Minecraft review | `accepted`, `good_enough`, `equivalent_or_better` | Existing revision, waiting, and blocked states |
| Strict Minecraft review | `accepted` | Other labels do not release review |
| A18B criterion review | `pass`, `good_enough` | `fail`, `uncertain` |

`good_enough` describes an evidenced result with an incidental discrepancy.
Its reason explains the discrepancy and its practical consequence.
`equivalent_or_better` describes a supported alternative that meets the request without violating an explicit requirement.
The original judgment remains visible beside the result.

[`SpeechReview`][workflows.public_speech.speech_review.SpeechReview] restricts public labels to the declared classes.
[`parse_review`][workflows.public_speech.speech_review.parse_review] rejects a passing judgment with `false_task_result: true`.
It also rejects `block` with that flag false.
The existing format retry repeats an invalid review.
It does not silently convert the contradiction into approval.

## Practical examples

| Discrepancy | Context that can permit `good_enough` | Context that makes it material |
| --- | --- | --- |
| Slightly rounded coordinates | Casual description of the same location | Exact placement, a required boundary, or safe landing |
| Omitted sapling stack | Incidental loot in a wood-gathering summary | Requested exact inventory or required saplings |
| Small count difference | Requested practical quantity remains satisfied | An explicit required quantity or decision depends on that count |
| Different implementation | A permitted alternative meets the original goal | The alternative violates an explicit constraint |

A location 10,000 blocks away describes a different place.
Diamonds absent from the evidence do not become a minor discrepancy because their claimed count is small.
False completion, destructive effects, and a wrong recipient can change the user's decisions.
The model must explain the concrete consequence rather than count digits or apply an error percentage.

## Prompt coverage

Story and Assistant generation receive examples of ordinary approximation and material misreporting.
The speech reviewer receives the same distinction beside its classification instructions.
Action selection avoids repeating completed work for cosmetic precision.
Minecraft planning, plan review, verification, and both controller paths receive equivalent guidance.
The native controller keeps this guidance outside compiled survival resources, so skill compilation cannot erase it.
Both A18B reviewer audiences classify minor discrepancies explicitly within their existing evidence boundaries.

Historical comparisons retain their object and time scope.
An earlier harvest can remain true after death empties the inventory.
A tree base and the player's later position describe different objects.
A conditional explanation of pickup mechanics does not claim a completed pickup.
Supplied tool receipt times and source scope remain in [`task_lines`][workflows.public_speech.public_context.task_lines].

## Evidence and acceptance

An approving model judgment does not alter the original observation, required check, or acceptance certificate.
Minecraft's independent conditions and publication checks still apply.
The planner can explicitly repair an unaccepted specification that adds precision or incidental requirements beyond the user's request.
The verifier cannot pretend that a failed check passed.
Previously accepted specifications and their original proof remain exact.

[`validate_review`][dev.work_items.a18b.policy.validate_review] requires citations for `good_enough`, just as it does for `pass`.
Both labels need new companion speech when a new-case boundary exists.
[`score_case`][dev.work_items.a18b.policy.score_case] gives credit when every required repeat supplies a passing label.
If either repeat returns `fail`, the criterion fails.
Missing or uncertain evidence receives no credit.
Mixed `pass` and `good_enough` repeats retain `good_enough` in the aggregate.
Calibration also retains the observed label when it satisfies a control whose expected result is `pass`.

Existing failed receipts remain unchanged.
New prompts do not reroll old valid judgments or rewrite an evaluated conversation.
Historical receipt retrieval and repeated draft repair remain separate follow-up work.
This change does not remove the current runtime error path for a blocked draft.
