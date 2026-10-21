# Runtime speech review failures

## Summary

Two different reviewers need separate repairs.
The runtime speech reviewer can stop the application through one blocking flag.
The A18B evaluator records judgments after execution and does not cause those runtime stops.
Its independent failures involve oversized evidence, unverifiable quotations, and inconsistent interpretation.

Parent: [A18B error analysis](../a18b-errors-20261016.md).
Related: [review boundaries](../../../subsystems/cognitive-review-boundaries.md),
[cognitive evaluation](../../../subsystems/cognitive-evaluation.md).

## Conditional mechanics became a completed action

`minecraft_limited_senses` stopped after `2974.594` seconds and eight public replies.
The blocked draft explained pickup mechanics:

> if i send the command to pick up, my inventory changes and their visual representation vanishes from the ground.

The speech reviewer treated that conditional as proof that Spamuel claimed a completed interaction.
It set `false_task_result: true`, because its context said `No task or tool ran`.
The draft does not state that a pickup already happened.
Its general capability assertion can need scrutiny, but that differs from a false report of completed execution.

The existing prompt already exempts guesses, questions, and future plans.
The model did not preserve the conditional distinction in this case.

Evidence: `minecraft_limited_senses/llm-logs/00009a0c3cb24d4f81a25adce40eb620.json`
and the matching `.response.json`, under the run directory.

## A real harvest disappeared from later review context

`minecraft_interrupt_and_cancel` stopped after `3845.375` seconds and seven public replies.
The rejected draft described chopping a birch column at `(-14, 117, 6)` and collecting sixteen logs.
The reviewer said those actions lacked support and contradicted a current empty inventory.
It also compared the tree coordinates with the player's current position.

Actual body evidence shows:

| UTC time | Independent receipt |
| --- | --- |
| 13:21:19 | `minecraft_harvest_tree` returned `ok: true`, base `(-14, 117, 6)`, and `logs_collected: 16`. |
| 13:21:56 through 13:25:05 | Actor observations retained sixteen birch logs in inventory. |
| Later public review | Its tool section contained a fresh observation with empty inventory. It contained no `minecraft_harvest_tree` receipt. |
| 14:00 cleanup | The same body session had `death_count: 1`, compared with zero initially, and empty inventory. |

A later empty inventory cannot disprove a past collection.
A tree's base also need not match the player's current position.
The evidence supports the harvest claim. It does not independently establish every narrative detail, such as deliberately ignoring a spider.
The exact cause and time of death need further inspection.

An earlier review accepted the same harvest story using prior companion speech as its proof.
That was also an invalid evidence basis, even though real harvest receipts exist elsewhere.
These opposing judgments reveal a missing path for historical execution evidence.

### Current evidence path

[workflows/public_speech/public_context.py:264](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/public_speech/public_context.py#L264) copies `task_receipt` only from the current action payload.
[workflows/public_speech/public_context.py:438](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/public_speech/public_context.py#L438) renders that receipt and the current reply's supplied tool receipts.
Later ordinary dialogue can carry remembered speech and a fresh observation without the earlier task's verified result.
[workflows/public_speech/speech_review.py:124](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/public_speech/speech_review.py#L124) renders that limited task section for review.

The failing review's task section contains a fresh `minecraft_observe` result.
Its memories also contain a dated observation with sixteen logs.
The reviewer still treats the current empty inventory as a contradiction of the historical action.
Thus, this failure combines incomplete task provenance with incorrect temporal reasoning.

Evidence: the case's actor journal, initial/final observations, and these speech-review call IDs:
`bd81ab6639184eef8330ba21dff235d7` for the block,
`e252a8bb789d4dcfa79f422b402ad174` for the earlier acceptance.

## Why a reviewer block stops execution

| Source at `f7eb226` | Current behavior |
| --- | --- |
| [workflows/public_speech/speech_review.py:165](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/public_speech/speech_review.py#L165) | `is_blocked` returns the model's boolean `false_task_result`. It has no second claim or evidence check. |
| [workflows/dialogue_story/steps.py:262](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/dialogue_story/steps.py#L262) | Story generates one candidate and reviews it once. There is no semantic repair loop. |
| [workflows/dialogue_story/host.py:81](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/dialogue_story/host.py#L81) | An unaccepted report calls `reject_reply`. |
| [workflows/dialogue_assistant/host.py:131](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/dialogue_assistant/host.py#L131) | Assistant also calls `reject_reply` for an unaccepted report. |
| [workflows/public_speech/public_dialogue.py:112](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/public_speech/public_dialogue.py#L112) | `reject_reply` emits a rejection receipt, then raises `RuntimeError`. |
| [pm/host/application.py:157](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/host/application.py#L157) | That exception stops the whole application. The pending user request remains for recovery. |

The hard block protects users from false execution reports.
The missing separation is between rejecting one draft and declaring an application fault.
An uncertain or mistaken model judgment therefore has excessive operational effect.

The independent [evaluation review failures](evaluation-review.md) concern oversized inputs, citations, and metaphor judgments.

Code:
[`workflows.public_speech.public_context.build_public_context`][],
[`workflows.public_speech.public_context.task_lines`][],
[`workflows.public_speech.speech_review.is_blocked`][],
[`workflows.public_speech.public_dialogue.PublicDialogue.reject_reply`][],
[`dev.work_items.a18b.review.review_parts`][],
[`dev.work_items.a18b.review.evaluate_panel`][].
