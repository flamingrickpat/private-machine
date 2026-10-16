# The missing reply and the three-hour deadline

## Summary

`hurt_and_partial_repair` remained active while its final user exchange stayed pending.
The runtime completed 1,079 model calls after that message, but it published no further reply.
The failure was conversational progress, rather than one three-hour HTTP request.
The harness eventually stopped the case at its global deadline.

Parent: [A18B error analysis](../a18b-errors-20261016.md).
Related: [private loop](../../../subsystems/private-loop.md),
[output recovery](../../../subsystems/cognitive-output-recovery.md).

## Observed timeline

All times below use UTC on 16 October 2026.

| Time | Observed event |
| --- | --- |
| 02:11:02 | The case started. |
| 02:38:08 | Spamuel promised silence at the end of scene one. |
| 02:46:14 | Spamuel published its seventh and final reply. |
| 02:47:01 | Jesse submitted record `11419`, correlation `c10d0b06ec6a46409b60c8c5d2d67e59`. |
| After 02:47:01 | Attention frame `11447` explicitly selected records `11419` and `11420`. The user message reached cognition. |
| 05:10:57 | The controller reached the case deadline and requested worker cancellation. |
| 05:11:03 | The case receipt ended at `10800.485` seconds, with seven new companion messages. |

The checkpoint retains `exchange_status: pending` for record `11419`.
The interval from its delivery to the final cancellation receipt was `8636.270` seconds.
That interval is approximately 2 hours 24 minutes.

## What the runtime did instead

After the final user message, successful model calls were:

| Request owner | Completed calls |
| --- | ---: |
| `main_cog` | 549 |
| `inner_thought` | 290 |
| `inference_0` | 132 |
| `inference_1` | 108 |
| Total | 1,079 |

All 44 later action-selection results were quiet choices.
Thirty-three returned no actions. Eight selected `Idle`, and three selected `Ignore`.
For example, call `b5621e0edfbf4dc798f8c784ddd8199c` called silence the only honest move.
Call `45f2c89d01f843fea7bc49a9cec64d17` selected `Idle` to maintain silence without a reply.
The longest recorded chat attempt in the case lasted `47.797` seconds.
Several temporary connection errors and HTTP 504 responses occurred, but requests recovered.
They do not explain the long period without public speech.

## Code path

| Source at `f7eb226` | Current behavior and consequence |
| --- | --- |
| [dev/work_items/a18b/scenarios.yaml:9](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/scenarios.yaml#L9) and `:11` | The manifest declares a 180-second response allowance and a 10,800-second case deadline. |
| [dev/work_items/a18b/runner.py:333](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/runner.py#L333) and `:529` | `wait_for_companion` calls `wait_case(None, ...)` after normal user delivery. It waits for actual new speech. |
| [dev/work_items/a18b/runner.py:144](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/runner.py#L144) | `seconds=None` creates no local deadline. Only observed speech, application error, STOP, or the case deadline ends the wait. |
| [dev/work_items/a18b/runner.py:535](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/runner.py#L535) | The 180-second allowance occurs after a scene. Execution never reaches it while a message-level wait remains blocked. |
| [workflows/main_cognition/schemas.py:299](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/main_cognition/schemas.py#L299) | Action selection permits an empty action list. |
| [workflows/main_cognition/pipeline.py:167](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/main_cognition/pipeline.py#L167) | `Ignore`, `Idle`, and `Sleep` produce no workflow dispatch. These choices do not suspend the private loop's timer. |
| [workflows/inner_thought/triggers.py:46](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/inner_thought/triggers.py#L46), `:77`, `:85`, and `:94` | Old goals, fresh main frames, idle time, and repeated sources can trigger further private work. These paths have no repeated-progress cutoff. |
| [workflows/inner_thought/host.py:322](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/inner_thought/host.py#L322) | An accepted private thought publishes `thought.created`. Main cognition can react and publish another frame. |

Late active goals included `11538`, which requested an end to commentary, and `11975`, which requested silence until new external input.
Despite those goals, the private timer continued to supply model calls.
The model's repeated explanations about an impossible silence loop are its own descriptions, not independent implementation evidence.
The implementation evidence is repeated accepted cycles, quiet action choices, and the unresolved exchange.

The runtime permits a conversational pause.
It lacks a separate outcome or progress policy for a pause that outlives a pending user exchange.
The harness also lacks a shorter diagnostic boundary for that condition.

## Why the receipt says operator stop

[dev/work_items/a18b/supervision.py:91](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/supervision.py#L91) detects the elapsed case deadline and creates the case's `STOP` file.
[dev/work_items/a18b/runner.py:128](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/runner.py#L128) checks that file before its own elapsed-time check.
It raises `InterruptedError` with the text `The operator requested a stop`.
Thus, the receipt text attributes an automatic deadline to an operator.
The separate `generation-worker.json` retains the accurate reason: `case deadline exceeded`.
The controller allowed cleanup and did not force-kill this worker.

## Repair boundary

A reply deadline must produce an explicit incomplete diagnostic rather than send another unanswered message.
A product progress policy also needs to distinguish deliberate quiet, active external work, and repeated private cycles without progress.
It must preserve the pending user source and permit deliberate silence.
An automatic canned reply conceals the unresolved behavior.

Evidence lives under `.runtime/a18b-current/fresh/hurt_and_partial_repair/`:
`chat.jsonl`, `ghost.json`, `generation-worker.json`, `receipt.json`, and `llm-logs/usage-5428.jsonl`.
The diagnostic index retains the 44 exact action-selection results.

Code:
[`dev.work_items.a18b.runner.wait_case`][],
[`dev.work_items.a18b.supervision.run_worker`][],
[`workflows.main_cognition.pipeline.action_dispatches`][],
[`workflows.inner_thought.triggers.evaluate_triggers`][].
