# Evidence and quality

## Summary

Foundation tests establish routing, state, and runtime contracts.
Live tests establish endpoint behavior and saved execution evidence.
Conversation evaluation measures a separate quality requirement.
Successful dummy execution does not establish real dialogue quality.

Parent: [manual](../index.md).
Procedure: [development loop](../workflows/development.md).

## Evidence levels

| Evidence | Establishes | Does not establish |
| --- | --- | --- |
| Foundation tests | Isolation, commits, lifecycle, typed contracts | Useful model decisions |
| Dummy console exchange | Process control, trace, chat routing | Character quality or real prompts |
| Real application smoke | Real model path and a correlated reply | Full conversation acceptance |
| Full conversation gate | Saved trajectories and evaluator results | Quality after every later change |
| Minecraft gate | Real observation, action, and recovery | Sustained survival behavior |

The full conversation gate requires 200 completed exchanges across ten scenarios.
It also requires no infrastructure or contract errors and at least seven good trajectories.
The evaluator runs after generation and receives the execution evidence.
Read `tests/live/conversation.py` and `tests/live/test_conversations.py` for the current gate.

## Console evidence

Elapsed times include model waiting, prompt work, and kernel overhead.
Parent durations include child durations.
Adding all rows can double-count time.
Token usage appears in model results only when the backend supplies it.

The console does not infer usefulness from response length or speed.
Human review and saved acceptance results provide that judgment.
The latest 10,000 events form the browser view.
The persistent event file can contain older exchanges outside that view.

## Go deeper

Read [next experiments](../decisions/experiments.md), [one exchange](../workflows/exchange.md), and [plan status](../decisions/plan.md).
The [console verification record](../decisions/console-verification.md) lists the checks for this delivery.
