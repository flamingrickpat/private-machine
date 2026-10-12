# Plan the next experiments

## Summary

Use recorded stage inputs, reports, prompts, and timings to choose an experiment.
Compare behavior before and after the change.
Keep the acceptance boundary intact.
Treat usefulness as an explicit review criterion.

Parent: [plan status](plan.md).
Evidence: [quality guide](../concepts/evidence.md).

## Compare a shorter classification chain

1. Save several real exchanges with different action requirements.
2. Identify the slow stages in **Timing**.
3. Inspect their prompts and reports in the network view.
4. Record which report fields later stages actually consume.
5. Implement the alternate chain inside a separate workflow.
6. Compare action selection, accepted memory, reply quality, errors, and elapsed time.
7. Run the relevant direct contracts and live conversation gate.

A two-second target is an experiment goal.
The current implementation does not guarantee that duration.
Model activation and backend waiting can dominate short exchanges.
Parent workflow durations include child work.

## Record usefulness

| Criterion | Review question |
| --- | --- |
| Action correctness | Did the system select the required action? |
| Memory relevance | Did the accepted frame include useful memories? |
| Claim support | Did receipts support external-task claims? |
| Character behavior | Did Story dialogue preserve its contract? |
| Cost | How many calls and backend tokens did the exchange require? |
| Latency | Which stage dominated elapsed time? |

The console supplies execution evidence.
It does not assign an automatic usefulness score.
Keep your score beside the saved trace and tested Git revision.

## Cognitive architecture extensions

A new retrieval method belongs at the memory-candidate boundary.
A new appraisal method belongs in its stage implementation.
A new managed action belongs in the supervisor's configured targets.
The generic bus and Ghost service do not own cognitive policy.

JEPA or LAYA integration needs a named implementation and an input/output contract.
The current source contains no such integration.
Its placement depends on whether it predicts world state, selects context, or controls an external action.

## Go deeper

Read [development loop](../workflows/development.md), [memory experiments](../workflows/memory.md), and [`workflows.main_cognition.pipeline.run_cycle`][].

