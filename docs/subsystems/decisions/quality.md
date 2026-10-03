# A02 model observations and limits

## Summary

Real typed inference can return incorrect interpretations.
A02 records these observations without adopting Laya in a workflow.
Long-input acceptance does not establish reliable classification.
The caller must evaluate each possible use against its LLM alternative.

Parent: [optional typed decisions](index.md).
Receipts and provenance: [A02 verification](verification.md).
Repeat: [operation](operation.md).

All evidence paths are relative to `.runtime/alpha/a02/`.

## Typed inference and limits

Choice, score, probability, and multi-question calls returned valid typed answers.
The short captured completed-run input selected `COMPLETE` with confidence 0.881.
The empty `answer` field received an expected completeness score of 1.788 on levels zero through two.
That score incorrectly favors the complete-answer level.

The plain report received a nonempty-answer probability of 0.387.
The structured multi-question state received 0.648 for the same question text.
The different state layouts prevent a claim that question count caused this difference.
These observations do not establish a reliable probability classifier.
The initial protocol probe also assigned 0.997 probability to a public reply despite the explicit missing-reply description.

Eight history layouts tested report placement at the start and end.
The history budgets were 256, 1024, 4096, and 7800 observed tokenizer tokens.
All eight selected `COMPLETE`, but long-input confidence depended strongly on placement.
The 7800-token prefix produced a 7,951-token end-placement request with confidence 0.154.
The equivalent start-placement confidence was 0.777.
These examples establish observations, not accepted classifier accuracy.

Two longer layouts with an 8300-token history prefix returned HTTP 400.
The configured architectural context, batch, and microbatch each equal 8192.
The decision head budget equals 192 tokens and can silently truncate long questions and options.
The [publisher model card](https://huggingface.co/convaiinnovations/laya) identifies the base English checkpoint's trained context as 512 tokens.
Architecture limits and observed request acceptance do not establish reliable behavior at those larger lengths.

## Chat contention

The final repeat used three warm samples per condition after other live package tests finished.
Chat alone took 0.328, 0.328, and 0.359 seconds.
Chat during Laya load took 0.313, 0.359, and 0.375 seconds.
Their medians were 0.328 and 0.359 seconds.
The load comprised eight sequential 4,247-token decisions per sample.
Those 24 decisions took 0.312 to 0.407 seconds each.

`live-final/contention.json` stores every chat response and decision receipt.
This short shared-workstation experiment cannot establish a stable latency target or a speed advantage over an equivalent LLM classifier.
The first measurement overlapped pm-coder live tests and remains explicitly contaminated.
Its evidence remains in `live-verified/contention.json`.
It supplies no accepted contention comparison.
