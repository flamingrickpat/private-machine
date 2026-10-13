# Model selection and reliability

## Summary

A02 found poor Laya quality, including confidently wrong answers.
The LLM is the default.
Retain Laya only for simple classifications that pass realistic live-endpoint unit tests.
If making Laya work requires expensive effort, use the LLM and stop that experiment.

Parent: [alpha plan](index.md).
Contracts: [classification inputs](workflow-inputs.md) and [decision register](decision-register.md).
Evidence: [A02 model observations](../../subsystems/decisions/quality.md) and [model operation](../../models.md#context-handling).

## Governing policy

Laya was planned as a frequent classification path.
The earlier 95% Laya target is withdrawn.
No alpha item requires a percentage of Laya use or a successful Laya adoption.
The delivered client, local artifact, deployment instructions, and receipts remain available for later experiments.
Keeping that technology does not require enabling it in PM.

The LLM can be the primary classifier, rather than waiting for Laya to fail first.
A high-confidence wrong answer can bypass an uncertainty fallback.
A distribution, margin, or successful HTTP response cannot establish semantic correctness.

Complex report interpretation, ambiguous file selection, cognitive appraisal, planning, dialogue, and reflection use the LLM.
Exact metadata and deterministic parsing use ordinary code when they establish the required result.
Laya is a candidate only for a short input, explicit classes, and a simple fact-based decision.
Separate tests must establish file selection and content classification when both use Laya.

## Before enabling a Laya use

1. Define the input schema, class descriptions, evidence requirements, and permitted input lengths.
2. Write unit tests that call the actual Laya endpoint and assert independently defined expected results.
3. Use realistic synthetic data with normal application formats, distractors, missing fields, contradictory facts, and invalid cases.
4. Include captured application inputs when available.
5. Exercise decisive evidence at different positions within the intended input budget.
6. Test long questions and options against the decision head budget.
7. Run the same cases through the normal LLM for a useful comparison.
8. Record wrong answers, critical misroutes, INVALID handling, timing, endpoint identity, and exact input digests.
9. Enable Laya only after every required case passes and a useful reason for adoption remains.

Synthetic data supplies realistic test inputs, never simulated inference.
Expected results follow explicit source facts rather than either model's answer.
Straightforward synthetic assertions need no separate human-label approval.
Ambiguous interpretation and persona immersion still need human review.

Untouched cases protect against prompt changes that only fit known examples.
Each candidate needs an appropriate independent test set, rather than mandatory review of the old A01 corpus.
Preserve the A01 holdout and its recorded exposure status.
Neither that pending review nor a failed optional Laya experiment blocks LLM delivery.

If tests reveal recurring errors or require extensive prompt tuning, retraining, or integration work, keep the LLM.
Do not expand alpha scope to improve Laya.
Model, server, class, prompt, or budget changes require the affected live tests again before use.

## LLM reliability is a core requirement

The LLM path must remain usable independently of Laya.
Service absence, direct Python failures, unusable outputs, and lost required context need immediate diagnosis and repair.
An optional model experiment cannot excuse a failure in the normal LLM path.
Every required live acceptance case must pass before the affected mechanism is delivered.

pm-coder's agent loop discovers the serving context and compacts accumulated execution history after context exhaustion.
It retains the original request, creates a checkpoint, and resumes from recorded work.
A03 through A20 must preserve that mechanism for agentic tasks and long-file inspection.

PM's direct `ChatClient` sends one supplied message list and does not run the pm-coder loop.
Its caller must construct suitable task context through scoped retrieval, selection, or explicit evidence-preserving compaction.
An unbounded history dump is not a replacement for those mechanisms.
Long agentic work must use pm-coder rather than a raw completion with accumulating history.

Laya's 192-token decision head and configured 8192-token state context belong to its separate endpoint.
They do not configure the chat service or disable pm-coder compaction.
Chat still has its own finite serving context.
Required evidence must survive context handling, and an incomplete generation cannot become an accepted classification or task result.

## Known companion defects

[C01](baseline/receipts.md#diagnosed-conversation-failure) recorded a completed managed worker but left the console exchange pending without a public reply.
This is an action-to-reply defect, rather than a model connection failure.
The control worker's artifacts do not establish an exact-content note or arbitrary instruction execution.
A09 owns a real agent-backed note acceptance case, actual file verification, and useful public reply delivery.

[C02](baseline/receipts.md#current-verification) restored equal records, GWT, and task receipts.
Sunny still claimed that a pre-restart codeword did not survive, although the only supplied codeword was `copper lantern`.
This is a recall or interpretation defect, rather than evidence of lost checkpoint data.
A08 verifies original evidence retrieval, and A09 verifies grounded public claims.
Keep both defects visible until new real console evidence establishes their repair.
