# Alpha evaluation and session evidence

## Summary

Engineering, behavioral quality, and user immersion require different evidence.
Reviewed scenarios establish acceptance facts.
Real endpoints establish behavior.
The user records continuity and regeneration wishes.
Failures and unavailable services remain explicit in the final handoff.

Parent: [manual milestones](verification.md).
Related: [later milestones](cognition-verification.md), [personal trial](personal-trial.md), and [A01](work-items/a01.md).

## Evaluation contract

A01 defines stable scenarios and a separate holdout set.
The [delivered matrix](baseline/scenarios.md) records review status and holdout exposure.
Routing labels come from reviewed task facts.
The current LLM is a comparison, rather than the ground-truth label.
The [model policy](model-policy.md) keeps the LLM as the default and limits optional Laya to simple tested uses.
Straightforward synthetic live tests assert independently defined source facts without separate human-label approval.
Human review remains necessary for ambiguous interpretation and persona immersion.
The old corpus's pending review does not block unrelated LLM work.

A18B supplies the [cognitive evaluation repertoire](../../subsystems/cognitive-evaluation.md) and
its [blinded review contract](../../subsystems/cognitive-review-boundaries.md).
Native tests cover the evaluation boundaries; real model calls calibrate the reviewers.
Fresh conversations and saved-evidence review carry separate execution scopes.
The owner's 24-hour example does not impose a suite duration during alpha or beta.

Track these measures:

| Measure | Evidence |
| --- | --- |
| Routing quality | Human agreement, class confusion, abstention, fallback, and critical misroutes |
| Latency and cost | Stage timing, request count, model contention, and background consumption |
| Memory continuity | Correct retrieval across distraction and restart; reality/provenance attribution |
| Emotional continuity | Traceable event-to-state-to-behavior effects; plausible regulation over time |
| Agency | Initiated goals, sustained progress, useful stopping, and justified changes of method |
| Minecraft success | Inventory/world changes, task acceptance, recovery outcomes, and return evidence |
| Reflection effectiveness | Supported belief updates and improved behavior after a concrete intervention |
| Immersion | User rating per exchange, contradiction incidents, and desire to regenerate |

Measure zero unsupported task completion receipts and zero overlapping body mutations in the acceptance scenarios.
Track accidental false public claims separately from explicitly intended deception.
Use the baseline to define persona-quality comparisons before tuning prompts.
The owner currently accepts slow conversation under [companion development priorities](companion-development.md).
Latency and cost are diagnostic measures, not current companion acceptance thresholds.
Do not reduce required evidence or cognitive depth merely to improve those measures.
Report the sample size and conditions beside every rate.

The automated evaluator runs after generation.
It must not secretly rewrite replies until it likes them.
Protocol repair for malformed model output remains distinct from rewriting valid personality output.

For A16 through A19, retain the selected causal path and the next relevant behavior.
Distinguish a typed proposal, accepted state change, chosen action, and observed consequence.
Record opposing evidence and unsuccessful trials.
Machine assertions establish source and execution facts.
Human review establishes neither a causal fact nor an automatic truth label.

The owner's [personal trial](personal-trial.md) precedes broad A20 hardening and spans multiple actual calendar days.
Its ordinary conversation uses optional feedback instead of a required rating after each reply.
Detailed integrated acceptance keeps its existing per-exchange review contract.
Separate trial bug fixes, runtime learning, design feedback, and later operator feature changes.
Automated summaries cannot replace the owner's review of whether the experience fits the intended companion.

Large Minecraft fixtures remain deferred rather than replacement requirements for cognitive quality.
Use modest real tasks when an acceptance case needs an external consequence.
The [revised work items](index.md#ordered-delivery) define the selected companion cases.

## Optional Laya acceptance

Write use-specific unit tests that call the real endpoint before enabling a Laya classifier.
Use realistic synthetic data, ordinary application formats, contradictions, distractors, missing evidence, and captured inputs when available.
Assert the expected class from source facts independently of model answers.
Retain untouched cases and record decisive evidence at different input positions.
All required cases must pass, and a useful reason for adoption must remain.
Confidence thresholds cannot excuse a confidently wrong answer.

No Laya adoption rate or 95% target remains.
If a candidate needs expensive tuning, retraining, or integration work, reject it and keep the LLM.
Its rejected experiment does not become an alpha failure or a prerequisite for later items.

## Live failures and handoffs

Endpoint absence is an infrastructure failure.
An unavailable optional Laya experiment leaves that candidate disabled, rather than blocking the normal LLM path.
Failures in required direct Python LLM calls, context handling, or agent compaction require immediate diagnosis and repair.
Transport recovery may retry while the operator keeps the run active.
Logic errors fail loudly.
Expected gameplay failure is a routing outcome, rather than an infrastructure retry.

Each completed item supplies commands, configuration, fixture details, revisions, journals, and a short limit statement.
The console must let the user find the same evidence without reading implementation source.
