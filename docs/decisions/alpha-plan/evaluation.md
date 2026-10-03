# Alpha evaluation and session evidence

## Summary

Engineering, behavioral quality, and user immersion require different evidence.
Reviewed scenarios establish acceptance facts.
Real endpoints establish behavior.
The user records continuity and regeneration wishes.
Failures and unavailable services remain explicit in the final handoff.

Parent: [manual milestones](verification.md).
Related: [later milestones](cognition-verification.md) and [A01](work-items/a01.md).

## Evaluation contract

A01 defines stable scenarios and a separate holdout set.
Routing labels come from reviewed task facts.
The current LLM is a comparison, rather than the ground-truth label.

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
Use the baseline to set persona-quality and latency targets before tuning prompts.
Report the sample size and conditions beside every rate.

The automated evaluator runs after generation.
It must not secretly rewrite replies until it likes them.
Protocol repair for malformed model output remains distinct from rewriting valid personality output.

## Live failures and handoffs

Endpoint absence is an infrastructure failure.
Transport recovery may retry while the operator keeps the run active.
Logic errors fail loudly.
Expected gameplay failure is a routing outcome, rather than an infrastructure retry.

Each completed item supplies commands, configuration, fixture details, revisions, journals, and a short limit statement.
The console must let the user find the same evidence without reading implementation source.
