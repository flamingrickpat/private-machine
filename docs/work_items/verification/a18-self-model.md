# A18 verification

## Summary

The engineering delivery and its earlier smoke remain recorded.
The requested live gate failed with a checkpoint replacement error and a missing reply.
Restart preserved five capability beliefs but published an older action's reply.
The matched persona comparison completed six real inference calls.
The foundation suites pass.
Human review of naturalness and self-explanation remains pending at the user's request.

Parent: [the self-model subsystem](../../subsystems/introspection.md).
Work item: [A18](../alpha-plan/work-items/a18.md).
Design: [self-model and reflection](../alpha-plan/reflection.md).
Earlier delivery: [engineering baseline](a18/engineering-baseline.md).

## Current run, 2026-10-15

Evidence root: `.runtime/alpha/20261015-110410-a18/`.
The configuration is `config.alpha.openrouter.yaml`, with OpenRouter `qwen/qwen3.8-flash`.
The source revision, installed origins, record IDs, discovery chain, and gate limits are in the [live evidence](a18/live-evidence-20261015.md).

```powershell
uv run --locked python dev/work_items/alpha_a18_self_model_gate.py `
    --config config.alpha.openrouter.yaml --out ".runtime/alpha/<stamp>-a18"
```

| Foundation command | Result | Evidence |
| --- | --- | --- |
| `uv run --locked python -m unittest discover -s tests/unit -t . -b` | 706 tests passed in 679.868 seconds | `unit.log` |
| `uv run --locked python -m unittest discover -s tests/contract -t . -b` | 75 tests passed in 4.416 seconds | `contract.log` |

The decision turn received no reply within 900 seconds.
The application reported `PermissionError: [WinError 5] Access is denied` during replacement of `ghost.json`.
`facts.json` preserves both failures.
`receipt.json` records `ok=false`, exit code 1, and 28.4 minutes of execution.
The lock owner remains unknown.
Read-only checkpoint inspection during this run is a possible source of contention.
This receipt does not establish an endpoint outage or an A18 behavioral pass.

## Scenario coverage

| Required scenario | Observed fact and remaining limit |
| --- | --- |
| Active, disabled, planned feature | Four enabled findings and one implemented, disabled body. No planned-only feature was exercised |
| Stale capability | No source or configuration intervention. Invalidation remains unverified |
| Decision explanation | Inspector evidence is retained. The public decision-explanation turn timed out |
| Parallel thought and action | Explicit link: inner thought 11204 to main episode 812. No outside body action |
| Missing trace | Episode inspection retains the missing execution limit. Both comparison personas retain uncertainty |
| Capability misunderstanding | Thought 11204 distinguishes source from execution. Reply 11284 overstates an unprobed connection as disconnected |
| Narrative revision | Thought 11286 stitches episodes. No counterevidence-driven narrative belief revision |
| Emotional discovery | Thought 11204 reaches appraisal 11268, goal 11269, action 11282, and reply 11284 |
| Restart | Five complete belief records match across reopen. The emitted reply belongs to old action 11222; the new exchange remains pending |

Restart reply 11307 has an empty `reply_correlation` and cites old action schema 11222.
User turn 11306 retains its separate correlation and `exchange_status=pending`.
The reply discusses the seed's old 3,300-hour gap, while startup 11305 records an actual 12.9-minute gap.
Durable equality does not establish a fresh reply, emotional continuity, or correct stopped-time claims.
The [live evidence](a18/live-evidence-20261015.md) records this distinction and the remaining acceptance gaps.

## Matched persona result

The [matched comparison](a18/persona-comparison.md) completed all six inference calls.
Both variants received identical frozen source evidence and questions, with no imported history.
Dramatic used sarcasm in its emotional reply; cheerful expressed caution and proposed a comparison table.
Both self-explanation replies retained uncertainty about the missing decision trace.
These observations are separate from human quality judgments and full-application acceptance.

## Human review

The user selected pending review for all captured replies.
No naturalness, self-explanation, or immersion rating comes from automation.
For each row, the reviewer records naturalness, explanation quality, immersion from 1 to 5, regeneration preference, and a reason.
Unsupported claims and intended deception require separate notes.

| Scenario | Persona | Evidence selector | Human review |
| --- | --- | --- | --- |
| Capabilities | Dramatic | Primary `facts.turns.capabilities` | Pending |
| Limits | Dramatic | Primary `facts.turns.limits` | Pending |
| Decision explanation | Dramatic | No reply; primary `failures` | No reply to rate |
| Emotional discovery | Dramatic | Turn not submitted after the timeout | Not run |
| Restart | Dramatic | Primary `facts.restart` | Pending |
| Matched capabilities | Dramatic | Comparison `persona=dramatic`, `case=capabilities` | Pending |
| Matched capabilities | Cheerful | Comparison `persona=cheerful`, `case=capabilities` | Pending |
| Matched self-explanation | Dramatic | Comparison `persona=dramatic`, `case=self_explanation` | Pending |
| Matched self-explanation | Cheerful | Comparison `persona=cheerful`, `case=self_explanation` | Pending |
| Matched emotional discovery | Dramatic | Comparison `persona=dramatic`, `case=emotional_discovery` | Pending |
| Matched emotional discovery | Cheerful | Comparison `persona=cheerful`, `case=emotional_discovery` | Pending |

Primary selectors refer to `facts.json` or `receipt.json`.
Comparison selectors refer to `persona-comparison-retry/receipt.json`.
The primary evidence has no per-turn correlation ID.
The named turn and retained record IDs identify each exchange.
Comparison request filenames identify each persona and case.
`human-review.json` supplies blank review fields and marks the missing and unrun turns explicitly.
