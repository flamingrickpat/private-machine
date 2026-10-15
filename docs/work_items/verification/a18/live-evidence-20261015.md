# A18 live evidence, 2026-10-15

## Summary

This page records run provenance, retained discovery evidence, and limits of the requested gate.

Parent: [A18 verification](../a18-self-model.md).
Comparison: [matched persona evidence](persona-comparison.md).

## Run and provenance

The requested gate uses `.runtime/alpha/20261015-110410-a18/`.
All evidence paths in this section are relative to that ignored directory.
The application uses the complete, unchanged Spamuel seed and a fresh checkpoint.
The selected configuration is `config.alpha.openrouter.yaml`.
The model is OpenRouter `qwen/qwen3.8-flash`.

`origins.json` records source revision
`f8e783d99c0968f345c9bb52a3ebdd4a036a76b6`, interpreter, package origins, and source digests.
PM imports from this checkout; it has no installed `pm-next-v2` distribution.
The installed `pm-workflows` version is 0.7.1 at
`47e24d8e1dadd97e20b2143e374be8f6b29240bb`.
The installed `pm-coder` version is 0.10.0 at
`1e96e03c536bd6744b33afdc521f40985af5ff08`.
This session changes no dependency or deployment setting.

## Checkpoint failure

The decision reply timed out after 900 seconds.
The application surfaced `PermissionError: [WinError 5] Access is denied` while replacing `ghost.json` from a temporary file.
The first application stopped at 09:17:18 UTC, which was 11:17:18 local time.
The gate continued its wait before reporting the stored exception.
`facts.json` preserves the timeout and checkpoint exception.

The specific lock owner remains unverified.
The session read the checkpoint during live execution, so observer contention is one possible explanation.
No checkpoint read occurs during the restart phase from the external inspection commands.
The error does not establish lost durable data or an unavailable model endpoint.
The failed receipt remains part of the evidence.

## Final receipt and restart

`receipt.json` records `ok=false` after 28.4 minutes.
The process exits with code 1 and retains both failures.
The inspector's evidence-or-limit assertions report no additional failure.
The gate reports no late private commit after shutdown.
It records no stale capability because it performs no revision intervention.

`checkpoint-audit.json` records 11,293 final records, including the same 11,202 imported background records.
The five complete capability belief records equal their pre-restart snapshots, including metadata, references, confidence, and uncertainty fields.
The captured beliefs have no explicit validity-interval field.
The explicit cross-loop link remains unchanged.
The final store still has four inner frames and two main frames.
No new main frame establishes a fresh answer to the reopened prompt.

The reopened user turn is record 11306, with correlation `017040d3d46044d6af04dcc54336f4a7` and `exchange_status=pending`.
Reply 11307 cites old action schema 11222 and has an empty `reply_correlation`.
The gate accepts any emitted reply after submission, so its restart result does not establish a reply to this exchange.
Startup 11305 records a clean previous stop and a 772.7-second gap.
The emitted reply instead describes the original seed's 3,300-hour gap and claims continuous resentment in the void.
That stopped-time claim lacks execution evidence and remains separate from human naturalness review.

The final checkpoint and the retained action schema establish the older reply's provenance.
`restart-audit.json` preserves the matching records and comparison results.
`llm-logs/` retains copies of the inference logs.
`source/` retains the gate scripts used for this run.

Remaining work includes a clean gate repeat without concurrent checkpoint reads, source invalidation, narrative counterevidence, and a correlation-aware restart check.
Minecraft ownership and recovery acceptance remain separate.

## Scenario facts before the interrupted conversation

`checkpoint-audit-before-gate-restart.json` records 11,202 imported background records,
four inner-loop frames, two main-loop frames, and five capability beliefs.
Thought 11204 performed `capability_inspection` after 136.5 seconds without user input.
Its metadata labels the thought `inferred` and retains all five capability states.
Beliefs 11205 through 11209 retain their evidence revisions.
Four beliefs say `enabled`; the Minecraft body belief says `implemented`.

The retained evidence shows a discovery-to-conversation path:

1. Thought 11204 distinguishes implemented Minecraft source from an enabled body.
2. Appraisal 11268 cites that thought and connects the limit with resentment and uncertainty.
3. Goal 11269 names thought 11204 and user question 11211 as evidence, with `appraisal_id=11268`.
4. Action schema 11282 selects a capability explanation and names goal 11269 among its intentions.
5. Public reply 11284 describes text capabilities and treats Minecraft as unobserved.

`discovery-chain.json` retains these records and their metadata.
The episode index links inner episode 11204 to main episode 812 through evidence record 11204.
That explicit link establishes influence without a timestamp-only inference.
It does not establish an external body action or a successful Minecraft task.

The first capability-turn publication was the fallback:
"That reply became stale before I could send it. Please ask again."
The later limits-turn reply answers the earlier capability question.
The wrapper associates replies by arrival order, so turn names alone do not establish question alignment.
Reply 11284 also says the connection "reads disconnected".
The gate disables body activity and supplies no transport probe that establishes a disconnected endpoint.
That wording exceeds the inspected wiring evidence and requires a separate unsupported-claim review.
Thought 11286 performs `narrative_stitching` but does not establish a counterevidence-driven narrative belief revision.

## Coverage limits of the requested gate

The script does not exercise every required scenario.
It makes no deliberate source or configuration change and applies no narrative counterevidence.
Its initial declarations contain no feature whose strongest state is `intended` alone.
An absent capability inspection becomes a note rather than a failed assertion.
The script records thought attribution and capability snapshots without asserting their complete contract.
Its restart comparison checks belief keys and evidence revisions, not complete validity intervals or episode references.
Its offline-cycle check ends after the reopened reply, so it can count genuine restarted activity as stopped-time execution.
Preserve any resulting failure until a corrected interval check supplies new evidence.
