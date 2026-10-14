# A14B agent memory evidence

## Summary

A14B adds optional scoped memory tools and connects one managed native planner.
It retains typed outputs for small decisions.
Engineering checks and small real comparisons establish the available boundaries.
A larger test remains pending for joint design with the operator.

Parent: [alpha plan](../alpha-plan/index.md).
Requirements: [A14B](../alpha-plan/work-items/a14b.md).
Implementation: [direct agent memory](../../subsystems/agent-memory.md) and [draft sessions](../../subsystems/draft-sessions.md).
Historical failures: [A14 cooperation](a14-cooperation.md).

## Source and deployment

Work starts from `c1e645f082c7a7163f61bd0191dcd2a7c5cbc03c`.
Evidence stays under `.runtime/alpha/20261014-a14b-implementation/`.
No dependency, lock, model route, or Minecraft world changes.
Real comparisons use `config.alpha.openrouter.yaml` and `qwen/qwen3.8-flash`.
They retain the complete Spamuel seed in isolated checkpoints.
Introduced preferences are synthetic user inputs through `UserInputProxy`.
They are not existing preferences or human ratings.

## Candidate inventory and decisions

The historical house run recorded 339 decisions across its workers and only two actor attempts.
Its original worker recorded 54 planner decisions, 73 reviewer decisions, and one verifier decision.
Those receipts establish repeated planning and semantic review as expensive paths.
The later diamond planner's first native turn used 49 model requests and 106 tools over 1,042.879 seconds.
It consumed 2,385,411 input tokens and 48,738 output tokens, including reasoning.
These are recorded provider statistics, not estimates.

Small MCP edits alone did not resolve that cost.
Captured house outputs include 149 planner results with a median size of 2,298 characters.
Their range is 948 through 21,506 characters.
The 189 reviewer results have a median size of 835 characters and a range of 405 through 16,779.
The sole verifier result contains 409 characters.
`inventory.json` retains these measurements and their source locations.
Its syntax inspection found no invalid JSON in those raw outputs.
Schema failures and individual inference costs remain unavailable from that inspection.
Later planner groups previously retained accumulated native conversation.
They now receive fresh session identities, accepted draft state, explicit evidence, and optional search.
Repair turns retain the same conversation within their group.
Queries can prepare eight missing embedding chunks by default instead of forcing a full index.

| Candidate | Evidence and decision |
| --- | --- |
| Minecraft planner | Keep its existing draft tools. Add direct memory and separate later decision groups. Historical planning cost justifies this focus |
| Review and retention | Keep typed decisions. The recorded failures concern judgment and repeated reviews. No measured large-output repair benefit supports migration |
| Goals and action selection | Keep typed decisions. Native tool overhead requires a measured benefit before migration |
| Attention and context selection | Keep known-ID selection. Preserve current invalid-ID recovery. No measured draft benefit supports migration |
| Appraisal and workspace | Keep typed decisions. No measured draft benefit supports migration |
| Ontology and reflection | Defer adoption until A18 and A19 implement those workflows |

Unavailable per-candidate costs remain unavailable.
Schema size alone does not establish inference latency or rejection cost.
The small note comparison measures transport choices, not the entire cognition pipeline.

## Engineering checks

The added tests use real records, snapshots, checkpoints, MCP listeners, and a separate client process.
They cover graph text, source pages, temporal validity, hidden endpoints, and nested hidden sources.
They also cover categories, entities, time, frontier, independent grants, refresh, and source conflicts.
An unrelated append leaves an unchanged source guard valid.

Cache tests exercise incomplete semantic coverage without another model.
Their manually stored vectors establish cache mechanics only.
A closed local endpoint establishes actual provider fallback.
Cancellation during its retry produces no returned search result.

A managed-run test invokes the real kernel and its workflow memory factory.
Its MCP query reads the authoritative owner's record without checkpoint restoration.
The accepted memory record set remains unchanged.
Standalone listener checks establish subprocess access and endpoint shutdown.

Foundation and final documentation results remain in the evidence directory's logs and exit files.
The root unit repeat passed 568 tests in 623.830 seconds.
Three later local regressions added required graph-ID, short-query, and session-close checks.
The final focused run passed 44 tests over their affected modules in 13.669 seconds.
The final contract run passed 63 tests in 2.100 seconds.
Failed development launches remain visible beside corrected repeats.
The first focused run exposed irrelevant zero-score results from direct lexical search.
The repaired direct path returns an empty result while ordinary cognition retains its existing context selection.
The first managed test supplied a positional argument to a keyword-only API.
The corrected test passes through the actual supervisor.

## Small real comparisons

[`dev.work_items.alpha_a14b_memory_gate.run`][] compares preselected context with native memory tools.
Each trial restores its own stopped checkpoint.
Fixture preparation is separate from decision latency.
The scenarios cover preference paraphrase, a newer strict requirement, and absent preference evidence.
One native trial requests a small repair in the same conversation.

```powershell
uv run --locked python dev/work_items/alpha_a14b_memory_gate.py --config config.alpha.openrouter.yaml --out .runtime/alpha/a14b-small-repeat --trials 2
```

The gate records actual source IDs, field checks, inspected revisions, model requests, tokens, tool calls, and decision latency.
Its first preparation run stopped before inference because it repeatedly rebuilt the same full-history fixture.
That stopped receipt remains under `comparison/`.
The corrected gate prepares each scenario once and restores isolated copies for both modes.
Its receipts remain under `comparison-repeat/`.

That first inference comparison used reasoning on for typed decisions and reasoning off for native decisions.
It also supplied a typed temperature of `0.1` while native sessions used the provider default.
These results are diagnostic evidence, not a controlled performance comparison.
The aligned repeat uses reasoning off, provider temperature defaults, and serial tool calls for both paths.
It restores the same prepared checkpoints and records its results under `comparison-aligned/`.

A failed baseline decision remains a failed trial.
Correct prose does not erase an incorrect typed choice.
In the diagnostic comparison, both paraphrase baseline trials selected `ask` while their explanations described a compact layout.
Both native paraphrase trials selected `compact` and `quick` and cited the original preference.
One retained a draft through an explicit follow-up repair.
Native tool cost does not establish lower latency by itself.
Human review remains pending.

The aligned repeat passed all 12 trials: two trials per scenario and interface.
Both interfaces chose the expected layout and effort.
The absent-preference oracle permits either practical layout with quick effort.
The native preference trials inspected and cited the original user statement.
One repaired its draft in the same session.
Both strict native trials used the sufficient current request without a memory search.
No trial changed accepted memory or performed a file or world task.

| Scenario | Preselected seconds | Native seconds |
| --- | --- | --- |
| Preference paraphrase | 31.906, 32.484 | 46.437, 66.578 |
| New strict requirement | 32.656, 31.703 | 14.671, 23.828 |
| Absent preference | 27.125, 28.546 | 44.312, 16.671 |

The overall medians are 31.805 seconds for preselected context and 34.070 seconds for native tools.
Both modes include the same scoped snapshot preparation.
Listener setup, tool discovery, queries, and repair remain inside native decision timing.
The native trials used 20 model requests and 14 tools, versus six model requests for preselected context.
Reported chat input totals were 119,106 native tokens and 19,114 preselected tokens.
Reported output totals were 2,537 native tokens and 610 preselected tokens.
Embedding requests remain separate in the source traces and prompt logs.
Native aggregate dollar cost remains unavailable from its turn statistics.

These results support optional tools for sessions that need direct evidence and retained edits.
They do not support replacing every small typed decision or claiming a general speed improvement.
The one forced native repair also adds work to its aggregate cost.
Cache order alternates between interfaces, but shared cache warmth remains a comparison limitation.

## Preserved graph and local HTML

`graph-inspection.json` reads the actual preserved Spamuel graph through the new tools.
Its view contains 50 concepts, 1,387 nodes, and 1,625 edges.
Lexical graph search, stored neighbors, and original-source lookup pass without changing the checkpoint.
This inspection performs no new extraction or model inference.

The first strict build passed in 81.53 seconds.
Local inspection opened the index, changed guides, and generated API pages.
It checked 4,222 local links and anchors across 11 pages with zero errors.
Final regeneration and inspection remain in their own logs and receipt.
Headless Brave produced no DOM within its bounded attempt.
The owned browser processes stopped afterward.
Rendered appearance remains unavailable; actual HTML contents and links remain verified.

## Remaining acceptance

The operator withdrew diamond testing.
This delivery starts no game server, body actor, house gate, or diamond gate.
World execution, integrated interruption recovery, public reply latency, and immersion remain separate pending acceptance.
The next step is a jointly designed test with modest complexity, a useful outcome, and an agreed observation budget.
