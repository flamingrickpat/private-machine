# Reasoning controls and inference speed

## Summary

The LLM wrapper now accepts reasoning enablement, numeric allowances, effort levels and returned-reasoning exclusion.
The OpenRouter adapter translates these controls during development.
The local adapter uses the selected Strata server's native fields.
Independent codelets and attention batches run concurrently with stable result ordering.
Checkpoint encoding now uses a serializer boundary and one binary write.

Parent: [A17/A18 hardening](a17-a18-hardening.md).
Implementation: [inference settings](../../subsystems/inference-settings.md).
Related: [whole-conversation review](../../subsystems/async-conversation-review.md).
A19 remains on hold.

## Scope and source

The changes extend repository revision `c5134e9`, which contains the earlier hardening.
No dependency, installed package origin, model deployment or canonical seed changed.
Direct PM calls use the new reasoning profiles.
Native pm-coder sessions retain their existing driver controls.
The database and canonical YAML import path remain outside this change.

The original reviewed long-gap conversation remains accepted by the owner.
Its repeated dandelion reference does not require regeneration.
Its timings motivated this work; they do not establish a controlled performance comparison.

## Engineering checks

| Check | Result | Local evidence |
| --- | --- | --- |
| Foundation unit suite | 728 passed in 735.543 seconds | `.runtime/speed-unit-final.log` |
| Foundation contract suite | 75 passed in 4.565 seconds | `.runtime/speed-contract-final.log` |
| Reasoning and serializer boundary checks | 24 passed | `.runtime/speed-wrapper-focused.log` |
| Final stage-helper checks | 24 passed | `.runtime/speed-stage-support-final.log` |
| Final speech and policy checks | 24 passed | `.runtime/speed-final-focused.log` |
| Real speech review suite | 8 passed in 109.183 seconds | `.runtime/speed-live-speech-03.log` |
| Python compilation | Passed for `pm` and `workflows` | Delivery command |
| Strict HTML build | Passed | `.runtime/speed-docs-delivery.log` |
| Local HTML and API navigation | Passed | `.runtime/speed-browser/receipt.json` |

The full foundation suite began before the small exception-preservation repair in `stage_support`.
The affected stage checks passed afterward.
The real speech suite used the final review prompt.
Its captured inventory case requires an observation even when the claim uses humor.
The permitted exaggeration and deliberate-teasing case still passes.

The HTML check opens the local index, changed guides and exact API anchors.
Every checked local navigation target is an HTML file.
The strict build reports no unresolved links.

## Real provider and scheduling evidence

Three captured private prompts returned valid structured thoughts with bounded reasoning.
The observed counts were 423, 548 and 248 tokens for ceilings of 512, 768 and 1,536.
Evidence: `.runtime/alpha/20261015-speed-budget-probe/`.
These calls used different prompts; they do not establish equivalent conversational quality.

Four additional calls exercised disabled thinking, a numeric allowance, low effort and exclusion through the current wrapper.
All four returned the correct arithmetic answer.
Their driver later failed because it expected an on-disk request file that existed only in memory.
The failed driver and completed calls remain under `.runtime/alpha/20261015-speed-controls-probe/`.
A separate captured-startup replay passed generation and review under `.runtime/alpha/20261015-speed-startup-replay/`.
Local wire translation has deterministic checks; this delivery does not run local inference.

Actual conversation logs show overlapping codelet requests and overlapping attention requests.
The overlapping codelet requests have identical first two turns and different concrete jobs.
Evidence: `.runtime/latency-review/speed-overlap.json` and its referenced raw logs.
This establishes concurrent execution and prefix equality, without attributing a measured cache-hit or latency gain.

The isolated checkpoint benchmark retained identical bytes for a 36 MB snapshot.
The single-buffer writer took about 0.87 seconds, compared with 2.79–4.26 seconds for streaming JSON.
This measures encoding and sync cost, not the complete commit or conversation.
Evidence: `.runtime/latency-review/serialization-result.json`.

## Retained failures and pending review

`20261015-211528-hardening-recent-01` failed when private integration resolved an expectation without observation IDs.
The repaired prompt shows candidate IDs and text.
Semantic validation permits one explicit model correction and retains rejected proposals.
It never creates a host-written replacement.

`20261015-220900-speed-recent-02` failed when a startup reply invented game status.
The reviewer correctly blocked that reply.
Story now receives an explicit absence-of-observation statement.
Unsolicited speech remains permitted, including when new user speech arrives during preparation.

`20261015-222000-speed-recent-03` is a diagnostic conversation with source changes during execution.
Its first public reply invented inventory and passed the earlier reviewer.
The repaired reviewer blocks the captured inventory excerpt in the real speech suite.
Another live-suite attempt missed an advisory invented-event finding before the prompt repair.
That failed receipt remains visible in `.runtime/speed-live-speech-final.log`.
The final prompt removes the ready-made acceptance reason and requires a reason tied to actual evidence.

`20261015-speed-recent-04` is queued after diagnostic service cleanup.
It uses the full recent-history transform, real Minecraft and an independent Jesse model.
The runner captures its source before execution and registers its result after cleanup.
Inspect `.runtime/speed-recent-04.json` for the actual result.
Queue state is separate in `.runtime/speed-review-queue.json`.
Human quality review remains pending.
No complete-runtime speedup or unchanged conversational quality is claimed by these engineering checks.
