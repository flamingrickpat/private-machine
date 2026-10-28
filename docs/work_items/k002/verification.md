# K002 execution evidence

## Summary

K002 implementation and real acceptance remain in progress.
This record distinguishes focused engineering checks from live companion behavior.
No human quality rating has been supplied.

Parent: [K002](index.md).
Operation: [simulation and review](../../subsystems/simulation.md).

## Source and services

Implementation started from repository revision `aa57839e517b19d28776083dac5e17238a717690`.
The contributor source was `542143f4b01dfc75add7fff66bc9efac939cbb85`.
Each simulation attempt captures the actual local source and installed package origins.

The configured chat endpoint is `http://127.0.0.1:8080/v1`, model `qwen`.
The embedding endpoint is `http://127.0.0.1:8081/v1`, model `qwen3-embedding-0.6b`, with 1,024 dimensions.
The owner authorized restarting both services during acceptance.
The chat server previously divided its context across three slots.
It now reports one 131,072-token slot; embeddings report one 32,768-token slot.

The unchanged Spamuel canon contains 11,202 records.
Two distinct homes retain that canon and their own identities.
Connected personas use Other Spamuel as the partner display name.

## Focused checks

Evidence root: `.runtime/kanban/k002-implementation-20261027/`.
These selections overlap. Their counts must not be added as unique tests.

| Receipt | Result and scope |
| --- | --- |
| `affected-components-tests.log` | 194 focused runtime, storage, workflow, simulation, and architecture checks passed. |
| `related-loops-fixed-tests.log` | 66 wait, focus, private-thought, felt-state, and trigger checks passed. |
| `cycle-source-causality-tests.log` | 55 causal recovery, cognition state, publication, and ontology checks passed. |
| `durable-ack-tests.log` | 61 checks passed, including real HTTP delivery with an unavailable encoder. Acknowledgment requires no encoder request. |
| `final-harness-boundaries-tests.log` | 16 plan, review, coordinated restore, fork-clock, transport, and architecture checks passed. |
| `storage-heads-tests.log` | 48 storage checks passed, including the full Spamuel checkpoint round trip. |
| `parked-fork-fixed-tests.log` | A real parked kernel resumed after restart, older-checkpoint restore, and a fork into another home. |
| `scoped-http-cleanup-tests.log` | 22 MCP, memory-tool, and parked-kernel checks passed through real scoped HTTP listeners. |
| `embedding-quality.json` | The real embedding geometry probe passed. The restarted service also passed. |
| `state-projection-retry-tests.log` | 14 workflow and parked-recovery checks passed. A real Windows file lock did not repeat a completed phase. |
| `final-clean-export-fixed-tests.log` | 20 causal recovery checks passed, including refusal to overwrite an existing clean export. |
| `final-compatibility-tests.log` | 17 transport, coordinated checkpoint, review, and architecture checks passed. |

Earlier failed receipts remain beside the repair checks.
They include obsolete same-ID mutation expectations and repaired historical lookup, vector index, and cross-home control-file failures.
No full unit or contract suite was requested or run.

## Forced interruption and recovery

The real recovery gate lives in `dev/work_items/k002/recovery.py`.
It forks the full Spamuel checkpoint, resumes its approved dialogue, delivers a newer message, and waits for actual main-cognition inference.
It then terminates the owned process and reopens the ghost through normal clean recovery.

The first gate failed because speech preparation created ontology records without the selected action's authority.
The completed reply consequently depended on provisional records and disappeared from the accepted view.
That failure remains under `.runtime/kanban/k002-live-recovery-20261028/`.

The repaired gate passed under `.runtime/kanban/k002-live-recovery-20261028-fixed/`.
Approved action 11509 produced reply 11517, which survived forced termination and restart.
Newer message 11518 remained in raw history but disappeared from the restored accepted view.
The accepted tick remained 812. The gate retained the actual inference request and terminated process ID.
These results establish recovery behavior, not a human quality rating.

## Live scenarios

The current cat run is `20261028-134105-30637473` under `.runtime/simulation/runs/`.
It uses the unchanged full Spamuel canon and actual local inference.
Arrival is 23:58 local time. Midnight sleep and the 07:00 morning event remain mandatory.
The run has delivered the adoption and midnight sleep facts and is advancing through the overnight timers.

The current two-companion run is `20261028-134111-69653404`.
It owns two separate homes and PM processes, with reciprocal Other Spamuel connections.
Both processes produced actual speech. The harness retains delivery acknowledgments and source output IDs.
Attempt 1 failed when a transient Windows lock caused the runner to restart the kernel after a completed phase.
The resumed kernel rejected that phase as already executed.
The runner now retries only its diagnostic state-file projection. Role permission errors propagate without kernel replay.
The failed attempt remains separate from ready review evidence. The repaired live attempt is still required.

Earlier cat attempts remain available as development evidence.
One failed with HTTP 400 against a smaller serving slot; another exposed a historical-record lookup error.
The provider error body was not retained in that first attempt, so prompt size does not prove its exact rejection reason.
Earlier peer attempts also retain the actor identity error and its later repair.
Stopped or failed attempts are not ready companion-quality evidence.

## Browser and manual checks

The browser showed persona history, checkpoint choices, attempt reports, captured cycles, causal links, and separate feedback controls.
Screenshots and browser events remain under the implementation evidence root.
No human rating was entered.
An earlier strict manual build passed, with 2,178 local HTML links checked and no missing targets.
The final manual rebuild must include the remaining edits.

## Outstanding acceptance

- Finish all mandatory cat-scenario stages, including the morning event.
- Exercise two distinct PM processes and inspect actual routed speech.
- Inspect browser history, editable plans, captured cycles, causal links, and feedback placement.
- Rebuild and inspect the complete strict HTML manual.
- Record the final local commit without pushing.

## Owner checkpoint before the typed prompt refactor

The owner authorized intermediate local commits after pausing live acceptance.
The repaired peer attempt 2 completed both mandatory stages with actual speech from both PM processes.
Its transit ledger records accepted delivery in both directions. Later undelivered messages remain explicit pending entries.
The completed report awaits human quality review.

Cat attempt 2 reached 02:20:30 local time before private inference failed with HTTP 400.
The exact provider response reports 693,473 requested tokens against 131,072 available tokens.
The trigger prompt included 313 reflection records and their complete metadata.
The request log is `participants/primary/logs/prompts/10a46ed3abb04a288e495fcf8e62250f.json` within that run.
Source-revision maps, dispatch payloads, and memory traces dominate the reflection payload.
The mandatory 07:00 events remain untested. This checkpoint does not mark K002 complete.

The owner requested typed reflection contracts, centralized record and collection rendering, and shared conservative token budgeting.
Live acceptance remains stopped while that focused refactor proceeds.
