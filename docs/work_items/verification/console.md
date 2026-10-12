# Console verification record

## Summary

The console passed process, browser, memory, and live model checks on October 2, 2026.
The complete foundation run passed 253 tests.
Two later observer checks also passed.
Live checks covered two real exchanges, not the full conversation quality gate.

Parent: [evidence and quality](../../concepts/evidence.md).
Operation: [console procedure](../../workflows/console.md).

## Automated checks

| Check | Result |
| --- | --- |
| All root foundation modules | 253 passed in 374.633 seconds |
| Console worker integration | Deploy, two users, memory edits, stale rejection, creation, restart, stop, and kill passed |
| Bus observation | Detached observer copy and explicit observation errors passed |
| Browser regression | Chat sessions, user names, correlation, child definitions, edges, memory edits, timing, and kill passed |
| HTTP origin check | An unrelated browser origin received status 403 |
| Python compilation | Application and console source compiled |
| Documentation | `python -m mkdocs build --strict` passed |

The browser checks used Chromium through Playwright.
Playwright is an optional local verification tool.
The application has no browser automation dependency.

## Live Qwen evidence

The configured endpoint was `http://127.0.0.1:8080/v1` with the `qwen` alias.
Both exchanges completed through eight model calls and nine workflow invocations.
Replies retained their session and correlation IDs.
The second exchange also recorded child definitions and parent links.

| Model call | Elapsed seconds in the second exchange |
| --- | --- |
| Triage | 2.510 |
| Appraisal | 11.755 |
| Context | 10.032 |
| Attention | 13.936 |
| Goals | 2.993 |
| Workspace | 4.184 |
| Action selection | 4.390 |
| Story dialogue | 1.773 |

These measurements describe one saved exchange.
They do not establish a stable benchmark or a usefulness score.
Trace recording adds overhead.

The original live evidence remains under `.runtime/console-live/`.
The prepared `Spamuel / Qwen` deployment contains its accepted checkpoint and a copy of its trace.
Browser previews remain under `.runtime/console-browser/`.
The prepared worker is stopped so that the operator controls its next start.

## Limits

This change did not rerun the 200-exchange conversation gate or the Minecraft live gate.
It did not change the cognitive policy.
The default seed contains approximately 11,200 records and occupies about 30 MB.
Startup, snapshots, and durable commits can take substantial time with this seed.

## Go deeper

Read [next experiments](../experiments.md), [plan status](../plan.md), and [`console.server.Instance`][].
