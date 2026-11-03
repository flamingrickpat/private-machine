# K003 execution timeline

## Summary

K003 is the owner's review tool for why a companion acted as it did.
The POC reads one simulation attempt and shows its conversation, model calls,
workflow phases, GWT cycles, and captured records in one timeline.
It is read-only. It does not run cognition, call a model, or change evidence.

Parent: [local Kanban board](../../workflows/kanban.md).
Concept: [execution timeline](../../subsystems/timeline.md).
Source report: [contributor A/B test](../../ab_test_2026-10-23.md).
Process: [work-item development](../process.md).

## Status

The POC is implemented and ready for owner review of the K002 acceptance runs.
It is not an accepted K003 finish. The owner has not reviewed the page yet, and
the richer graph, comparison, and live-run views remain open.

## Owner requirements from the board

The card asked for a modular replacement for the old memory lists and the
outdated operator console. The owner wants to inspect one asynchronous run:

- when workflows started and stopped;
- which prompts ran and what the model returned;
- which artifacts were generated;
- what entered the global workspace and what left it;
- a readable timeline, chat, nodes and edges where they help, and color coding;
- a way to trace why a companion did something from telemetry and execution
  evidence, not from a reconstructed story.

The card also says K003 must stay modular, and that it will later integrate
with the simulation harness.

## POC scope

The POC covers the questions the owner asks first during acceptance:

| Area | POC behavior |
| --- | --- |
| Run selection | List every simulation run and every attempt below the evidence root. |
| Timeline | Merge speech, actor deliveries, routing, model calls, workflow phases, GWT cycles, facts, failures, and harness agent calls by time. |
| Calls | Show the exact prompt messages, provider response, reasoning, usage events, latency, module, and invocation ID. |
| Workflows | Show the kernel journal phases and every local artifact. Open text artifacts and JSON receipts. |
| Cycles | Use the shared record formatter to show stimuli, mental-state changes, appraisals, codelets, structures, shards, decisions, and replies. |
| Records | Search captured records and follow their recorded links. |
| Chat | Read the public conversation in captured order and open the record behind each line. |
| Evidence honesty | Show missing evidence, wall-time-only rows, unreadable rows, and errors instead of inventing links. |

The POC does not implement:

- graph views or force-directed layouts;
- run-to-run comparison;
- branch or removal-test playback;
- live refresh while a run is executing;
- model or human scoring;
- a replacement for the simulation review capture and registration path.

## Evidence sources

The tool reads only files that a simulation attempt already wrote:

| Source | Use |
| --- | --- |
| `state.json`, `plan.json` | Run identity, participants, edges, scenario stages, facts, and failures. |
| `attempts/<n>/report.json` | Attempt status, controller timeline, failures, and delivered facts. |
| `attempts/<n>/evidence/<participant>/chat.json` | Public dialogue records. |
| `attempts/<n>/evidence/<participant>/activity.json` | Captured private records. |
| `attempts/<n>/evidence/<participant>/cycles/*.json` | GWT turns and cycle state. |
| `attempts/<n>/evidence/<participant>/logs/prompts/*` | Exact model requests, responses, usage, and readable prompt logs. |
| `attempts/<n>/evidence/<participant>/runtime/kernel/*` | Workflow journals, state files, role traces, and artifacts. |
| `attempts/<n>/agents/*.json` | Harness actor, draft-check, and stage-director calls. |
| `attempts/<n>/evidence/<participant>/runtime/telemetry.jsonl` | Routing evidence, available for later views. |

The tool does not import the application assembly, bus, workflows, or the simulation package.
It reads the persisted evidence format and projects only the fields it shows.

## Public interface

Start the page from the repository root:

```powershell
uv run --locked python -m pm.tools.timeline --port 8772
```

Open the printed loopback address.
Select a run, an attempt, and optionally one participant.

A selected view can be linked. The hash has this shape:

```text
#/run/<run_id>/<attempt>/<tab>
#/run/<run_id>/<attempt>/<tab>/<detail>/<participant>/<id>
```

The server binds to `0.0.0.0` by default.
It has no write endpoint.

## Verification

The POC was exercised against the actual K002 runs under
`.runtime/simulation/runs/`:

| Check | Result |
| --- | --- |
| Import and compile of the new package | Passed. |
| Focused timeline reader tests | 3 checks passed. |
| Simulation review tests after the formatter move | 5 focused checks passed. |
| Architecture import checks | 7 checks passed after adding the allowed `pm.tools` to `pm.introspection` edge. |
| Combined focused checks | 15 checks passed. |
| Real run listing | 4 runs found. |
| Two-Spamuel attempt 4 | 2 participants, 931 timeline events, 746 calls, 11 workflow invocations, 4 cycles. |
| Two-Spamuel attempt 3 | Call, cycle, workflow, record, and artifact details opened from the live server. |
| Cat attempt 5 | 1 participant, 2169 timeline events, 1693 calls, 34 workflow invocations, 9 cycles. |
| Browser page and API | Served on `127.0.0.1:8772` and `127.0.0.1:8773`; page, run list, attempt, timeline, call, cycle, record, workflow, and artifact requests returned 200. |
| JavaScript syntax | `node --check` passed. |

Not verified:

- The owner has not read the page yet.
- The original POC had no browser rendering check. The preparation check below supersedes this limitation.
- The page has not been used on a live or incomplete run.
- The POC does not prove a causal claim. It shows recorded links and recorded
  order, and it labels wall-only time and missing evidence.

## Overnight preparation on 2026-11-01

The saved cat attempt 5 opens from `.runtime/debugger/runs/` after the active
simulation runs were archived. The archive is
`.runtime/archive/20261101-overnight-prep/`. These historical runs remain
debugging evidence, not completed morning acceptance.

The timeline now validates its cycle and dialogue projections with named types.
Model-call and journal JSON retain concrete value types at their read boundary.
The page no longer silently stops at 4,000 events or 600 records. It searches
all summaries and shows an explicit count with incremental display.

Fresh preparation receipts are under `.runtime/overnight-prep-20261101/`.
The focused timeline, review, checkpoint, and architecture selection passed
16 checks. The configured strict type selection passed. A separate strict
timeline scan reports three diagnostics from the shared legacy formatter's
untyped signatures; it previously reported 248 diagnostics in the POC.
These signatures remain an explicit typing limitation.

The actual Chromium page rendered the captured cat evidence. API checks opened
record, cycle, model-call, and workflow details. Screenshots and browser receipts
are in the preparation folder. Owner quality review remains pending.

## Open work

1. Owner review of the K002 two-Spamuel and cat attempts with this page.
2. Add run-to-run and attempt-to-attempt comparison.
3. Add branch and removal-test playback when that harness work exists.
4. Add a graph view only for links that the evidence actually records.
5. Decide whether this page replaces the operator console or remains the
   post-run review surface.
6. Add live refresh only after the owner needs it.
