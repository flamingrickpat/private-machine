# K001 memory session lifetime

## Summary

K001 fixes the read-session leak described in the tester's [A/B report](../../ab_test_2026-10-23.md).
MCP callbacks now hold weak references to their sessions.
The current checkout reproduced the leak with the full Spamuel seed and real loopback MCP requests.
The repaired callbacks release the session, its snapshot, and its serialized records after host cleanup.

Parent: [managed memory lifecycle](../../subsystems/agent-memory-lifecycle.md).
API: [`pm.memory.agent_tools.MemoryReadSession.tools`][] and [`pm.memory.agent_tools.MemoryReadSession.server`][].

## Cause and repair

The earlier registration passed bound methods directly to FastMCP.
Its cached Pydantic validators retained those methods after listener shutdown.
Each method retained the read session, its configuration, and its per-session memory copies.
Calling `close()` revoked reads but did not break that ownership chain.

The tester's fork identified this path in commit `542143f4b01dfc75add7fff66bc9efac939cbb85`.
Its proposed repair cleared the session's data on closure.
K001 instead removes the strong reference from the registered callback to the session.
Both `MemoryReadSession.server()` and `build_mcp(memory=session)` use these callbacks.
The host retains the session until active requests and listener cleanup finish.

Each callback preserves the tool's signature, annotations, documentation, and synchronous or asynchronous execution.
The callback obtains a temporary bound method for one call.
A retained callback raises `GhostCancelled` after closure or owner release.
The implementation avoids `functools.wraps` because its `__wrapped__` field would retain the original bound method.
Accepted records, visibility grants, source guards, and tool names remain unchanged.

## Current verification

The starting repository revision was `7709070f2da58d458c43bb7dc9132a603babf7d7`.
The installed packages were FastMCP `3.4.7` and Pydantic `2.13.5`.
No dependency changed.
Package origins and the exact starting revision are in the evidence directory.

The focused run passed 26 tests in 21.344 seconds:

```powershell
uv run --locked python -m unittest tests.unit.pm.memory.test_agent_tools tests.unit.pm.host.test_mcp_listeners tests.unit.pm.memory.test_lineage -b
```

These tests cover real MCP reads, visibility, revision guards, cancellation, pending retrieval, and listener shutdown.
New regressions retain registered tools while releasing the session, store, source record, and view.
They also exercise retained callbacks after closure and owner release.
Both new regressions failed before the repair.
An intermediate registration error exposed missing wrapper annotations; that failed receipt remains available.

## Full-seed memory comparison

The probe loaded the unchanged `test_data/personas/spamuel/canon.yaml`, with 11,202 accepted records.
Each iteration constructed a fresh read session, started a loopback MCP server, and listed its tools.
It then called `memory_view` and `memory_get`, stopped the listener, closed the session, and ran garbage collection.
Only weak references to completed sessions remained in the probe.

The table compares the first three iterations of each process.
MB means 1,000,000 bytes of traced Python allocations above the pre-session baseline.

| Completed sessions | Retained sessions before | Retained sessions after | Retained serialized records before | Retained serialized records after | Python growth before | Python growth after |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 11,202 | 0 | 117.93 MB | 2.72 MB |
| 2 | 2 | 0 | 22,404 | 0 | 233.16 MB | 2.76 MB |
| 3 | 3 | 0 | 33,606 | 0 | 348.41 MB | 2.80 MB |

Snapshot membership maps and source maps followed the same retained-record counts.
All six repaired iterations released every completed session.
The sixth iteration retained 2.91 MB of traced allocations above baseline and no session records.
The remaining allocations include MCP initialization and validator caches.
These measurements establish the reported read-session retention defect and its repair.
They do not measure full-companion resident memory, day-long stability, model quality, or other possible leaks.
No model inference or Minecraft operation ran for this lifecycle check.

## Evidence

Evidence stays under `.runtime/kanban/k001-20261027-200041/`:

| File | Content |
| --- | --- |
| `origins.json`, `revision.txt` | Installed package origins and starting repository revision |
| `memory_probe.py` | Actual seed and loopback MCP measurement procedure |
| `before.json`, `before.log` | Three unrepaired iterations |
| `after-fixed.json`, `after-fixed.log` | Repaired repeated-session measurements |
| `regression-before.log` | Two failing regressions before the repair |
| `focused-tests.log`, `after.log` | Intermediate annotation errors |
| `focused-tests-fixed.log` | Passing focused tests |

The [managed memory lifecycle](../../subsystems/agent-memory-lifecycle.md) documents the required host ownership and cleanup order.
