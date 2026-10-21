# Direct memory tools for native decisions

## Summary

Native agents can search accepted records and inspect their original sources through an explicit workflow grant.
Search remains optional.
The workflow chooses the view and decision policy.
The runtime supplies stable snapshots, graph pages, embedding coverage, cancellation, and source revision guards.

Parent: [workflow runtime](../runtime.md).
Related: [draft sessions](draft-sessions.md), [experienced memory](experienced-memory.md), and [A14B evidence](../work_items/verification/a14b-agent-memory.md).

## Construction and callers

[`pm.memory.agent_tools.MemoryView`][] defines the permitted IDs, reality/frame pairs, frontier, filters, and response limits.
Required sources and speakers must appear in its permitted IDs.
Filters never expand the grant.
Query words rank relevance within that grant.

[`pm.memory.agent_tools.MemoryToolConfig`][] selects semantic, lexical, and graph tools.
At least one group must be enabled.
Its optional embedding service reads vectors from the shared records and ranks them with FAISS.
Construction performs no network work.

[`pm.memory.agent_tools.MemoryReadSession`][] captures the accepted owner and constructs the read tools.
The host owns this session and closes it after agent cleanup.
Optional `refresh_view` supplies a host selector for explicit refresh.
The agent cannot select a broader view.

[`pm.memory.store_tools.build_mcp`][] accepts `memory=read_session` beside its existing environment and batch.
Legacy tools retain their existing environment grant.
Memory tools retain their separate accepted view.
Without `memory`, construction adds no tools or embedding requests.

[`pm.runs.draft_session.run_draft_session`][] accepts the same optional session.
It mounts a separate `memory` server on an OS-assigned loopback port.
The native process receives its private descriptor in `draft-mcp.json`.
It does not restore the live owner's checkpoint.

## Tool results and limits

| Tool | Result |
| --- | --- |
| `memory_view` | Snapshot identity, visible categories, grant bounds, enabled groups, limits, and inspected revisions |
| `memory_search` | Ranked excerpts, original IDs, revisions, dates, attribution, score components, and semantic coverage |
| `memory_get` | Original serialized JSON in bounded pages with numbered lines |
| `memory_sources` | Pages of stored supporting references, including original sources and speakers |
| `memory_neighbors` | One graph hop with edge and endpoint metadata, source IDs, and pagination |
| `memory_refresh` | Explicit snapshot replacement through the host selector |

Graph configuration adds `memory_neighbors`.
A refresh selector adds `memory_refresh`.
Search accepts `auto`, `lexical`, `semantic`, or `combined` according to the enabled groups.
Graph-only configuration searches graph text with lexical ranking.

Default limits are 16 results, 1,200 excerpt characters, 4,000 record characters, and 16 graph relations per page.
`memory_get` offsets count characters in the original indented JSON.
Its returned `next_offset` continues that exact representation.
Source and graph offsets count entries.
Unavailable IDs use one generic error.

Startup embeds every missing record and saves its vector in the pickle checkpoint.
Search prepares any remaining missing record vectors before FAISS ranking.
Missing coverage does not hide records from lookup or lexical search.
Provider errors produce an explicit lexical fallback.
Cancellation and index errors propagate.

## Evidence boundaries

The session removes records with unavailable typed references before search or lookup.
It repeats this check for nested summaries and graph dependencies.
Hidden endpoints cannot enter graph labels, counts, or source pages.
Adjacency indexes use only permitted stored edges.
The tools perform no graph extraction or answering-agent call.

Graph text includes node names, concept descriptions, and edge `fact_text`.
Results retain temporal validity and supersession fields.
Scores describe relevance.
They do not establish truth or create a personality policy.
Original statements, corrections, speculation, companion speech, and old world observations retain their evidence roles.

## Managed ownership and operation

The [managed memory lifecycle](agent-memory-lifecycle.md) explains the enabled planner view, commit guards, cancellation, restart, and diagnosis.
The [small comparison gate](../work_items/verification/a14b-agent-memory.md) measures source use, practical choices, and tool overhead.
It does not establish house completion, public reply latency, or human immersion.
