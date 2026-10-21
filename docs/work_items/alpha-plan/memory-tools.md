# Scoped memory search and graph tools for agents

## Summary

A14B supplies optional memory tools for native pm-coder sessions.
The agent can search original records, inspect graph neighbors, and select evidence for its own decision.
Each workflow defines the available view.
The runtime shares retrieval and MCP wiring.
The real managed Minecraft planner enables these tools.
Other workflows retain their existing paths.
The [implementation guide](../../subsystems/agent-memory.md) records the current API and limits.
The design below remains the rationale and rollout contract.

Parent: [A14B](work-items/a14b.md).
Related: [experienced memory](../../subsystems/experienced-memory.md), [draft sessions](../../subsystems/draft-sessions.md), and [memory consolidation](../../subsystems/memory-consolidation.md).

## Current implementation and gap

[`pm.memory.store_tools.build_mcp`][] exposes scoped `ghost_get`, literal `ghost_search`, and `ghost_gwt` tools.
[`workflows.dialogue_assistant.host.build_assistant_mcp_factory`][] supplies that server to Assistant workflow invocations.
[`pm.runs.draft_session.run_draft_session`][] accepts workflow-owned MCP servers for a native agent session.
These entry points use factory functions rather than an abstract MCP base class.

[`pm.memory.store.MemoryStore.retrieve_experiences`][] provides semantic and lexical retrieval from a stable permitted snapshot.
[`pm.memory.retrieval.scoped_records`][] applies source, frontier, and reality/frame boundaries before indexing.
Default cognition retrieval still excludes `GraphNode`, `GraphEdge`, and `ConceptNode`.
The optional agent tools explicitly include their useful text.

Graph models, record indexes, reference fields, and checkpoint serialization remain in the current runtime.
A source inspection on 2026-10-14 loaded the unchanged Spamuel seed through `MemoryStore.from_yaml`.
It contained 11,202 records: 50 concepts, 1,387 graph nodes, and 1,625 graph edges.
These counts establish preserved data, not useful graph retrieval.

The older `pm_next/pm_core/subsystems/memory/graph_memory.py` contains graph extraction and an `answer_question` pipeline.
Its retrieval, question expansion, pruning, and answer synthesis are design material.
A14B gives raw retrieval tools to the active agent instead of adding that answering pipeline.
The old runtime has not been exercised by this inspection.

## Runtime option and workflow views

Shared MCP construction accepts an optional [`pm.memory.agent_tools.MemoryReadSession`][].
The configuration selects semantic search, lexical search, graph tools, or a combination, plus one workflow-defined view.
Disabled memory configuration preserves the existing tools and starts no memory listener or embedding request.
Enabled configuration requires an explicit view; it does not silently expose the whole store.

The implementation reuses the existing factories and listener helper.
A workflow can mount the shared tools on its task-group server or attach a separate memory server to the same session.
Native draft sessions attach a separate scoped memory server.
Ordinary factories can mount the same tools through their optional `memory` argument.
No new inheritance hierarchy is required.
The [implementation guide](../../subsystems/agent-memory.md) records exact configuration and API names.

The runtime owns snapshots, search indexes, graph queries, tool registration, cancellation, and source traces.
Workflow code owns view selection, domain relevance, and decision policy.
`pm/` must not import workflow code.
All tool groups read the same accepted memory owner through their selected views.

A view defines these inputs:

- Permitted record IDs or a host-resolved selector over accepted memory.
- Allowed reality/frame pairs and the accepted causal frontier.
- Record categories, relevant entities, and an optional time interval.
- Result limits, excerpt limits, and graph page sizes.
- Relevant source and speaker records required to interpret the selected evidence.

Separate visibility from ranking hints.
Visibility defines available records; ranking hints help find useful results within them.
For Minecraft, a view can include user preferences, prior task outcomes, and relevant world history.
A narrow word filter must not hide paraphrases, corrections, or related evidence.
The agent can browse freely within the declared view.
Broader access follows an explicit workflow grant rather than automatic graph expansion.

## Direct tools

[`pm.memory.agent_tools.MemoryReadSession`][] supplies these operations.
An optional host selector also supplies `memory_refresh`.

| Tool | Result |
| --- | --- |
| `memory_view` | Available categories, scope, snapshot identity, and query limits |
| `memory_search` | Ranked excerpts from semantic, lexical, or combined search, with original IDs and score components |
| `memory_get` | Numbered original record text and metadata, with bounded pages for long records |
| `memory_sources` | Original evidence linked to a summary, belief, or graph relation, within the same view |
| `memory_neighbors` | A bounded page of incoming or outgoing graph relations, endpoint records, and source references |

The agent can search with natural language, inspect a result, and follow its neighbors.
Semantic search uses the shared embedding service and cache.
Lexical search remains available when semantic coverage is incomplete or the provider is unavailable.
Expose the selected method, coverage, and fallback reason in the result.
Tool construction performs no network work.

No tool runs another language-model agent to answer the question.
The native agent reads source excerpts and synthesizes its own decision.
The backend supplies result metadata and serialization.
Search remains optional; a workflow does not require a lookup before every decision.

## Evidence and lifecycle

The [evidence contract](memory-tool-evidence.md) defines graph indexing, source attribution, snapshots, subprocess access, cancellation, and restart.
Its [acceptance cases](memory-tool-evidence.md#verify) require real source inspection and measured agent decisions.

## Delivery order and handoff

First define views and the shared runtime option, using current scoped snapshots and MCP factories.
Then add direct retrieval and source inspection, followed by indexed graph search and neighbors.
Connect one representative managed pm-coder session to the authoritative memory owner.
Run the real source-use and lifecycle comparisons before wider rollout.

Record the final API names, view selectors, tool groups, index coverage, snapshot refresh policy, and subprocess connection path.
Link every enabled workflow to its view and source traces.
Keep preserved graph records, newly implemented tools, and verified agent decisions distinct in the handoff.
