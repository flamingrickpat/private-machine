# Memory-tool evidence and lifecycle

## Summary

A14B's memory tools retain original evidence through graph queries, snapshots, and managed-agent connections.
The authoritative memory store remains the state owner.
This page defines the read boundaries and acceptance cases.
The [implementation guide](../../subsystems/agent-memory.md) documents the enabled planner connection.
The [verification record](../verification/a14b-agent-memory.md) separates engineering checks from pending behavioral acceptance.

Parent: [scoped memory tools](memory-tools.md).
Related: [A14B](work-items/a14b.md) and [experienced memory](../../subsystems/experienced-memory.md).

## Graph search and evidence

Index useful graph text explicitly: node names and content, concept descriptions, and edge `fact_text`.
Preserve record type, endpoints, relationship label, timestamps, `valid_at`, `invalid_at`, scope, and source-memory links.
Graph traversal uses stored relations without another extraction call.
Restoring the old graph writer is a separate decision after useful query behavior is established.

Return only permitted nodes, edges, and supporting sources.
A hidden endpoint or source must not leak through labels, excerpts, counts, or graph expansion.
Reuse the existing linked-source boundary rules for summaries and derived records.
Make one-hop traversal the initial operation, with pagination for further browsing.
Cache or index adjacency rather than repeatedly scanning the full graph for each neighbor request.

Every result retains its ID, revision, type, source attribution, date, and applicable historical or superseded status.
Search scores rank relevance; they do not establish truth.
Preserve conflicting statements and corrections so the agent can interpret their chronology.
A graph fact remains a sourced claim.
A generated companion utterance remains an utterance.
An old world observation remains historical and requires a current observation for present-world execution.

The wood-type example is an acceptance scenario, not an observed preference in the current history.
A retained user statement that any wood is acceptable can support a flexible review.
A newer explicit request for oak supplies the current requirement.
Absent preference evidence leaves room for a practical agent choice instead of forcing refusal.
Memory informs judgment without adding a new truth or agreeableness policy to [A09R](work-items/a09r.md).

## Snapshots, subprocesses, and lifecycle

Use a stable accepted snapshot for one decision group.
All searches and graph reads in that group retain the same view and revisions.
Expose original evidence IDs so the workflow can record the records it actually used.
Existing commit owners guard relevant source revisions before accepting state.
An unrelated memory append does not invalidate an unchanged source.
If the workflow refreshes its snapshot, retain the boundary and replace the old view explicitly.

Managed agents run in separate processes.
Define their connection to the authoritative memory owner before enabling these tools for Minecraft roles.
A runtime-owned read service can issue a scoped connection for the owning run and decision group.
The agent receives only the configured tool descriptors, and the server enforces that grant on every request.
Do not reopen the live checkpoint as another writable memory owner.
Checkpoint restoration also changes active managed-run receipts to interrupted state, so it is not a live query adapter.

Read tools cannot commit memory, publish speech, start tasks, or change the world.
Draft tools retain their separate state and commit contracts.
Cancellation stops pending retrieval, awaits session cleanup, and revokes the group's access before listener shutdown.
Completed embedding cache entries can remain after cancellation; no late accepted-state commit can occur.
On restart, obtain a new permitted snapshot and connection before resuming the native session.
Retain old evidence revisions in the trace without presenting the old grant as current.

## Verify

Use the current OpenRouter configuration and complete Spamuel history for real native-session comparisons.
Preserve failed trials and report human review separately.
Keep source captures and tool traces under a fresh ignored runtime directory.

1. Compare disabled, semantic-only, graph-only, and combined tools through the actual runtime construction path.
2. Retrieve an original user preference by paraphrase after distractions and restart.
3. Follow an existing graph edge to its endpoint and original memory without generating a replacement fact.
4. Verify that lookup, source expansion, and neighbors obey the same view, including nested summaries and hidden endpoints.
5. Present conflicting preferences and a newer explicit task constraint; inspect the source IDs and resulting judgment.
6. Verify a practical choice with absent preference evidence, without forced refusal or a mandatory lookup loop.
7. Verify incomplete embedding coverage and provider fallback without changing source attribution.
8. Cancel during search, refresh a changed source, restart, and isolate concurrent managed-run grants.
9. Inspect that searches leave accepted memory, workflow state, and the Minecraft world unchanged.

Use existing recorded preferences and graph sources where available.
If a wood-type preference is introduced, record it through the normal user-input path in an isolated checkpoint with full history.
The illustrative quote does not count as an existing source or a human rating.

Measure setup time, lookup latency, input/output tokens, native model requests, tool calls, and total decision latency.
Record graph and embedding coverage, result size, source inspection, and actual preference use separately.
Compare the same task and view with preselected context and optional direct tools over multiple trials.
Retain the simple path when tool overhead exceeds its benefit.
Run the foundation suites and regenerate the complete HTML site after implementation.
