# Scoped Minecraft workflow decisions

## Summary

The project workflow separates historical proof retention, task repair, node review, and final integration.
It selects evidence from task dependencies and current failures.
Scenario geometry and material provenance remain in the acceptance gate.
Real retained-task acceptance is still pending.

Parent: [task and gate review](gameplay-task-gate-review.md).
Contract: [separate M3 checks](gameplay-gate-contract-proposal.md).
State: [task evidence](minecraft-task-evidence.md).

## Workflow ownership

The implementation belongs to `workflows/minecraft_tasks`.
Its manifest declares an initial retention phase before planning.
The planner also checks retention during a resumed repair.
Existing task-store validation, certificate guards, body leases, and model activation remain unchanged.

The generic workflow selects stable node identities and recursive dependencies.
It does not select scopes through house coordinates, material names, or fixture node names.
The A13 gate still owns its exact geometry, iron provenance audit, saved home, and tested route requirements.

## Historical proof and current requirements

[`workflows.minecraft_tasks.steps.retain`][] makes a separate native inference request from archived specifications and certificates.
The request includes original dependency hashes, current specifications, current body observation, and prior rejection feedback.
The model returns explicit restoration IDs through `RetentionChoice`.
[`workflows.minecraft_tasks.evidence.retention_operation`][] derives renewals for superseded valid prerequisites from actual accepted state.
It copies every current specification field exactly into a fresh identity.
The model does not copy node fields or decide validity arithmetic.
The derived operation stays beside the original native response.

[`workflows.minecraft_tasks.evidence.retention_tree`][] prepares the proposed tree without a durable state change.
Every renewal needs a fresh leaf identity and the complete current specification.
Its original parent gains a required acceptance reference.
The existing restoration operation rejects missing recursive dependencies or changed original files.

Normal task revision archives the replaced current certificate and restores eligible original proof.
The selector prioritizes unfinished renewals before other physical work.
A renewal gains acceptance only through a new native actor and independent verification.
Historical recovery therefore cannot establish present safety.

## Repair and review inputs

[`workflows.minecraft_tasks.evidence.dependency_scope`][] follows task subtrees and recursive acceptance prerequisites.
The selected repair receives complete node criteria, ancestor contracts, and full task topology.
Omitted specifications stay exact in the durable tree.
A scoped patch cannot change an unrelated existing node.

[`workflows.minecraft_tasks.evidence.scoped_history`][] selects methods from the target and unfinished dependencies.
Completed prerequisites retain their original certificates and accessible journals.
Global recovery, transport waiting, body life, and failed recovery evidence remain explicit.
The full accumulated gameplay history stays in a separate original evidence file.

[`workflows.minecraft_tasks.steps.review`][] makes separate decisions for leaves, stories, and the root.
[`workflows.minecraft_tasks.evidence.review_nodes`][] selects unfinished nodes and the full root contract.
Valid completed nodes retain their original independent review and exact certificate.
Their old execution instructions do not describe current work or current stock.
Each decision receives its full criteria, dependency specifications, valid historical certificates, and independent current reads.
[`workflows.minecraft_tasks.steps.caller_plan_constraints`][] supplies the same complete caller contract to planning and review.
It retains the original receipt path and digest. These requirements are not acceptance evidence.
Named publication audits can require ordered travel or durable notes that static world conditions cannot establish.
The actor procedure must retain those effects, and the independent publication audit must still verify them.
The captured `0191-reviewer` input omitted this contract. Its native revision request therefore lacked the caller's separate route audit.
The workflow now includes that source without changing certificates or adding scenario-specific condition kinds.
Story reviews receive preceding child results.
Root review checks the complete request and integration criteria after the preceding decisions.
Only a complete accepted review sequence marks the tree revision as reviewed.

[`workflows.minecraft_tasks.evidence.protect_accepted_specifications`][] rejects ordinary repairs that remove or rewrite valid attained contracts.
Root integration can gain new required work under independent review.
A new current requirement needs a fresh node identity.
Fresh assumption failures can revoke old proof before repair.
The separate retention phase validates its own exact restoration and current renewals.

## Original evidence retrieval

[`workflows.minecraft_tasks.steps.reason`][] supplies an evidence catalog with each decision.
The catalog names task specifications, journals, actor reports, surveys, and verification files.
A compact input lists source groups and named external sources.
The complete catalog stays in the readable `evidence_index` source.
This keeps archived filename lists out of each selected decision.
A caller can supply additional original sources through `minecraft_evidence_sources`.
The A13 gate supplies the real companion seed and checkpoint.

The decision can return `evidence_requests` before its final result.
Each request names a catalog source, a numbered line window, and an optional text query.
[`workflows.minecraft_tasks.evidence.read_source`][] rejects undeclared sources and changed file hashes.
Retrieved lines remain in the next request and in the original inference receipt.
Repeated identical retrieval requests fail explicitly.

Every inference records its actual messages, schema, generation configuration, and measured context budget.
The gate accepts optional `--decision-generation` JSON settings for direct decisions.
These settings cannot change messages, model routing, streaming, or the result schema.
Provider controls therefore stay in the caller, outside the task runtime.
The selected provider supplies serving capacity or uses an explicit configured bound.
Providers without compatible tokenizer routes use the existing labelled token estimate.
Evidence selection does not replace real inference or the real companion history.

## Final integration and cancellation

[`workflows.minecraft_tasks.steps.finish`][] requires all original leaf, story, and root certificates.
It then collects fresh independent root reads and evaluates the complete root criteria again.
The body life must match the accepted root observation.
The scenario completion callback still audits its separate requirements before publication.

Cancellation keeps original actor journals and physical effects.
A later run resumes the same task through the existing worker and instruction boundaries.
Retained character startup skips the test-memory reset.
Fresh fixtures retain their existing reset procedure.

An explicit `--allow-model-change` records previous and current chat identities in the continuation receipt.
The gate still rejects changed embeddings, seed, or retained task paths.
The hosted configuration remains under ignored runtime evidence and contains credential references only.

## Provider protocol and live evidence

The [provider protocol](gameplay-provider-protocol.md) explains output allocation, reasoning replay, and hosted generation controls.
The [acceptance evidence](gameplay-scoped-acceptance-evidence.md) records native continuations, known failures, and current limits.
The selected-leaf observation source uses [`workflows.minecraft_tasks.steps.execution_observations`][].
It supplies current read-only facts without replacing required effects or independent verification.
