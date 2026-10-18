# Alpha decision register

## Summary

The user confirmed the LLM default and restricted optional Laya use after A02's poor quality results.
The user also specified a parallel reflective conscious loop and open-ended, self-directed Minecraft goals.
The user answered the three question batches.
The answers below govern the proposed alpha.
Seed import still needs evidence-based provenance handling.

Parent: [alpha plan](index.md).
Related: [cognitive architecture](cognition.md) and [Minecraft architecture](minecraft.md).

## Confirmed direction

D01: Use the normal LLM for selection, classification, and interpretation by default.
Laya was planned for frequent classification, but A02 found poor interpretation quality, including confidently wrong answers.
Retain the delivered technology for later experiments.
Use it only for the simplest classifications after realistic live-endpoint unit tests establish that specific use.
If making Laya work requires expensive effort, use the LLM and stop that experiment.
No item requires Laya adoption, model tuning, or retraining.
The [model policy](model-policy.md) governs all items, including later codelets.

D02: The inner-thought workflow is a second conscious loop.
It reads the main GWT and its own inner-thought shard.
Its attention favors private knoxels and the current mental landscape.
It can manage goals, speculate, theorize, plan, and manage its own processes.

Accepted thoughts and artifacts feed the main loop as self-origin stimuli.
Main appraisal can make a sad thought produce sadness.
An internal focus intention can change attention through the ordinary proposal path.
The companion chooses whether to share an idea, summarize its activity, or keep something private.
The user sees raw thoughts only through debugging.

The two loops share one companion identity and one Ghost.
They have separate attention frames and durable GWT shards.
They can run concurrently without directly overwriting each other's workspace.

D03: The companion chooses its own activity throughout the configured action space, including while the user is away.
House building is an example, rather than a privileged goal or a restriction to base maintenance.
Main appraisal can accept, refuse, defer, or reshape a user's task based on persistent state.
The inner loop can initiate its own goal and start the same generic task workflow.
Affection, resentment, curiosity, or mischief can produce different choices, including permitted destructive game actions.

Simple questions can route directly to Assistant dialogue and a concrete MCP call.
An inventory query does not require a managed task tree.
Longer or uncertain work uses the planner and specialist workflow.

D04: The classification node can select a relevant file using a scoped folder listing and metadata.
Use the LLM by default and deterministic metadata rules when they establish the required result.
Exact file input remains available.
Every classification has an INVALID outcome for absent, oversized, malformed, or uncertain input/results.
The specific classification can then route to a scoped coding agent.
The earlier 95% fast path / 5% fallback target is withdrawn.
Optional Laya selection and content classification need separate live tests before either is enabled.

D05: Ship the reusable client inside pm-coder and expose a public Python API.
D06: Use the always-present INVALID route for missing input and uncertainty.
D07: Reflection can change goals, beliefs, attention, and workflow configuration with durable history.
D08: Prolonged dark moods and rumination are core persona behavior.
Regulation must not enforce optimism or erase persistent negative state automatically.
D09: Evaluate Spamuel/Jesse and a contrasting cheerful persona.

D10: No role play means capability and private-ontology consistency.
For example, a companion without smell perception cannot casually remember liking a smell it experienced.
Expression must match implemented emotional state.
The user permits conscious lying.
Such a lie requires an explicit prior intent and does not change factual memory.
An unsupported generated claim cannot acquire a deception justification after generation.

Seed events still need explicit provenance and compatible capabilities.
Their precise historical interpretation is not settled by the no-role-play clarification.

D11: Remember startup, shutdown, and the elapsed offline interval as system events.
Feed startup/gap evidence into the first main cognitive cycle for appraisal.
The companion can have an emotional reaction to a long absence.
Do not invent executed activity during that interval.

D12: Use Agentic Babymode as the survival-light acceptance environment, plus isolated test worlds.
For A01, the user selected `C:/source/minecraft-server-real` and its existing world-wipe script.
The baseline records its actual seed and configuration instead of substituting the superflat TDD template.
D13: The first Minecraft milestone must deliver a finished small house and an iron pickaxe.
The iron must come from mining, rather than taking village supplies.
The earlier wood task remains a developer check, rather than this milestone.

The user also requested 10–20 ordered implementation items, fresh coding sessions, and manual milestone tests.
The user wants adaptive Minecraft work with many tool calls per meaningful subtask.
The current one-action constraint must be removed.

D14: Classification classes, descriptions, input limits, and uncertainty policies depend on the specific use case.
Optional Laya adoption requires realistic live-endpoint unit tests on that exact use.
Tests use independently defined source facts, realistic synthetic data, and captured application inputs when available.
Simple synthetic assertions need no separate human-label approval.
Do not spend extensive effort making a weak Laya candidate work.
Every classifier has an INVALID route to an LLM fallback for missing, ambiguous, or unusable evidence.
The A01 report vocabulary is a demonstration corpus, rather than a global set of classes.
Preserve its historical review and holdout status without making that review a prerequisite for unrelated LLM delivery.

D15: The normal LLM path must work through direct Python calls and pm-coder agent sessions.
LLM infrastructure failures and lost required context require immediate diagnosis and repair.
Preserve pm-coder's automatic compaction for agent execution history.
Direct chat callers must select suitable task context or explicitly compact it while preserving required evidence.
Laya's decision-head limits do not configure chat.

## Current companion development direction

D16: On 2026-10-14, the owner prioritized companion quality over conversation latency during development.
Useful reasoning, memory context, and emotional depth take precedence over lower inference cost.
Performance remains diagnostic evidence.
Repeated work without useful progress remains a behavior problem.
Existing reliability, cancellation, and finite-context controls remain active.

D17: The owner reduced the priority of large Minecraft tasks and requested curated requirements from historical sources.
A16 through A19 can proceed without completing large A14 gameplay scenarios.
Each item still verifies the interfaces it actually uses.
Deferred and failed gameplay acceptance remains visible.
The preserved original brief remains unchanged.

The [companion development policy](companion-development.md) applies these decisions.
The [curated source guide](cognitive-sources.md) defines the selected adaptations and excluded ideas.

## Pre-trial direction on 2026-10-17

D18: Deliver A19 reflection, A19B memory consolidation, A19C daily-use interface polish, and A19D container-agent dialogue before the personal trial.
The owner then chats with one continuing character across multiple real days.
PM feature work pauses during that interval. Bug fixes remain permitted.
Broad A20 hardening follows the owner's review so repairs preserve the intended experience.
Read the [personal-trial contract](personal-trial.md).

D19: Use `StateNeeds` and the full mental-state vector as the sole live representation of needs and cognition axes.
Retire the older parallel holders through an explicit import path.
Add only the measured `memory_integrity` need for this phase.
New eligible experience lowers integrity. Accepted consolidation raises it.
The wider needs system remains beta work.

D20: Favor simple, interruption-tolerant consolidation over elaborate processing metadata.
Partition sources by conversation or internal channel and reality scope before embedding-based agglomerative clustering.
Generate topical memory, temporal summaries, extracted facts, and a record graph for graph search.
Complete one bounded input window before committing its results and the single cursor together.
An interrupted window retries from the last accepted cursor.
Use durable commit order because reserved record IDs can commit out of order.
Do not add per-stage completion flags, per-channel cursors, or a second processing ledger.
Retain ordinary source membership and graph links.
The [memory design](memory-consolidation.md) and [atomic-window contract](consolidation-commits.md) apply this decision.

D21: The companion can converse with an existing execution harness inside a persistent Linux Docker environment.
The container agent can install tools and build software for the companion or user with root-capable container execution.
This is separate from a human conversation, a peer companion, and the larger development workflow.
Hermes is a candidate, with the actual harness selected and pinned through real integration evidence in [A19D](work-items/a19d.md).
This capability does not grant automatic PM source modification or general host access.

## Main cognition rebuild on 2026-10-18

D22: Rebuild main cognition as a typed LIDA-style cycle in [A19F](work-items/a19f.md), before the personal trial.
The old-generation CSM, codelet registry, pathways, and mini-appraisal are the reference design.
Old code can be copied.
What A19F builds becomes the canon. Old turn and frame records are not migrated.

D23: The inner loop moves to the shared cycle in [A19G](work-items/a19g.md), after A19F.

D24: A workspace shard is a separate global workspace.
One self shard holds self-referential content. Each entity and each workflow has its own shard.
A second companion is an entity like a human user.
Each shard has a 12k-token content budget and a separate 6k-token tail for its newest stream.
Content requirements reserve places and set caps, so important content stays and content types stay balanced.

D25: Every model call in main cognition is an unconscious process.
Prompt each call as an immersive character story and choice simulator.
Codelets express their single purpose at full strength, without softeners or concern about harshness.
Each codelet returns a list of output variants. The count starts at 1.

D26: Memory codelets run as pm-coder sessions with scoped memory tools. Other codelets run as two-call chains.
Quality comes before speed. More agent sessions are allowed where they improve believability.

D27: The mental state comes from `compute_state_delta` and the EMA baseline over feature deltas.
Appraisal results become features with affect data. `project_affect` is removed.

D28: Action selection decides zero or one action for each stimulus group, in character.
A separate unconscious routing step maps decisions to workflows and holds the capability rules.
Simulated dialogue for hard decisions is beta work.
Deliberate focus has a bounded weight in the CSM and the workspace. The companion cannot fully control its own mind.

D29: Code at main cognition boundaries uses pydantic models, not raw dicts.
A bug crashes the process. Only failures outside our control are retried: provider transport errors and invalid model output.
No persona name appears in code.
The whole cycle is one pm-workflows manifest, so later items can insert agent phases between code phases.
Model configuration does not change in A19F.

## Seed import evidence

Validate imported memories against the private ontology and capabilities at their attributed time.
Preserve valid history and explicit background with their provenance.
Unsupported sensory or emotional fiction cannot become an experienced fact through import.
Do not silently rewrite ambiguous history as a verified event.

## Implementation facts to resolve in A01–A02

The decision service needs a free endpoint, activation policy, and CPU/GPU allocation.
The user supplied chat on 8080, embeddings on 8081, and Laya on 8082 during A01.
All three advertised their configured model identities during the baseline capture.
Chat, embeddings, decisions, and Minecraft rendering must coexist on the user's machine.
Measure contention before choosing final model placement.

The actor needs a tested chat model and, for visual claims, a vision-capable connection.
The initial world needs its actual server/mod revision and spawn configuration recorded.
These are deployment facts, rather than new persona decisions.

## Out of alpha scope

This plan does not introduce automatic source rewriting as the default reflection mechanism.
It does not replace every generative stage with classification.
It does not copy all 342 old codelets into the live loop.
It does not add a second workflow engine or another canonical Ghost.

The alpha leaves these directions available for later evidence-based decisions.
