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
