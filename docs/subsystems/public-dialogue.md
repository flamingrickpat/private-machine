# Public dialogue

## Summary

Story and Assistant share one accepted identity, affect, goals, evidence, and disclosure contract.
Each candidate receives time interpretation and evidence review before publication.
The Ghost commits public speech and exchange resolution together.
Rejected, stale, or failed work produces an observable recovery outcome.

Parent: [cognition](cognition.md).
Related: [experienced memory](experienced-memory.md), [publication and recovery](public-publication.md), and [A09 verification](../work_items/verification/a09-public-dialogue.md).
Design: [private ontology](../work_items/alpha-plan/ontology.md).

## Context and entry points

[`workflows.public_speech.public_context.PublicContext`][] defines the common contract in the root repository.
[`workflows.public_speech.public_context.build_public_context`][] freezes accepted state before generation.
[`modules.StoryDialogue`][] and [`modules.AssistantDialogue`][] use this same builder.
The standalone Story planner also uses this contract.

The context contains character identity, accepted affect, regulation, active goals, attributed memories, and the selected workspace.
It retains the selected action, source IDs, source revisions, UTC timestamps, scope, and provenance.
Current stimuli and original records from selected source chains accompany the selected summaries.
The three latest original startup events supply restart chronology beside retained facts.
They carry `lifecycle_event=startup` and source revision guards.
The context also contains domain capabilities, applicable ontology records, presentation intent, and an optional durable task receipt.
Every evidence record has an `evidence_role`.
Original user reports and actual runtime events can support event claims.
Companion utterances support claims about speech.
Derived summaries require their original source records.
Action intent cannot establish execution.

The main loop rewrites a user message into a short memory search query through the normal chat model.
The rewrite removes recall boilerplate and retains concrete topics and outcomes.
The retrieval trace retains both the original message and the search query.
The rewrite also selects `event`, `utterance`, or `general` evidence use.
Event retrieval excludes companion utterances, intentions, imagined events, and interpretations.
For event recall, the public builder also omits unselected companion utterances from expanded source chains.
Their source IDs and revision guards remain available, but their prose cannot compete with the selected original reports.
Explicit utterance recall can still select that speech and retain its attribution.
Derived event memories need original user reports, runtime events, or world observations.
The trace retains excluded IDs under `speech_only_excluded_ids`.
Retrieval reserves eight of 24 slots for imported episodes and original user reports.
This reserve prevents recent recall questions and generated answers from consuming the whole older-history pool.

The public prompt retains four recent dialogue turns, with at most 240 characters per turn.
The canonical records remain complete.
This bound gives original evidence space beside conversational voice examples.
Current input IDs are excluded from that prior transcript.
The current message appears once, rather than inventing a repeated user turn.
Both builders repeat original reports at the answer boundary after the voice context.
The block includes each record's timestamp and the actual current clock.
Selected reports form a partial snapshot, rather than an exhaustive record of every event.
That block preserves exact report text, actor attribution, timestamps, and uncertainty.
Imported history carries an explicit historical flag, even when its record status is active.
A past report of a plant, room, or game condition does not establish its condition today.
Current physical claims require a current user report or observation.
Stable names, preferences, and plans remain usable until later evidence changes them.
For technical advice, missing object details or verified sources require an uncertainty statement or a question.
The selected action cannot supply missing treatment evidence.

The source render contains at most 48 records.
Each memory excerpt contains at most 1,200 characters.
Current stimuli, known deception facts, and selected memories precede other source-chain records.
Omitted IDs remain visible in the diagnostic context.
An omitted source cannot establish a claim through an invented quotation.

Raw recent thoughts no longer become Story's `feelings` field.
Accepted affect values supply that field.
The selected workspace guides speech but does not grant raw thought disclosure.

## Two generation paths

Story preserves five message roles: system, user, assistant, user, assistant.
The last message retains the companion prefill.
The history provides voice examples and conversational continuity.
Earlier generated speech cannot independently establish its described event.

Assistant preserves normal user, assistant, and tool messages.
Its tool loop retains execution results, denied calls, errors, and transport attribution.
The common context supplies the selected action and accepted emotional state beside those receipts.
The [direct Minecraft inventory route](minecraft-actor-controls.md#direct-inventory-dialogue) uses one coordinated observation and starts no managed task.
The existing management guard still rejects an operation claim without execution support.

Both prompt resources require public speech.
Both public generators disable hidden reasoning for their direct speaking turn.
The cognitive stages retain their configured reasoning behavior.
They exclude narrator lines, stage directions, and raw thought transcripts.
The companion can disagree, refuse, tease, express affection, or choose privacy.
The prompts do not require a repeated disclaimer about the companion's nature.

## Domain capabilities and private ontology

[`workflows.minecraft_body.capabilities.capability_snapshot`][] reads configured workflows and current Minecraft transport availability.
It discovers MCP tools with a five-second deadline and probes the game socket with a two-second deadline.
This operation does not move the character or reset the world.

`minecraft.status` is `connected`, `disconnected`, or `unknown`.
`unknown` applies to callers that supply no live provider.
Tool discovery proves availability, rather than a successful world action.
Game movement differs from a human body.
The current public contract declares no human smell, taste, hearing, touch, or sight.
`persistent_memory=true` describes the actual retained Ghost store.
Missing human senses do not imply missing memory.
`visual_interpretation` remains false without interpreted pixel evidence.
An image path alone cannot grant vision.

Implemented emotion names come from `StateEmotions`.
Accepted values and relevant existing ontology terms and bindings accompany the prompt.
The host retains at most 24 ontology records in this context.
An imported term cannot establish a new sensory mechanism.
Metaphor and compound emotion descriptions remain permitted with clear attribution.

All seven cognitive generating stages receive the same domain and ontology boundary.
The cognitive affect projection still supplies each stage's prospective state separately.
The production action path uses domain evidence instead of the older global physical-verb guard.
Legacy direct requests retain their compatibility guard until callers supply the common contract.

## Disclosure and deception

[`workflows.public_speech.public_context.Presentation`][] accepts `candid`, `withhold`, and `deceive`.
The cognitive action commits this choice before dialogue starts.
Each choice applies to its selected public action.
An older request for one teasing answer does not become an ongoing preference to falsify later recall.
Context selection, goal management, and action selection retain the exact scope of the original request.
`withhold` retains the companion's purpose and permitted summary.
It differs from failed generation or rejected publication.

`deceive` requires a prior purpose and retained `known_fact_ids`.
The runtime rejects an incomplete deception intent before generation.
[`workflows.main_cognition.experience.utterance_metadata`][] links the resulting utterance to that prior action.
Adding an intent after generation cannot justify the earlier prose.
Deception cannot waive capability, task verification, or private-disclosure checks.
Public speech remains an attributed communicative event and cannot automatically establish a factual belief.

## Time interpretation and claim review

[`workflows.public_speech.claim_review.review_public_claims`][] uses the normal LLM at temperature 0.1.
All three review phases enable reasoning for attribution, exact spans, paraphrases, and chronology.
Each phase runs once.
The first phase receives only the candidate and extracts literal plant, room, or game conditions with their time scope.
It retains exact text spans and rejects invented spans.
Complete clauses retain quoted attribution, negation, and uncertainty.
The candidate limit is 12,000 characters.
Oversized or malformed input raises an explicit protocol error.
This phase excludes capabilities, task verification, emotions, names, preferences, and plans.
It cannot reinterpret a present assertion as a past report after reading old evidence.
The second phase receives these conditions, original evidence, accepted state, and execution receipts.
It must retain each extracted condition as a claim.
If that review accepts supported historical claims, a third phase checks only those claims and their cited original text.
It receives no character, workspace, or prior dialogue transcript.
Speaker and addressee names resolve pronouns in the candidate.
Each source author resolves first-person pronouns in that original report.
Original startup events and the current user report supply chronology and request scope.
The current question cannot establish a missing external fact.
An original source in accepted memory supports remembering that report or having it on file.
That wording does not claim a literal file write or a complete memory inventory.
An unknown antecedent does not qualify an asserted conditional technical rule.
A definite treatment effect or another person's intent still requires evidence.
A partial snapshot cannot establish an exhaustive count of canonical memory records.
The supplied UTC clock supports relative dates from source timestamps.
Faithful paraphrases and ordinary conversational acknowledgments do not require identical source wording.
A prior joke request cannot authorize a lie in the current exchange.
Derived summaries resolve through their visible original source records.
Generated utterances can establish their recorded words when the claim concerns speech.
It checks the topic, actor, action, object, and relation separately from the broader conversational interpretation.
Adjacent sentences cannot establish an action-object relation or causality.
For example, a coffee report beside a plant update does not establish that coffee was given to the plant.
Each citation check retains its claim index, support decision, and reason in `citation_checks`.
The limit is 32 claims; missing, repeated, or ungranted indices raise a protocol error.
Unsupported citation checks reject the candidate.
This conditional phase does not repeat a review until it passes.
The first live review stalled with temperature zero and hidden reasoning enabled.
The retained failure and the direct repair probe are part of the A09 evidence.
The host also validates citation roles after semantic review.
The request uses the review model's JSON schema to constrain protocol fields and enum values.
Malformed transport or schema output still triggers explicit recovery.
A supported historical event requires an original report or an actual event citation.
For a cited abstraction, the host follows its rendered source chain and appends visible original event IDs.
A cited earlier companion answer cannot satisfy that condition.
Speech acts use `evidence_use=utterance`; their recorded text supports what was said.
The `public_complete` tool is an explicit grant in both dialogue manifests.
The reviewer receives the candidate, accepted evidence and state, and executed tool receipts.
Its copy of the common context omits character prose, workspace text, selected-action prose, and duplicate memory excerpts.
Those fields guide voice but cannot establish an external event.
The review retains identity, affect, goals, capabilities, original evidence, ontology, and the prior presentation choice.
Pure greetings, thanks, apologies, and acknowledgments do not need factual citations.
A plausible interpretation does not establish a fact.
For example, reporting a wrong-file attachment does not establish that the reporter attached that file.

Each claim identifies its text, domain, status, evidence IDs, and reason.
Each claim also has `verification_claim`.
`time_scope` accepts `past`, `current`, `stable`, or `unspecified`.
`external_physical_state` identifies literal conditions that need fresh evidence.
The host retains the first phase's time interpretation for matching claims.
A supported current physical condition requires a nonhistorical original report or observation.
`tool_receipt_ids` cite invocation-local `tool_call_id` strings separately from numeric Ghost `evidence_ids`.
A successful executed `minecraft_observe` receipt can establish its current Minecraft snapshot.
Absent, denied, and failed receipts cannot supply that condition.
Historical claims still require their original numeric source chain.
Omitting a current extracted condition rejects the review.
This flag identifies a current executed operation, verified task state, or actual current file or world effect.
Conversational naming, metaphor, future speech, and imported historical reports do not establish those effects.
The constrained response requires every claim field, including this flag.
Its default is conservative for older captured protocol records.
Domains include history, tasks, human senses, Minecraft, emotions, privacy, and other claims.
Statuses distinguish supported facts, metaphor, uncertainty, intentional deception, and unsupported certainty.
Unknown source IDs and unsupported claims reject the whole utterance.
An external claim in the `other` domain cannot pass as supported when it has no source or state evidence.
General model knowledge does not independently verify that claim.
Every supported positive historical assertion requires a supplied citation, including assertions labeled as knowledge state.
An absent report cannot support a claim that the companion has that report.
The generator must distinguish supplied evidence from uncertain technical possibilities.
A report of a neighbor's advice does not establish that the advice is correct.
Prior authorized deception can apply to a teasing name even when the model uses the coarse `task` domain label.
It cannot apply to a claim that current task execution or file verification occurred.
Malformed review output fails the workflow and triggers publication recovery.
The review also classifies `public_form` as `speech`, `narration`, `private_transcript`, or `empty`.
Only `speech` can pass.
This check applies beside factual claims, because a stage direction can avoid making an explicit factual claim.

The reviewer does not score persona quality or rewrite prose.
A rejected candidate remains in its report beside the source comparison.
No repeated quality-score loop rewrites a valid reply until it passes.
Model review remains imperfect and requires independent source checks in live acceptance.
Human immersion ratings remain with the operator.
After a semantic rejection, the module can invoke [bounded grounding recovery](public-publication.md#bounded-grounding-recovery).
That new response has its own review and a distinct recovery outcome.

## Inspect a real exchange

Run the [A09 console gate](../work_items/verification/a09-public-dialogue.md).
Read the correlated public request, workflow report, and `claim_review`.
Compare every historical claim with the supplied original evidence IDs.
Inspect the public Feature's `factual_promotion`, `action_schema_id`, and presentation attribution.
For publication failures, inspect the recovery receipt and durable exchange marker.

Go deeper: [publication and recovery](public-publication.md), [managed work](managed-work.md), and [memory consolidation](memory-consolidation.md).
