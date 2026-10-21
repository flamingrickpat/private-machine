# Public dialogue

## Summary

Story and Assistant share one accepted identity, affect, goals, evidence, and disclosure contract.
Each candidate receives time interpretation and evidence review before publication.
The Ghost commits public speech and exchange resolution together.
Exhausted model-output repair retains its evidence and leaves the exchange pending.
Programming and filesystem errors still reach application error handling.

Parent: [cognition](cognition.md).
Related: [experienced memory](experienced-memory.md), [publication and recovery](public-publication.md), [the private ontology](private-ontology.md), and [A09 verification](../work_items/verification/a09-public-dialogue.md).
Design: [private ontology](../work_items/alpha-plan/ontology.md).

## Context and entry points

[`workflows.public_speech.public_context.PublicContext`][] defines the common contract in the root repository.
[`workflows.public_speech.public_context.build_public_context`][] freezes accepted state before generation.
[`workflows.dialogue_story.host.StoryDialogue`][] and [`workflows.dialogue_assistant.host.AssistantDialogue`][] use this same builder.
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

A reply that main cognition chose reads the workspace shards of its cycle ([shard context](context-planning.md#shard-context)).
The reply addresses the partner of the action request: the user, or another entity such as a second companion.
The prompt names the partner, and the transcript is the dialogue with that partner.
The [context planner](context-planning.md) selects the dialogue window and the memory lanes only for the standalone Story situation.
The canonical records remain complete.
This bound gives original evidence space beside conversational voice examples.
Current input IDs are excluded from that prior transcript.
Only actual dialogue of the user or another agent appears as the current message ([`partner_stimuli`][workflows.public_speech.public_dialogue.PublicDialogue.partner_stimuli]). Private and system events keep their attribution.
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

The evidence holds the shard records or the planned lane records.
Without either, it holds the stimuli, the startup events, and the disclosed private state.
Current stimuli, known deception facts, and startup events always precede the shard or planned records.
The revisions of planned records and of their sources guard the commit.

Raw recent thoughts no longer become Story's `feelings` field.
Accepted affect values supply that field.
The selected workspace guides speech but does not grant raw thought disclosure.

## Two generation paths

Story uses three messages: a story-writer system message, a user message with all derived context, and an assistant message with only verbatim dialogue and the open companion prefill.
[Story speech](story-speech.md) describes the layout and the one agentic review.
Earlier generated speech cannot independently establish its described event.

Assistant preserves normal user, assistant, and tool messages.
Its derived context has a separate turn. The final user turn contains the current message verbatim.
The review receives that message and the same spoken history as generation.
An automatic terminal reply reports existing work and receives only tools marked `read_only`.
It can inspect status or fresh game state, but it cannot repeat the original task through a new management call.
New user actions retain the full management grant.
[`reply_tools`][workflows.dialogue_assistant.tools.reply_tools] selects that grant before generation.
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
[The private ontology](private-ontology.md) writes durable terms and bindings for those implemented states.
The host observes the ontology before it freezes the public context.
Accepted values and the active ontology terms and bindings accompany the prompt.
The host retains at most 24 ontology records in this context.
An imported term cannot establish a new sensory mechanism.
Metaphor and compound emotion descriptions remain permitted with clear attribution.

The generating cognitive stages receive the same domain and ontology boundary. The codelets and attention scoring calls omit it; see [codelets](codelets.md).
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

## Agentic review

Both public paths review a draft once with [`workflows.public_speech.speech_review.review_speech`][].
The review is one model call with a persona-aware prompt.
It judges the draft against the accepted state, the character card, the abilities, the private ontology, and the task results.
It returns a short `judgment`, a `reason`, and advisory `findings`.
Its passing judgments are `accept`, `good_enough`, and `equivalent_or_better`.
An incidental discrepancy can pass with its reason and finding intact.
[Review materiality](review-materiality.md) explains practical examples and the evidence boundary.
Findings name an invented ability, an invented event, an unsupported label, or a private leak.
A finding stays in the receipt for review. It never blocks a reply and never triggers regeneration.

Only a material false task result is a hard block.
A false task result can change what the user does with a real object.
Python decides the block from the model's explicit `false_task_result` flag, so an invented kind cannot reject a reply.
A chosen lie, a tease, an exaggeration, or a metaphor is not a false task result.
The accepted reply records `review.reason` beside its durable utterance and in the published reply.
The A09 claim-by-claim review and its grounding recovery were removed on 2026-10-13.
The [A09 verification](../work_items/verification/a09-public-dialogue.md) keeps their historical evidence.
[A09R](../work_items/verification/a09r-private-ontology.md) replaced the weighted hallucination policy with this review.

## Model-output failures

The LLM boundary distinguishes unusable external output from faults in the application.
[`EmptyCompletionError`][pm.llm.http_client.EmptyCompletionError] means that the provider returned no usable text or tool call.
Whitespace-only content also counts as empty.
A completion with tool calls remains valid even when it has no text.
The shared text client retries declared empty output once.
The tool-completion client permits two empty-output retries without repeating any tool execution.
Other model errors do not enter these empty-output retries.
HTTP transport retries keep their existing client policy.

Story permits one [empty-output recovery](story-speech.md#completion-echoes) with the original context.
Review syntax, findings, and speech-span errors enter the existing context-bounded JSON repair loop.
The draft stays unchanged during review repair.
The standalone Story adapter binds context measurement to the same active client as generation.
Its reviews can therefore repair invalid JSON or speech spans without a missing-measurement error.
Assistant stores completed tool receipts before requesting more model output.
An empty Assistant result cannot cause those tools to run again.

[`PublicDialogue.defer_reply`][workflows.public_speech.public_dialogue.PublicDialogue.defer_reply] handles only declared model-output exhaustion:

- `EmptyCompletionError`: the empty-completion retries ended without usable output.
- [`ProviderCompletionError`][pm.llm.http_client.ProviderCompletionError]: the provider failed or filtered generation. Partial text never reaches publication.
- [`SpeechOutputError`][workflows.public_speech.speech_text.SpeechOutputError]: the speech workflow exhausted its usable-output attempts.
- [`ModelFormatError`][pm.llm.structured.ModelFormatError]: JSON repair exhausted the available input context.

The host writes `<runtime>/<module>/<invocation>/speech-failure.json` before it publishes a `generation_failed` action receipt.
This file retains the request and scratch evidence, including Story attempts or completed Assistant tool receipts.
No public reply commits. The exchange stays pending for a later action or restart recovery.
The module continues with the next mailbox action and releases its inference reservation.
The host does not publish an invented explanation as companion speech.

This boundary does not catch ordinary `TypeError`, `KeyError`, `ValueError`, or filesystem errors.
Faulty source lookup, broken callbacks, and failed evidence writes still stop the application with their original tracebacks.
Cancellation also retains its existing signal boundary.
A valid review that reports a false task result keeps the existing rejection policy.

Focused checks:

```powershell
uv run --locked python -m unittest tests.unit.pm.llm.test_empty_completions tests.unit.workflows.public_speech.test_speech_recovery tests.unit.workflows.public_speech.test_speech_review -b
uv run --locked python -m unittest tests.live.test_runtime_stability -v -b
```

The live checks use the synthetic Spamuel corpus and the explicit local model connection.
They read no private character home or user configuration.
They keep their prompts under their own fresh `.runtime/stability/` directory.

## Inspect a real exchange

Run the [A09 console gate](../work_items/verification/a09-public-dialogue.md).
Read the correlated public request and the workflow report.
For Story, read `attempts` and `speech_review`: the attempt holds the draft and its one review.
Inspect the public Feature's `review_reason`, `review_judgment`, `factual_promotion`, `action_schema_id`, and presentation attribution.
For publication failures, inspect the recovery receipt and durable exchange marker.

Go deeper: [publication and recovery](public-publication.md), [managed work](managed-work.md), and [memory consolidation](memory-consolidation.md).
