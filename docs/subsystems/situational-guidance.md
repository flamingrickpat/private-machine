# Situational guidance

## Purpose

A meaningful action depends on the current situation, not only the latest words.
A partner can receive a message without being able or willing to answer it.
An earlier action can already satisfy the purpose of another proposed message.
The character needs this context when it chooses any action, including private activity or waiting.

Parent: [cognitive pipeline](cognition.md).
Related: [public dialogue](public-dialogue.md) and [decisions and routing](decisions-and-routing.md).

## One projection of accepted evidence

[`workflows.public_speech.situational_guidance.SituationalGuidance`][] is a read-only view of the memory store.
It contains the runtime world time, configured capabilities, relevant partner evidence, active intentions, and recent selected actions.
Each source retains its record ID, revision, attribution, world time, and scope.
The projection does not create a second memory store or an availability flag.

The character card supplies identity and character traits.
Configured capabilities describe what the character can actually do.
Partner narratives and beliefs remain interpretations.
Actual public dialogue remains attributed interaction evidence.
An action's selected purpose remains distinct from a committed utterance and its delivery.
Delivery does not prove that the partner read the message.

## Data and execution

[`build_situational_guidance`][workflows.public_speech.situational_guidance.build_situational_guidance] reads an existing snapshot without retrieval or model calls.
The caller selects the partner and supplies runtime world time.
Entity-group decisions receive that entity's context.
Self and workflow decisions use the configured user under the existing action-target convention.
This convention does not introduce a fixed human-only interpretation of availability.

| Boundary | Information passed |
| --- | --- |
| Accepted records to projection | Attributed sources, revisions, scope, and selected purposes |
| Projection to action selection | Current partner evidence, intentions, and previous moves |
| Projection to public speech | Frozen situation sources with publication dependencies |
| Action execution to records | Actual results and delivery lifecycle |

[`workflows.main_cognition.story_prefix.StoryPrefix`][] includes the projection in the shared story context.
This path reaches decision calls even though those calls do not use the separate grounding turn.
Public dialogue freezes a fresh projection with its reply context.
The story renderer and the shared public-context renderer use the same guidance renderer.

The projection itself runs synchronously and performs no inference.
The surrounding main-cognition and public-dialogue workflows retain their own execution and acceptance boundaries.
This mechanism alone does not establish an independent fast interaction path.

## Reasoning about the partner

The prompt asks the model to infer the partner's ability to receive and respond from available evidence.
It distinguishes presence, activity, interruptibility, and willingness.
An attributed statement can support an inference without proving it.
Silence or elapsed time does not prove a replacement state.
A wake timer provides an occasion to reconsider, not an observation of the partner.

For example, a person can announce sleep after the companion already said goodnight.
The next timer presents that evidence and the previous move to action selection.
The character can then wait, think privately, or choose a justified interaction.
A new incoming message supplies new evidence and can change that choice.
The same reasoning applies to an AI peer that disconnects or a game character occupied elsewhere.

There is no separate dialogue-eligibility agent or conversational permission layer.
The evidence informs the character's existing choice mechanism.
It does not impose a mandatory silence rule or a standard agreeable personality.

## Inspection and limits

The frozen cycle request and public context retain the typed projection.
Rendered prompts show source IDs and revisions beside the original text.
Public context includes these sources in its existing publication dependencies.
The owner can compare the supplied evidence, selected purpose, generated speech, and actual delivery.

The current view selects bounded recent evidence without cutting its text.
It reports its limits: eight dialogue lines, four context sources, eight intentions, and four recent actions.
An important earlier statement can fall outside that window.
Persistent entity claims, freshness, corrections, and observer-relative knowledge remain separate beta work.
This projection must not be mistaken for completion of that entity-model target.

## Source map

- [`SituationalGuidance`][workflows.public_speech.situational_guidance.SituationalGuidance] defines the boundary contract.
- [`build_situational_guidance`][workflows.public_speech.situational_guidance.build_situational_guidance] selects accepted evidence.
- [`render_situational_guidance`][workflows.public_speech.situational_guidance.render_situational_guidance] renders evidence and generic reasoning guidance.
- [`StoryPrefix`][workflows.main_cognition.story_prefix.StoryPrefix] supplies main-cognition context.
- [`PublicContext`][workflows.public_speech.public_context.PublicContext] carries the frozen speech context.
