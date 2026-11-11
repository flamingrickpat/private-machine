# Ordinary interaction and background cognition

## Purpose and ownership

Ordinary conversation can use accepted character guidance while a full cognitive cycle calculates deeper changes.
The local choice still comes from the character's decision and routing mechanisms.
It can speak, wait, decline a direction, or ask for reconsideration.
Main cognition retains authority over intentions, appraisal, and consequential actions.
Critical input holds ordinary interaction until the required reconsideration is accepted.

Parent: [cognitive pipeline](cognition.md).
Related: [situational guidance](situational-guidance.md), [public dialogue](public-dialogue.md), and [publication](public-publication.md).

## Information flow

| Producer | Consumer | Information |
| --- | --- | --- |
| Early triage | Cognitive control and selection | Attributed input, significance, and critical epoch |
| Accepted main state | Ordinary selection | Intentions, feeling, workspace, and prior experience |
| Current records | Situation projection | Partner evidence, new input, capabilities, and previous moves |
| Ordinary selection | Public speech | Durable local decision and selected action |
| Ordinary selection | Main cognition | Consequential proposal or guidance mismatch |
| Main cognition | Future interaction | Accepted direction and appraisal |

[`workflows.interaction.host.Interaction`][] runs the ordinary selection workflow.
Its request combines accepted main guidance with current partner input and the shared situation projection.
The kernel reuses decision and routing without running appraisal, retrieval, attention, or goal revision for that local choice.
Prior experience remains labeled as prior context; an unexecuted phase supplies no fabricated result.
Before any accepted main guidance exists, initial input uses the full main path.

The route sees the available action space.
A reply uses the existing speech owner.
A quiet choice records a local outcome for its input.
A consequential action becomes a proposal for main cognition, rather than an immediate external effect.
Quiet selection creates no permanent silence rule or replacement intention.
Partner availability remains an inference from evidence, without a conversational permission agent.

## Concurrent execution

Main cognition continues to appraise ordinary input while interaction selects its local move.
Queued ordinary inputs can join the next main batch.
A new ordinary message does not restart the current calculation merely because it arrived later.
Selection ownership prevents the full cycle from also choosing a reply for the same ordinary group.
Acceptance checks that ownership again under the store lock, including receipts that arrived after the cycle snapshot.

Critical triage keeps priority over this overlap.
It interrupts obsolete calculation and speech through cooperative cancellation.
The atomic critical epoch and source guards still reject a late result if cancellation loses the race.
An already committed utterance remains an event.

[`pm.llm.concurrency.InferenceAdmission`][] schedules individual calls for each endpoint.
The initial policy retains one active stage-client call per endpoint in the PM process.
Triage has priority within the urgent lane, followed by interaction.
After four urgent grants, a waiting normal call receives a turn.
Ordinary cognition and private work share the normal lane.
Critical reservations admit triage and required cognition while unrelated work waits.

The quota limits admission starvation outside critical reconsideration.
It does not guarantee wall-clock latency from an active provider call.
OpenRouter owns the inference hardware; a Python thread alone does not prove responsiveness.
Admission logs record roles, queue time, and active time.
Native pm-coder sessions use separate transports and are not covered by a global provider-wide reservation.
Their workload and cost remain part of live verification.

## Evidence and acceptance

[`workflows.public_speech.dependencies.WorkDependencies`][] records source meanings, observed partner input, scopes, governing tick, and critical epoch.
The governing tick records provenance; it is not a global freshness veto.
A new message from another partner does not invalidate this exchange.
New input from the same partner refreshes the interaction context.
Older selected input whose reply never committed remains part of that refreshed conversation.

[`pm.memory.dependencies.SourceDependency`][] distinguishes source meaning from operational revision changes.
Exchange resolution, delivery, and execution bookkeeping can change a revision without changing the supplied evidence.
An actual correction, scope change, or replacement of required current authority still rejects the proposal.
Historical evidence can remain historical without becoming current authority.
Exact status-dependent operations retain strict guards.

Intention and focus changes still require their owning workflow's acceptance checks.
An inner-workspace turn does not replace the accepted main guidance parent.
The first ordinary-interaction slice leaves private episode continuation conservative after new external input.
The background main cycle and sustained private inquiry therefore have different continuation limits.

## Durable selection and publication

An ordinary selection receipt and its selected reply action are append-only recovery records.
They create no main GWT turn and claim no new appraisal.
The canonical memory store retains their exact input and accepted guidance dependencies.
Restart can replay selected speech without paying for another decision.
A quiet receipt prevents restart from manufacturing a reply.

The speech owner freezes current evidence and relevant later partner lines.
It commits the utterance and covered exchange resolution atomically.
Competing actions cannot both answer the same resolved input.
An explicit terminal-task reply can follow an interim answer because the actual task outcome is new evidence.
Generic recovery text does not grant that exception.

The existing record packages expose conversation, selected moves, accepted cycles, and their supporting sources.
The owner can inspect the guidance and critical epoch used for a reply beside its actual delivery.
Model and human quality ratings remain separate from these engineering records.

## Source map

- [`Interaction`][workflows.interaction.host.Interaction] owns ordinary selection and replay.
- [`select_interaction`][workflows.interaction.steps.select_interaction] reuses the character decision and route.
- [`WorkDependencies`][workflows.public_speech.dependencies.WorkDependencies] carries the observed frontier.
- [`SourceDependency`][pm.memory.dependencies.SourceDependency] defines source meaning and current-authority requirements.
- [`InferenceAdmission`][pm.llm.concurrency.InferenceAdmission] owns per-call scheduling.
- [`commit_public_reply`][workflows.public_speech.publication.commit_public_reply] owns durable speech acceptance.
