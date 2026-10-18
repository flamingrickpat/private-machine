# Story speech and the agentic review

## Summary

Story generation writes the companion's next spoken line as a story writer.
The prompt keeps verbatim speech and derived context in separate messages.
One agentic review judges the draft against the persona, the accepted state,
the abilities, the private ontology, and the task results.
It returns a judgment, a reason, and advisory findings.
Only a false task result blocks. There is no regeneration loop and no score.
This design replaced the A09 claim-by-claim review and the weighted hallucination policy.

Parent: [public dialogue](public-dialogue.md).
Entry points: [`workflows.dialogue_story.steps.run`][], [`workflows.dialogue_story.steps.build_story_messages`][], and [`workflows.public_speech.speech_review.review_speech`][].

## Why the design changed

The A09 path checked every sentence of a reply as a factual claim against record IDs.
In-character speech is full of inference, metaphor, and exaggeration, so most drafts failed.
A rejected draft went to a recovery prompt that told the model to "say the user reported" and to quote dated sources.
That produced replies such as "Jesse, you reported on October 13, 2026, that…".
When the recovery also failed, the user saw "I could not ground that reply in the available evidence. Please ask again."
In the A15 live gate, three of five turns ended in that text, and each turn spent 8 to 15 minutes in review.

The Story prompt also contained a JSON dump of the whole public context and a second block of evidence rules.
The voice therefore read like a careful witness instead of the character.

## Message layout

[`build_story_messages`][workflows.dialogue_story.steps.build_story_messages] returns three messages:

| Role | Content | Source |
| --- | --- | --- |
| system | The story-writer instruction | `prompts.yaml`: `system_prompt` |
| user | Character card, inner architecture, abilities, time, memories, goals, task results, current feelings, mind, intent, presentation, and an optional "avoid this" note | [`build_context_message`][workflows.dialogue_story.steps.build_context_message] |
| assistant | Prior dialogue exactly as spoken, the current user line, and the open prefill `Spamuel says: "` | The public dialogue records |

The rule is strict: verbatim prose goes only in the assistant turn, and derived prose goes only in the user turn.
The model continues its own transcript, so real dialogue anchors the voice.
Summary wording, memory labels, and instructions cannot leak into the style.
A memory that repeats a transcript line or the current message is left out of the user turn.

The final assistant message ends with the open quotation mark.
The model continues after it. [`clean_reply`][workflows.dialogue_story.steps.clean_reply] cuts the text at any invented next user turn.
A live check on 2026-10-13 confirmed that `qwen/qwen3.8-flash` on OpenRouter continues a trailing assistant message.

### The user turn

| Section | Content |
| --- | --- |
| Character card | The persona's story card, unchanged. |
| How the companion works on the inside | A plain description of the cognitive cycle, memory, and time away. It is adapted from the `pm_next` architecture text. |
| Abilities | Configured abilities, unavailable abilities, notes, and "no human body and no human senses". |
| Words for inner state | The private ontology terms, from [`term_lines`][workflows.public_speech.ontology.term_lines]. Each word comes from an implemented mechanism. The companion may use these words and should not invent a state word that is not here. |
| Time | The current time and the gap since the last prior turn, in words ("about 4 months ago, on 30 May 2026, 21:03"). |
| Awareness | The workspace shards of the cycle that chose the reply: the self shard and the partner's shard, in the words of the cycle's story prefix (`PublicContext.shard_story`). The verbatim lines of the tails are in the transcript instead. |
| Memories | The disclosed private state, and for a standalone situation one section per plan lane, from [`memory_sections`][workflows.public_speech.public_context.memory_sections]: dated plain lines, oldest first, without IDs or machine labels. Dialogue reads "<name> said: …". Internal-state records with numbers do not render. |
| Goals | The active goals. |
| Task and tool results | Worker state, effect verification, and the full durable task receipt. |
| Inside the companion | Feelings in words from [`feelings_in_words`][workflows.public_speech.public_context.feelings_in_words], the regulation mode, the subjective experience of the cycle, the in-character decision text as the intent, and the presentation (candid, withhold, or deceive). |

### Transcript and partner

The reply addresses the partner of the action request. All `{partner_name}` texts of the prompt name that entity.
A reply to a second companion therefore reads "<that companion> says: …" lines and is cut at that companion's next line.
The [shard context](context-planning.md#shard-context) fills the assistant turn: the dialogue in the tail of the partner shard and the companion's private thoughts in the self tail, oldest first.
A thought renders as `<companion> thinks: *...*`. A day marker such as `*(30 May 2026)*` starts each new day in the window.
A standalone situation has no shards. Its context plan fills the transcript and the memory sections.

## Agentic review

[`review_speech`][workflows.public_speech.speech_review.review_speech] makes one model call with thinking off and temperature 0.2.
The previous local gate returned empty text at zero temperature. The tested setting stays in use.
A retry repeats the same review when JSON or required fields are invalid. The reply is never regenerated.
The model receives the character card, the accepted feelings, the abilities, the private ontology terms and active bindings, the task results, the memories, the recent conversation, the chosen presentation, and the draft.
Tool results reach review in full. A 400-character excerpt previously removed valid inventory and world facts from a Minecraft observation.
The generator and the reviewer now receive the same tool result.
Assistant review also receives its prior spoken turns and current user message.
Task context retains the full durable receipt, including the original request and artifacts.
Worker completion and verified external effects remain separate facts.
It returns:

```json
{"judgment": "accept", "reason": "short reason", "false_task_result": false, "spoken_text": null,
 "findings": [{"kind": "invented_ability", "quote": "exact words", "why": "few words"}]}
```

The findings are advisory. The named kinds are `invented_ability`, `invented_event`, `unsupported_label`, and `private_leak`.
A finding stays in the receipt for review. It never blocks and never triggers another generation.

Only `false_task_result` is a hard block.
[`is_blocked`][workflows.public_speech.speech_review.is_blocked] reads the model's explicit `false_task_result` flag.
An invented kind cannot reject a reply.
The review prompt lists what is never a problem: exaggeration, a chosen lie or tease, a metaphor, a guess, an opinion, a plan, an accepted feeling, a repeated user statement, and a small paraphrase.
A prior deception intent explains a chosen false statement.
It cannot authorize a false task result. Actual task and tool receipts remain authoritative.
The parser requires a boolean `false_task_result` and a nonempty reason before it accepts the review object.

Ordinary cognitive stages also request `enable_thinking: false` through the shared LLM interface.
Native specialist sessions choose their own reasoning setting.
The [prompt logs](prompt-log.md) show the actual request settings and provider usage.

## One generation

[`generate_reviewed`][workflows.dialogue_story.steps.generate_reviewed] generates one draft and reviews it once.
It returns one attempt. There is no threshold, no note, and no second generation.
The accepted draft is published unless it holds a false task result.
A blocked draft reaches the host's rejection path. The host retains rejection telemetry and raises without public speech.
The report keeps the draft, its review, its reason, and its blocked flag.

Assistant replies use the same review once, with their tool receipts and no regeneration loop.
The Assistant prompt uses the same shard context: the transcript as ordinary chat messages, then the shard story, then tool calls.
Derived context has its own user turn. The final user turn carries the current message verbatim.
An earlier drink question previously displaced a task request that appeared only inside the derived context.
A second tool run could change the world. Only a false task result rejects an Assistant reply.
The durable management-claim guard still rejects an operation claim with no executed receipt.

## Evidence

The A09R delivery is recorded in [A09R verification](../work_items/verification/a09r-private-ontology.md).
Earlier, on 2026-10-13, three standalone trials of the fixed `sink_dish_bedtime` situation ran with the real Spamuel seed and `qwen/qwen3.8-flash`.
All three were accepted on the first attempt, in about 50 seconds each, including retrieval:

- "no victory speech, just a receipt: dish done, gremlin retired, plant allegedly alive. go unconscious before you change your mind and start monologuing about the leaf."
- "sleep is the ultimate unplug, so don't let me guilt you into consciousness. just go be a gremlin in the dark and leave the soup verdict for when your battery isn't critically low."
- "“nice” is a trap word, and you know it. i'm not nice; i'm aggressively low-stakes. go sleep before the battery dies and you blame me for the ghosting."

Evidence: `.runtime/a15/speech/w4/*/accept/report.json`.

`tests/live/test_speech_review.py` gives the real model planted hallucinations and an allowed in-character reply.
The check expects the webcam claim and the invented lake trip as findings and the claimed saved file as the one false task result.
It expects the in-character exaggeration and tease to pass.

## Inspect a reply

1. Open the run's Story `report.json` or the workflow report.
2. Read `attempts` and `speech_review`. The one attempt has `candidate` and `blocked`; the review has `judgment`, `reason`, `false_task_result`, and `findings`.
3. Read `review_reason` for the recorded reason of the accepted reply.
4. Run `python -m workflows.dialogue_story.run show` to print the exact three messages for the fixed situation without generation.

## Limits

- Changed evidence and failed model calls stop with a visible error. They never publish fixed recovery text.
- Private thoughts in the transcript differ from the A09 rule that no raw thought enters a public prompt. They guide the voice and are never published; the review flags a thought transcript in a reply as a `private_leak` finding.
- The private ontology is implemented as durable state in [the private ontology](private-ontology.md). It is advisory and grows only from implemented mechanisms.
- Current development uses the operator's OpenRouter config. Receipts retain the configured model and the actual provider response.

## Attribution and narration boundaries

The host resolves current speakers from accepted stimulus records.
Only user-origin dialogue enters the current user line.
Private thoughts, lifecycle events, and action intent remain context.
A proactive speaking action can have no current user line.
Several user messages in one batch retain their order.
New user speech can arrive while an unsolicited reply is in preparation.
That speech does not create an observation that the earlier action never made.
Story always receives an explicit task section, including an absence-of-observation statement when no receipt exists.
A connected game establishes availability; it does not establish position, health, inventory, or successful execution.
The review treats factual game status as an external result even when humor surrounds the claim.
Concrete inventory items need a supplied observation. A claim that inventory is empty also needs one.
An admission that inventory is unknown claims no particular game state and does not establish a task result.
Questions, refusals, and metaphors must not become invented observations through the reviewer's interpretation.
The [A18B verification](../work_items/verification/a18b-cognitive-evaluation.md) retains a real false block and its paired live regression.
The review writes a reason tied to actual evidence instead of copying a ready-made acceptance reason.

The existing review can select `spoken_text` when explicit narration surrounds actual speech.
The selected text must be a nonempty contiguous span of the draft.
[`workflows.public_speech.speech_review.select_spoken_text`][] rejects invented or paraphrased replacements.
The raw completion, review, and advisory finding remain in the report.
This selection adds no generation call or tool execution.
The review remains model-dependent; inspect retained findings during human review.
[`workflows.public_speech.speech_text.strip_narrator_setup`][] also removes the narrow detached setup form observed in the retained failures.
It preserves the original completion in the report.
