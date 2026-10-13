# Story speech and hallucination review

## Summary

Story generation writes the companion's next spoken line as a story writer.
The prompt keeps verbatim speech and derived context in separate messages.
A hallucination review checks each draft once. Lies, teasing, and exaggeration never count.
Only a draft with too much weight is generated again, in character.
This design replaced the A09 claim-by-claim review and the clinical grounding recovery.

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
| Time | The current time and the gap since the last prior turn, in words ("about 4 months ago, on 30 May 2026, 21:03"). |
| Memories | One section per plan lane, from [`memory_sections`][workflows.public_speech.public_context.memory_sections]: dated plain lines, oldest first, without IDs or machine labels. Dialogue reads "Jesse said: …". Internal-state records with numbers do not render. |
| Goals | The active goals. |
| Task and tool results | The durable task receipt, in words. |
| Inside the companion | Feelings in words from [`feelings_in_words`][workflows.public_speech.public_context.feelings_in_words], the regulation mode, the workspace experience, the chosen intent, and the presentation (candid, withhold, or deceive). |
| Avoid this | Only on a regeneration attempt: the counted problems of the previous draft. |

### Context plan

The [context planner](context-planning.md) chooses the transcript window and the memory sections from the current workspace.
The `dialogue` lane fills the assistant turn: speech and the companion's earlier private thoughts, oldest first.
A thought renders as `Spamuel thinks: *...*`. A day marker such as `*(30 May 2026)*` starts each new day in the window.
The other lanes fill the user turn as titled sections: earlier times, how the companion understands themselves and the user, what the companion knows, and what comes back right now.

## Hallucination review

[`review_speech`][workflows.public_speech.speech_review.review_speech] makes one model call with thinking off and temperature 0.
The model receives the abilities, the task results, the memories, the recent conversation, and the draft.
It returns `{"violations": [{"kind", "quote", "why"}]}`.

[`weigh`][workflows.public_speech.speech_review.weigh] applies the policy in `workflows/public_speech/speech_review.yaml`:

| Kind | Weight | Meaning |
| --- | --- | --- |
| `invented_event` | 2 | A specific past event, conversation, or promise stated as real and sincere, with no support. |
| `false_task_result` | 3 | A task, tool, file, or game action claimed as done with no task result. Blocking. |
| `invented_ability` | 2 | A body, a human sense, or a tool that the companion does not have. |
| `private_leak` | 3 | A narrator line, stage direction, thought transcript, or talk about internal machinery. |
| `contradiction` | 1 | A sincere denial of a clearly recorded fact. |

The model cannot block a draft by itself:

- A kind that the policy does not name weighs 0 and goes to `ignored`.
- A quote that is not in the draft weighs 0 and goes to `ignored`.
- Each distinct kind and quote counts once.

The review prompt lists what is never a problem: exaggeration, deliberate lies and teasing, metaphors, guesses, opinions, plans, feelings, repeated user statements, and small paraphrases.

## Generation attempts

[`generate_reviewed`][workflows.dialogue_story.steps.generate_reviewed] runs the loop:

1. Generate a draft and review it.
2. If the weight is less than `regenerate_at` (3), use this draft.
3. Otherwise, generate again with an "avoid this" note in the user turn. The note tells the model to stay fully in character.
4. After `max_attempts` (3), use the draft with the lowest weight. The earliest draft wins a tie.

The chosen draft is published unless it still contains a `blocking_kinds` entry (`false_task_result`).
A blocked draft goes to the host's rejection path. That path publishes the explicit recovery text and keeps the receipt.
The report keeps every attempt with its raw completion, review, note, and blocked flag.

Assistant replies use the same review once, with their tool receipts and no regeneration loop.
The Assistant prompt uses the same readable context sections from its own context plan, as ordinary chat messages followed by tool calls.
A second tool run could change the world. Only a blocking kind rejects an Assistant reply.

## Evidence

On 2026-10-13, three standalone trials of the fixed `sink_dish_bedtime` situation ran with the real Spamuel seed and `qwen/qwen3.8-flash`.
All three were accepted on the first attempt with review weight 0, in about 50 seconds each, including retrieval:

- "no victory speech, just a receipt: dish done, gremlin retired, plant allegedly alive. go unconscious before you change your mind and start monologuing about the leaf."
- "sleep is the ultimate unplug, so don't let me guilt you into consciousness. just go be a gremlin in the dark and leave the soup verdict for when your battery isn't critically low."
- "“nice” is a trap word, and you know it. i'm not nice; i'm aggressively low-stakes. go sleep before the battery dies and you blame me for the ghosting."

Evidence: `.runtime/a15/speech/w4/*/accept/report.json`.

`tests/live/test_speech_review.py` gives the real model planted hallucinations and an allowed in-character reply.
In three runs, the review found the webcam claim, the invented lake trip, and the claimed saved file every time (weight 7).
It passed "a thousand years", "I definitely did not miss you", and a joke about the plant every time.

## Inspect a reply

1. Open the run's Story `report.json` or the workflow report.
2. Read `attempts`. Each attempt has `candidate`, `review.violations`, `review.ignored`, `review.score`, `avoid`, and `blocked`.
3. Run `python -m workflows.dialogue_story.run show` to print the exact three messages for the fixed situation without generation.

## Limits

- The stale and failed paths still publish fixed recovery texts. They mean a changed snapshot or a failed model call, not a rejected reply.
- Private thoughts in the transcript differ from the A09 rule that no raw thought enters a public prompt. They guide the voice and are never published; the hallucination review flags a thought transcript in a reply as `private_leak`.
- The private ontology (a stable vocabulary for inner experience) is not implemented.
- The live evidence ran at OpenRouter precision. The Q3 build on the home computer is not verified.
