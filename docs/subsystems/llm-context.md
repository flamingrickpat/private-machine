# Python chat and agent context

## Summary

Laya and chat have separate context limits.
Direct Python chat sends one supplied request, while pm-coder owns agent history and automatic compaction.
Real-endpoint checks verify direct calls beyond Laya's input limit.
A04 also verifies actual server-context overflow and live agent recovery with decisive classification evidence intact.

Parent: [model connections](../models.md).
Policy: [alpha model selection](../work_items/alpha-plan/model-policy.md).

## Separate service limits

Chat, embeddings, and typed decisions are separate services with separate limits.
The verified local chat endpoint on 8080 advertises 131,072 serving tokens.
Laya on 8082 advertises 8192 state-context tokens and a separate 192-token decision head.
Laya's head can truncate questions and options.
That mechanism does not truncate the chat service's questions or disable pm-coder compaction.

[`pm.llm.activation.ChatClient.complete`][] sends one supplied message list and returns the full completion.
It does not own conversation history or run automatic agent compaction.
[`pm.llm.stage_model.LazyStageModel`][] exposes that path to synchronous and asynchronous Python roles.
The caller constructs context from relevant memories, selected workspace evidence, and the current request.
Structured stages use their actual parser and schema rather than treating every raw completion as an accepted result.
The cognitive parser accepts a JSON object inside a Markdown code block.

Agentic work uses pm-coder's `run_turn` loop.
Its settings discover the serving context unless the caller explicitly selects a budget.
Context exhaustion triggers a live summary, one execution checkpoint, and a bounded recent tail.
The original request remains exact, and the agent resumes from retained work.
A single oversized initial request with no execution history requires input selection rather than agent-history compaction.
Long-file classification uses scoped agent reads and preserves required conclusions through compaction.

## Direct project decisions use token percentages

[`pm.llm.context_window.measure_context`][] measures the selected connection before each Minecraft project decision.
The serving window comes from configured `context_window`, OpenRouter's selected model catalog entry, or a compatible endpoint's current `/props` response.
Training-context metadata cannot establish the serving window.
`context_input_fraction` defaults to `0.8` and accepts a number strictly between zero and one.
The input budget equals that fraction of the serving window, in tokens.
The remainder reserves room for the response; it does not impose a generation cap.

Compatible servers apply their actual chat template and tokenize the resulting prompt.
Receipts label this exact measurement `server.chat_template_and_tokenizer`.
Servers without the template route use a labelled tiktoken estimate.
Their serving capacity still comes from `/props` when available.
Other providers without that discovery route require configured `context_window`.
OpenRouter uses its model catalog and a labelled estimate without requests to local tokenizer routes.
The estimate includes chat framing but cannot promise the provider's exact token count.
The provider remains authoritative if its tokenizer differs.
Transport errors, invalid measurement responses, and cancellation retain explicit failure boundaries.

The project selects decision context before measurement, regardless of the serving capacity.
The original input remains archived with its digest and stays available through evidence retrieval.
An oversized selected input fails in tokens with its serving capacity and percentage in the error.
There is no character limit.
The same policy scales with 98,304, 186,624, and 1,000,000 serving tokens.
Native pm-coder history still uses its own serving-context discovery and automatic compaction.

## Shared object rendering

[`pm.context_render.render_context`][] renders selected data as [`pm.context_render.turns.Content`][].
Content contains ordered text and media parts. It has no conversation role or provider fields.
[`pm.context_render.turns.Turn`][] adds a role and optional tool protocol fields.
The renderer accepts dictionaries, dataclasses, and Pydantic models. It does not change the input object.
Type owners register specialized renderers through `render_context.register`.
The `PublicContext` adapter belongs to `workflows.public_speech`; the runtime does not import workflow types.

`ContextKind` selects facts, emotion, dialogue, memories, and ontology where a typed adapter supports those sections.
The default schema limits emotional display values to two decimal places.
Factual rendering without `ContextKind.Emotion` preserves measurement precision.
Caller-owned JSON files define field selection, omission rules, precision, exact fields, vectors, aliases, and table columns.
The renderer preserves row order. It keeps zero and false values unless a caller explicitly excludes them.
Unsupported objects, invalid numeric values, and unavailable schema files raise errors before inference.

Record deduplication defaults to `dedupe=True` within one render call.
An identical repeated record becomes `record_ref` with its original ID.
Records with different contents remain separate. Once an ID has multiple versions, later instances retain their full contents.
Separate render calls do not share references. Prompt-local IDs are not used.
Selected attachments remain typed media; repeated identical records do not repeat their attachments.
Formatting notes appear once when fragments are composed into the same turn.

The cognitive stages use this renderer for structured input and public context.
Ontology rows retain definitions and usage rules while excluding storage metadata.
Evidence retains source attribution, scope, and original IDs.
The current stage affect takes precedence over the earlier public-context affect.
Matching evidence already present in the current stimuli is excluded from the additional grounding turn.
The complete stored records and state revisions remain unchanged.

Minecraft project decisions use `workflows/minecraft_project/resources/llm_context.json`.
The renderer omits receipt hashes, duplicate hotbar data, display names, and empty optional fields.
Displayed world values use one decimal place. Displayed timestamps use whole seconds.
Explicit radii and count thresholds retain their original precision.
The runtime retains exact evidence, digests, and certificate checks outside the rendered prompt.
The output schema appears once through `response_format`; the system message does not repeat it.

The native planner receives a request overview with recovery state, recent diagnoses, and failed verification.
It reads task specifications through `task_get` and current world facts through `world_read`.
Full historical methods and survey outputs remain available through `evidence_read`.
Context selection determines relevance. Rendering controls the representation of that selected context.
The earlier `render_object_for_llm` function remains a text-only compatibility wrapper.
It raises an error if converting its result would discard media.

## Typed turns and media

Compose content before assigning its role:

```python
from pm.context_render import Content, ContextKind, Role, Turn, render_context

turns = [
    Turn(Role.System, "Describe the supplied evidence."),
    Turn(Role.User, Content("What does this record show?\n") + render_context(
        record, ContextKind.Dialogue | ContextKind.Memories,
    )),
]
result = model.complete_result(turns, enable_thinking=True)
answer = result.text
reasoning = result.reasoning
```

[`pm.context_render.turns.Media`][] holds an explicit modality, MIME type, and bytes or a remote URL.
`Media.from_file` captures file bytes once. `Media.from_base64` decodes supplied bytes once.
The renderer does not read files or download attachments.
String conversion raises an error when content includes media.
Record storage can supply these typed attachments; this change does not add a media persistence format.

[`pm.llm.prompt_encoding.OpenAIChatEncoder`][] creates the wire payload at the provider boundary.
It encodes images, WAV or MP3 audio, and file bytes using the documented OpenAI chat content fields.
See the [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
[`pm.context_render.turns.VideoFrames`][] carries sampled images with explicit offsets in seconds.
Sampling belongs to the media owner. The current adapter sends these frames as timestamped images.
Raw video and unsupported media formats raise errors before the HTTP request.
Media currently belongs on user turns in this adapter.

[`pm.llm.prompt_encoding.PromptEncoder`][] and [`pm.llm.completion.CompletionBackend`][] define the extension boundaries.
Anthropic and direct tokenizer backends can consume the same neutral turns through their own adapters.
Those adapters are not implemented here. Legacy message dictionaries remain accepted at the current boundary during migration.
Token estimates for media count text only and report `measurement_complete=False`.
They do not count base64 bytes as language tokens or claim that unknown media costs fit the context window.

## Completion and reasoning

[`pm.llm.completion.Completion`][] separates final content, tool calls, reasoning, usage, and finish status.
`Completion.text` returns final text. It never substitutes reasoning for an absent answer.
The text-only stage helpers return final text, while `LazyStageModel.complete_result` exposes the complete typed result.
Tool completions preserve provider state needed to replay signed reasoning with the assistant turn.
Raw provider receipts retain the original response under ignored runtime paths.

Thinking remains a per-call option through `enable_thinking=True`.
The provider translation uses its existing local-template or OpenRouter reasoning fields.
The cognitive stage runner accepts `generation_overrides`, keyed by stage name.
An override can enable thinking for one decision without changing other stages.
Overrides cannot replace model routing, streaming, messages, or the result schema.
Native Minecraft planner and specialist sessions already enable thinking.
Research and extended tool use remain suitable for pm-coder sessions.

## Ad hoc verification

The typed-context checks are retained under `.runtime/adhoc/typed-context-20261015/`.
Focused runtime, cognitive-stage, dialogue, and Minecraft checks passed.
A real Qwen endpoint identified a generated red image with thinking enabled.
That request returned final text and separate reasoning through the typed completion path.
Capture previews reduced the estimated Workspace input by 14% and Goals input by 57%.
These previews measure representation size; they do not establish behavioral quality or exact provider token savings.
The full foundation suites are deferred at the owner's request until A16 finishes.

## Historical context receipts

The earlier A13 planner rejected 130,147 characters before contacting the model.
The previous llama.cpp server tokenized that retained, templated request as 50,160 tokens.
It advertised 131,072 serving tokens, so the old character limit caused an unnecessary rejection.
Older context-size receipts remain historical evidence of that application defect.

The earlier local chat deployment used Strata and Qwen3.8-Flash-Next-Uncensored.
It advertises 262,144 serving tokens through `/props`.
It does not expose `/apply-template`, so direct project receipts label their token counts as estimates.
The default input budget is 209,715 tokens.
Strata reports actual prompt tokens through `/status`; those observations can differ from the application estimate.

Development now selects `config.alpha.openrouter.yaml` explicitly.
The workflows receive the same model interface for every endpoint.
Only `pm.llm` translates provider fields and discovers provider context limits.
The [live prompt logs](prompt-log.md) expose the actual requests and provider token usage.

## Verified direct Python calls

Run the retained gate against the actual chat service:

```powershell
uv run --locked python dev/work_items/alpha_llm_context_gate.py --config config.alpha.example.yaml --out .runtime/alpha/llm-context-repeat
```

The gate reads the unchanged real Spamuel seed and captured A01 completed-task report.
It checks the run state and empty answer field against their actual source values.
The history is an exact prefix selected through the chat tokenizer, rather than invented companion history.
The command writes requests, replies, identities, input hashes, and assertions under the new output directory.

The recorded run is `.runtime/alpha/model-policy/llm-context-verified/`.
The synchronous Python stage passed its source-fact assertions.
Async requests with the report at the start and end passed at 16,156 and 16,161 prompt tokens.
Both returned `finish_reason: stop` and the correct completed-run and empty-answer facts.
These requests exceed Laya's context while fitting chat's actual serving context.
They establish those tested direct-call boundaries, rather than behavior beyond chat's context limit.

The initial gate incorrectly required bare JSON and rejected a correct answer inside a Markdown code block.
The corrected gate uses the production stage parser and retains the original response under `llm-context/`.
No chat transport or lost-context failure appeared in these checks.

The live pm-coder checks also passed:

```powershell
# Run in C:/source/pm/pm-coder with its installed development environment.
.venv/Scripts/python.exe -m pytest test_session_isolation_live.py test_llm_coder.py::test_file_tools_roundtrip_on_real_filesystem -q
```

Both tests passed in 15.27 seconds.
They exercise real file edits and live compaction for two isolated agent sessions.
That compaction check does not force a server-context overflow.
The [A04 gate](../work_items/verification/a04-classification.md) also forces actual overflow with a real-history tool result.
It retains the provider rejection, live summaries, original instruction, class contract, checkpoint, and final source quotes.
