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
The serving window comes from configured `context_window` or the endpoint's current `/props` response.
Training-context metadata cannot establish the serving window.
`context_input_fraction` defaults to `0.8` and accepts a number strictly between zero and one.
The input budget equals that fraction of the serving window, in tokens.
The remainder reserves room for the response; it does not impose a generation cap.

Compatible servers apply their actual chat template and tokenize the resulting prompt.
Receipts label this exact measurement `server.chat_template_and_tokenizer`.
Servers without the template route use a labelled tiktoken estimate.
Their serving capacity still comes from `/props` when available.
Providers without that discovery route require configured `context_window`.
The estimate includes chat framing but cannot promise the provider's exact token count.
The provider remains authoritative if its tokenizer differs.
Transport errors, invalid measurement responses, and cancellation retain explicit failure boundaries.

When the original input exceeds its token budget, the project selects retained context and measures it again.
The original input remains archived with its digest.
An oversized selected input fails in tokens with its serving capacity and percentage in the error.
There is no character limit.
The same policy scales with 98,304, 186,624, and 1,000,000 serving tokens.
Native pm-coder history still uses its own serving-context discovery and automatic compaction.

The earlier A13 planner rejected 130,147 characters before contacting the model.
The previous llama.cpp server tokenized that retained, templated request as 50,160 tokens.
It advertised 131,072 serving tokens, so the old character limit caused an unnecessary rejection.
Older context-size receipts remain historical evidence of that application defect.

The current local chat server uses Strata and Qwen3.8-Flash-Next-Uncensored.
It advertises 262,144 serving tokens through `/props`.
It does not expose `/apply-template`, so direct project receipts label their token counts as estimates.
The default input budget is 209,715 tokens.
Strata reports actual prompt tokens through `/status`; those observations can differ from the application estimate.

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
