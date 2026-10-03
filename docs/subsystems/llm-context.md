# Python chat and agent context

## Summary

Laya and chat have separate context limits.
Direct Python chat sends one supplied request, while pm-coder owns agent history and automatic compaction.
Real-endpoint checks verify direct calls beyond Laya's input limit.
A04 also verifies actual server-context overflow and live agent recovery with decisive classification evidence intact.

Parent: [model connections](../models.md).
Policy: [alpha model selection](../decisions/alpha-plan/model-policy.md).

## Separate service limits

Chat, embeddings, and typed decisions are separate services with separate limits.
The verified local chat endpoint on 8080 advertises 131,072 serving tokens.
Laya on 8082 advertises 8192 state-context tokens and a separate 192-token decision head.
Laya's head can truncate questions and options.
That mechanism does not truncate the chat service's questions or disable pm-coder compaction.

[`runtime.model_activation.ChatClient.complete`][] sends one supplied message list and returns the full completion.
It does not own conversation history or run automatic agent compaction.
[`runtime.cognitive_model.LazyStageModel`][] exposes that path to synchronous and asynchronous Python roles.
The caller constructs context from relevant memories, selected workspace evidence, and the current request.
Structured stages use their actual parser and schema rather than treating every raw completion as an accepted result.
The cognitive parser accepts a JSON object inside a Markdown code block.

Agentic work uses pm-coder's `run_turn` loop.
Its settings discover the serving context unless the caller explicitly selects a budget.
Context exhaustion triggers a live summary, one execution checkpoint, and a bounded recent tail.
The original request remains exact, and the agent resumes from retained work.
A single oversized initial request with no execution history requires input selection rather than agent-history compaction.
Long-file classification uses scoped agent reads and preserves required conclusions through compaction.

## Verified direct Python calls

Run the retained gate against the actual chat service:

```powershell
uv run --locked python scripts/alpha_llm_context_gate.py --config config.alpha.example.yaml --out .runtime/alpha/llm-context-repeat
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
The [A04 gate](workflow-inputs/classification-verification.md) also forces actual overflow with a real-history tool result.
It retains the provider rejection, live summaries, original instruction, class contract, checkpoint, and final source quotes.
