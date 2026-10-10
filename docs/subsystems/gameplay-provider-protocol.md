# Native gameplay provider protocol

## Summary

The native adapter preserves provider reasoning metadata across tool turns.
It also separates input context capacity from generated output allocation.
Actual hosted file-tool probes establish these protocol boundaries.
They do not establish gameplay acceptance.

Parent: [scoped workflow decisions](gameplay-decision-scopes.md).
Context: [LLM context](llm-context.md).

## Generic native output budgeting

The hosted native probe exposed an independent `pm-coder` request error.
The adapter requested the entire serving context as generated output, in addition to input and tool schemas.
The provider rejected that request before inference.

The isolated dependency repair removes that output reservation from actor and summarizer requests.
The endpoint allocates output; the configured context still governs input tools and automatic compaction.
This output-allocation repair contains no scenario or provider branch.
Its temporary local installation has recorded source hashes; the final Git dependency pin remains pending acceptance.

## Native provider protocol

Local and hosted inference use the same OpenAI-compatible chat endpoint shape.
Provider extensions still need the correct adapter for parsing and replay.
The original native adapter retained thinking text but omitted OpenRouter's structured `reasoning_details` on subsequent tool turns.
Its local thinking-control field also did not disable hosted reasoning.

The dependency repair selects the SDK's `OpenRouterModel` and `OpenRouterProvider` for the exact `openrouter.ai` host.
Other hosts retain the OpenAI adapter. Selection does not depend on a model name or gameplay task.
Shared request logging remains unchanged.
The existing disabled-thinking setting sends the provider's `reasoning.enabled=false` control on hosted requests.
The [SDK provider documentation](https://ai.pydantic.dev/models/openrouter/) describes its reasoning support.

The real disabled-reasoning file-tool probe passed five native calls with no output reservation or hidden reasoning tokens.
The enabled-reasoning probe recorded eight actual responses.
Its initial dictionary-equality audit failed because SDK serialization added nullable metadata fields.
A separate audit checked 25 actual replay sequences against original values, block order, and tool-call identities.
All passed; added fields were limited to nullable `id`, `signature`, `format`, or `index` metadata.
The original failed diagnostic receipt remains unchanged.
Captured protocol tests cover saved-history serialization and exact host selection without new inference.
These results establish provider boundaries, not Minecraft acceptance.

## Hosted decision latency

The hosted replay used the exact `0144-reviewer` input.
Its 2,048-token diagnostic output limit produced 2,048 reasoning tokens and no visible answer in 36.672 seconds.
The same replay with `reasoning.enabled=false` returned an accepted review in 4.328 seconds, with zero reasoning tokens.
Both receipts remain separate from managed acceptance.

The local `chat_template_kwargs.enable_thinking=false` field did not disable hosted reasoning.
The [OpenRouter reasoning contract](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) describes its separate control.
The ignored gate settings file selects that control for direct decisions.
Native actors use their existing thinking setting through the provider-specific control described above.

The later `0175-reviewer` entry request has no recorded response from its interrupted worker.
Two independent replays accepted its entry contract in 3.047 and 2.234 seconds.
The second replay used the original generation configuration with no output limit.
Neither replay changes task acceptance or proves why the original execution stopped.
The interruption snapshot records released worker ownership and the original receipt hash.
