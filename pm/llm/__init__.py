"""Model access for the runtime: configuration, activation, transport, encoding.

This package owns every conversation with a model endpoint. It knows nothing
about companion behavior: a workflow supplies neutral
:class:`~pm.context_render.turns.Turn` objects, generation settings, and a
cancellation event, and receives text, a typed completion, or a validated
Pydantic model back.

The layers, in the order one call passes through them:

1. :mod:`pm.llm.config` resolves the configured chat and embedding connections
   without a network call. A credential is read from the process environment or
   the git-ignored ``.env`` (:mod:`pm.llm.env_file`).
2. :class:`pm.llm.activation.ModelActivation` owns which model this process
   admits calls for. Activation runs a finite startup script, waits until the
   endpoint serves the configured alias, and records the discovery metadata
   that reasoning budgets read.
3. :class:`pm.llm.stage_model.LazyStageModel` is the object workflows actually
   hold. It activates on first use, measures the rendered prompt against the
   serving context (:mod:`pm.llm.context_window`), and admits one request at a
   time per endpoint (:mod:`pm.llm.concurrency`).
4. :mod:`pm.llm.multimodal` resolves ``[image:<id>]`` tags in prompt text into
   provider image parts; :mod:`pm.llm.prompt_encoding` is the only place where
   typed turns and attachment bytes become provider JSON.
5. :mod:`pm.llm.http_client` owns transport. Transient failures wait and repeat
   the same request; permanent HTTP errors, provider failures, and invalid JSON
   fail immediately (:mod:`pm.llm.cache` replays a matching cached response).
6. :mod:`pm.llm.structured` repairs model JSON against a Pydantic schema before
   a workflow accepts it. :mod:`pm.llm.prompt_log` writes the readable prompt,
   response, and usage receipt beside the completion cache.

Invariants shared by every module here:

- A logic error keeps its traceback. Only an unreachable or not-yet-ready
  endpoint waits, and only under a cancellation event.
- Reasoning stays separate from final content; a final answer is never
  substituted with private reasoning.
- No credential, response body, or header enters a log line, an error message,
  or a trace.
- A generation group holds one active call, so no model switch can happen while
  another completion is in flight.

Reading map: start at :mod:`pm.llm.stage_model` for the call path,
:mod:`pm.llm.http_client` for retry and failure classification, and
:mod:`pm.llm.prompt_encoding` for the provider wire shapes.
"""
