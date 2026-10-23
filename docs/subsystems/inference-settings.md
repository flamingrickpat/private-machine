# Inference settings and parallel calls

## Summary

Shared constants control direct PM reasoning and independent cognition calls.
All inner-thought stages disable reasoning.
Codelets and attention batches run concurrently, then merge in a stable order.
Public answers keep their separate generation and review steps.

Parent: [model connections](../models.md).
Related: [cognition](cognition.md), [codelets](codelets.md), and [the private loop](private-loop.md).
Evidence: [reasoning and speed verification](../work_items/verification/inference-speed.md).

## Settings owner

`workflows/inference_settings.py` owns the companion's policy.
[`pm.llm.reasoning.ReasoningBudget`][] supplies the arithmetic without importing companion behavior.
[`pm.llm.activation.ChatClient`][] translates the resulting controls at the provider boundary.

The inner-thought runner selects the `off` profile for every stage, including reflection.
The following profiles remain available for other explicit callers.

| Reserved operation group | Fraction of remaining output | Reasoning ceiling |
| --- | ---: | ---: |
| notice | 2% | 512 tokens |
| remember | 3% | 768 tokens |
| interpret | 3% | 768 tokens |
| imagine | 4% | 1,024 tokens |
| evaluate | 5% | 1,024 tokens |
| inspect | 5% | 1,024 tokens |
| plan | 5% | 1,536 tokens |

The trigger, integration and main cognition stages keep reasoning disabled by default.
An explicit main-stage generation override takes precedence over that default.
Native pm-coder sessions retain their driver controls; these profiles govern direct PM calls.

## Remaining output allowance

Activation retains model metadata from its existing discovery request.
The configured context bound takes precedence over the advertised context bound.
The provider's advertised maximum completion length also limits the available output.
An explicit request output limit can reduce that maximum.

Input tokens use the existing labelled tiktoken estimate.
Tool definitions enter that estimate when present.
A ten-percent input margin reserves space for estimation error.
The policy reserves 2,048 visible-answer tokens, or 4,096 for planning, before allocating reasoning.

```text
available = min(provider_output_limit, context_window - estimated_input_with_margin)
headroom = available - visible_answer_reserve
reasoning = min(headroom, ceiling, max(128, floor(headroom * fraction)))
```

Insufficient headroom raises an explicit error.
The caller must reduce its input or adjust the policy.
The runtime does not shorten source evidence to make a request fit.
The answer reserve establishes budgeting headroom; it does not cap the visible answer.

OpenRouter receives `reasoning.max_tokens` separately from the total completion limit.
For a calculated OpenRouter budget, the model must advertise numeric reasoning-budget support.
Local calls use `reasoning_budget_tokens`, as supported by the selected Strata server.
Local discovery needs no OpenRouter-specific reasoning metadata.

The [OpenRouter reasoning contract](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) distinguishes reasoning limits from total completion limits.
Request logs retain the actual outbound allowance.
Provider usage remains the evidence of actual token use.
A per-call ceiling does not impose a total conversation budget.

## Explicit reasoning controls

[`pm.llm.reasoning.ReasoningOptions`][] accepts four optional fields.
Callers can supply this typed object or a dictionary as the completion's `reasoning` argument.
Omitted fields preserve provider defaults.

| Field | Accepted values | Meaning |
| --- | --- | --- |
| `enabled` | `true`, `false` | Enable or disable thinking |
| `max_tokens` | Positive integer | Request a reasoning token allowance |
| `effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | Select a provider-supported effort level |
| `exclude` | `true`, `false` | Hide reasoning from the returned completion |

An effort level and a token allowance are alternative controls.
Disabled thinking cannot carry a positive allowance or effort.
Contradictory controls raise a validation error before the request.
A calculated `reasoning_budget` and explicit reasoning controls cannot share a call.

```python
from pm.llm.reasoning import ReasoningOptions

result = model.complete_result(turns, reasoning=ReasoningOptions(max_tokens=768), max_tokens=4096)
```

The nested allowance controls thinking.
The top-level `max_tokens` controls the combined completion on most providers.
The final answer needs space after reasoning.
An exhausted completion raises an explicit model error, rather than publishing empty speech.

[`pm.llm.activation.provider_generation`][] owns translation.
OpenRouter receives the `reasoning` object and `provider.require_parameters=true` by default.
An explicit provider-routing setting takes precedence.
Local calls receive `chat_template_kwargs.enable_thinking`, `reasoning_effort`, and `reasoning_budget_tokens` as applicable.
The wrapper also removes returned reasoning when `exclude=true`, including responses from local servers that retain it.
Usage and raw diagnostic logs retain their existing boundaries.
Exclusion does not reduce reasoning work or cost.

Legacy `enable_thinking`, `reasoning_tokens`, `reasoning_effort`, and `reasoning_budget_tokens` arguments remain supported.
Explicit `reasoning` takes precedence over the legacy thinking switch.
Duplicate budget or effort sources raise an error.
Unrelated completion kwargs pass through unchanged.
Accepted effort levels still depend on the serving model.
Strata maps several levels to the same local template setting.
This implementation does not start or modify that external server.

Development uses `config.alpha.openrouter.yaml`.
Local wire translation has deterministic boundary checks; local inference remains outside this delivery's acceptance evidence.

## Wait through an endpoint outage

The program does not fail when a model endpoint is down. It waits.

Every endpoint contact repeats until the endpoint answers. The wait covers model
activation, serving-context discovery on `/props`, the tokenizer probes, and chat
and embedding requests. Each failed attempt prints one line to standard error and
waits 30 seconds. The attempt count has no limit.

Start the endpoint again, and the paused call continues on its own. The wait holds
no store lock and commits nothing, so the run resumes where it stopped.

Cancellation still ends the wait. An application stop or a run stop sets the
cancellation signal, and the waiting call raises `WorkflowTerminated` or
`asyncio.CancelledError`. See
[`pm.llm.http_client.wait_until_ready`][] and
[`pm.llm.activation.ModelActivation.activate`][].

The `live_test` mode keeps the first-error behavior. Unit tests, development
gates, and the startup context-budget probe use that mode, so the terminal
session opens even before the model server is up. Later inference waits.

## Parallel execution and cancellation

`CODELET_WORKERS` and `ATTENTION_WORKERS` each default to two.
`CHAT_CONCURRENCY` bounds application workers at four.
A shared HTTP admission gate admits one direct completion per endpoint at a time.
The independent Jesse model and native worker sessions have separate owners.

[`pm.llm.concurrency.ordered_parallel_map`][] joins all owned workers before returning.
The first failure sets the invocation's cancellation signal.
Sibling model requests use that signal to leave HTTP and retry waits.
The original failure propagates after cleanup.
Queued cancelled callbacks do not start external work.
A completion tool that returns the wrong Python type is a programming error and propagates.
Schema-repair exhaustion retains its last validation error; Python exception-variable cleanup cannot erase it.

Codelets of one wave run in parallel.
Their results apply to the CSM in run order before the next wave starts, and nomination IDs follow that order.
Attention batches receive disjoint nominations.
Their results merge before the existing competition and tie-breaking rules run.
The eight main stages keep their causal order.

Application stop signals also reach Story, Assistant and private-stage requests.
Admission waits use asynchronous polling and create no waiting worker threads.
Every admitted request releases its slot after cleanup.

## Shared prompt prefix

Every codelet call starts with three shared turns:

1. The story-simulator system prompt.
2. The story: character card, recent dialogue, memories, goals, the feeling in words, and the new stimuli.
3. The acknowledgement of the story.

The codelets phase renders these turns once per cycle, so they stay equal across all its calls.
The codelet's view of the CSM and its instruction come last.
See [codelets](codelets.md#prompt-frame).
The public Story voice prompt retains its existing contract.

## Evidence

The reviewed long-gap run used a median 101.25 seconds for its cognition chain.
Its spoken generation took a median 5.05 seconds.
Those timings exclude several preparation and commit boundaries.
The private thought stage emitted 43,794 hidden reasoning tokens across 19 requests.
These observations motivated the new settings; they do not prove a speedup.

The first real budget probe used captured private prompts with the configured provider.
The notice, interpret and plan profiles returned 423, 548 and 248 reasoning tokens.
Each response retained a nonempty, schema-valid thought within its configured ceiling.
Evidence stays under `.runtime/alpha/20261015-speed-budget-probe/`.
This establishes provider execution and output validity, not human conversational equivalence.

Real OpenRouter calls also exercised disabled thinking, a numeric allowance, low effort, and exclusion.
Each returned the correct JSON answer for a small arithmetic request.
Exclusion retained final content and usage without returning reasoning.
The first probe's later startup replay failed because its driver assumed an on-disk request file.
Its four completed control calls remain visible under `.runtime/alpha/20261015-speed-controls-probe/`.
The separate captured-startup replay passed generation and review under `.runtime/alpha/20261015-speed-startup-replay/`.
These small probes establish parameter wiring, not broad reasoning quality.

## Checkpoint serialization

[`pm.memory.checkpoint.CheckpointSerializer`][] owns encoding and decoding.
The default [`pm.memory.checkpoint.PickleCheckpointSerializer`][] writes typed state directly.
Every record retains its embedding. The save path does not convert records to JSON mappings.
[`pm.memory.checkpoint.JsonCheckpointSerializer`][] remains available for explicit JSON export and old checkpoint imports.
File ownership, flush, fsync, cancellation, and atomic replacement remain in [`pm.memory.checkpoint.write_checkpoint`][].
Legacy YAML seeds remain readable. Normal application saves and backups use pickle.
Read [memory ownership](../ghost.md) for shared records, isolated edits, and startup embedding.

## Foreground priority

The private loop already runs on its own thread.
[`pm.llm.concurrency.InferenceAdmission`][] controls direct completion admission across those threads.
An active private completion finishes normally.
Queued main cognition and dialogue events reserve foreground priority until their workflows finish.
The reply reservation starts during dispatch, before the main reservation ends.
Private inference resumes after the foreground reservations drain.
Cancellation removes a waiting request from the queue.

The shared story prefix remains unchanged.
Reconsolidation summaries and state descriptions now share their scene prefix.
Narrative evidence precedes the specific narrative and its instructions.
Mental-work prompts place the shared workspace before the task instruction.
These changes retain the supplied evidence and its order.
