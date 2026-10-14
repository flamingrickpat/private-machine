# Model selection and reliability

## Summary

The companion uses any OpenAI-compatible chat and embedding endpoint.
The endpoint can be local or remote.
Configuration supplies the endpoint, the model names, and the credential reference.
A02 found poor decision-classifier quality, including confidently wrong answers.
The normal LLM stays the default for selection, classification, and interpretation.
An optional decision classifier is enabled only after live tests establish one specific use.
If enabling it needs expensive work, keep the LLM.

Parent: [alpha plan](index.md).
Contracts: [classification inputs](workflow-inputs.md) and [decision register](decision-register.md).
Evidence: [A02 model observations](../../subsystems/decisions/quality.md) and [model operation](../../models.md).

## Model requirement

The companion needs an OpenAI-compatible chat endpoint and an embedding endpoint.
The endpoint location is irrelevant.
A local server and a hosted API are both valid.
The YAML configuration file supplies the base URL and the model names.
The process environment or a repository `.env` file supplies the credential.
The process environment wins. The `.env` file is the fallback.

The serving model must support at least **128,000 tokens of context**.
The serving model must have **at least the quality of the current `qwen3.8-flash-next`** model.
The model name can change when a better model appears.
The requirement is the capability class, not one model.
A later Qwen 4.0 release with better quality and speed is a valid substitute.

The current deployment uses `qwen3.8-flash-next` for chat and
`perplexity/pplx-embed-v1-0.6b` for embeddings.
The example configuration records the endpoints.
A credential comes from a Windows environment variable on one machine and from a
repository `.env` file on another machine.

Do not assume a specific serving route, a specific serving binary, or a specific
vision model.
vLLM is one possible server. It is not required.
A caller supplies the declared `context_window` when a server does not expose its
serving limit.

The embedding configuration must accept any OpenAI-compatible embedding model.
The current request and response identity checks exist because OpenRouter returns
an alias.
A different server that returns the requested model name needs no alias field.

## Generalization ladder

The [current development priorities](companion-development.md) favor companion quality over conversation latency.
Do not replace useful reasoning or memory access with a weaker path solely to reduce latency or tokens.
Keep performance measurements and reliability controls.
An optimization must preserve the required behavior on comparable cases.

Alpha uses inspectable deterministic rules for attention selection, routing, and
public review.
Beta replaces those rules with agentic equivalents when a tested reason exists.
The current rules are the regression baseline.
A replacement must match or beat the rule-based behavior on the same cases.
This policy turns "explicit logic" into a deliberate, test-backed alpha stage.
Read the [generalization ladder](index.md#generalization-ladder) for the method.

## Optional decision classifier

The decision classifier was planned as a frequent fast path.
The earlier 95% usage target is withdrawn.
No alpha item requires a percentage of decision-classifier use or a successful adoption.
The delivered client, local artifact, deployment instructions, and receipts remain
available for later experiments.
Keeping that technology does not require enabling it in PM.

The LLM can be the primary classifier, rather than waiting for the classifier to fail
first.
A high-confidence wrong answer can bypass an uncertainty fallback.
A distribution, margin, or successful HTTP response cannot establish semantic
correctness.

Complex report interpretation, ambiguous file selection, cognitive appraisal,
planning, dialogue, and reflection use the LLM.
Exact metadata and deterministic parsing use ordinary code when they establish the
required result.
The decision classifier is a candidate only for a short input, explicit classes, and a simple
fact-based decision.
Separate tests must establish file selection and content classification when both
use the classifier.

## Before enabling a decision-classifier use

1. Define the input schema, class descriptions, evidence requirements, and
   permitted input lengths.
2. Write unit tests that call the actual endpoint and assert independently defined
   expected results.
3. Use realistic synthetic data with normal application formats, distractors,
   missing fields, contradictory facts, and invalid cases.
4. Include captured application inputs when available.
5. Exercise decisive evidence at different positions within the intended input
   budget.
6. Test long questions and options against the decision head budget.
7. Run the same cases through the normal LLM for a useful comparison.
8. Record wrong answers, critical misroutes, INVALID handling, timing, endpoint
   identity, and exact input digests.
9. Enable the classifier only after every required case passes and a useful reason for
   adoption remains.

Synthetic data supplies realistic test inputs, never simulated inference.
Expected results follow explicit source facts rather than either model's answer.
Straightforward synthetic assertions need no separate human-label approval.
Ambiguous interpretation and persona immersion still need human review.

Untouched cases protect against prompt changes that only fit known examples.
Each candidate needs an appropriate independent test set, rather than mandatory
review of the old A01 corpus.
Preserve the A01 holdout and its recorded exposure status.
Neither that pending review nor a failed optional decision-classifier experiment blocks LLM
delivery.

If tests reveal recurring errors or require extensive prompt tuning, retraining, or
integration work, keep the LLM.
Do not expand alpha scope to improve the decision classifier.
Model, server, class, prompt, or budget changes require the affected live tests
again before use.

## LLM reliability is a core requirement

The LLM path must remain usable independently of the decision classifier.
Service absence, direct Python failures, unusable outputs, and lost required
context need immediate diagnosis and repair.
An optional model experiment cannot excuse a failure in the normal LLM path.
Every required live acceptance case must pass before the affected mechanism is
delivered.

pm-coder's agent loop discovers the serving context and compacts accumulated
execution history after context exhaustion.
It retains the original request, creates a checkpoint, and resumes from recorded
work.
A03 through A20 must preserve that mechanism for agentic tasks and long-file
inspection.

PM's direct `ChatClient` sends one supplied message list and does not run the
pm-coder loop.
Its caller must construct suitable task context through scoped retrieval,
selection, or explicit evidence-preserving compaction.
An unbounded history dump is not a replacement for those mechanisms.
Long agentic work must use pm-coder rather than a raw completion with accumulating
history.

The decision classifier has its own configured context.
That context does not configure the chat service or disable pm-coder compaction.
Chat still has its own finite serving context.
Required evidence must survive context handling, and an incomplete generation
cannot become an accepted classification or task result.

## Known companion defects

[C01](baseline/receipts.md#diagnosed-conversation-failure) recorded a completed
managed worker but left the console exchange pending without a public reply.
This is an action-to-reply defect, rather than a model connection failure.
The control worker's artifacts do not establish an exact-content note or arbitrary
instruction execution.
A09 owns a real agent-backed note acceptance case, actual file verification, and
useful public reply delivery.

[C02](baseline/receipts.md#current-verification) restored equal records, GWT, and
task receipts.
Sunny still claimed that a pre-restart codeword did not survive, although the only
supplied codeword was `copper lantern`.
This is a recall or interpretation defect, rather than evidence of lost checkpoint
data.
A08 verifies original evidence retrieval, and A09R verifies grounded public claims.
Keep both defects visible until new real console evidence establishes their repair.
