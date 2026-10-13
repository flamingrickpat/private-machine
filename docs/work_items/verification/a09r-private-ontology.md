# A09R private ontology and agentic voice verification

## Summary

A09R supersedes the A09 review mechanism.
It implements the private ontology as durable state and makes public speech
generation and review agentic and persona-driven.
Only a false task result is a hard block.
The accepted reply records the reason of its one review.

This record separates engineering delivery from behavioral acceptance.
The contract and unit checks pass with a real MemoryStore and a deterministic
review.
The live model gates are recorded as unavailable in this session, because the
configured chat endpoint was not running.
No live rating is invented.

Parent: [public dialogue](../../subsystems/public-dialogue.md).
Delivery: [A09R](../alpha-plan/work-items/a09r.md).
Design: [private ontology](../alpha-plan/ontology.md), [second conscious loop](../alpha-plan/private-loop.md), and [confirmed D02/D10](../alpha-plan/decision-register.md#confirmed-direction).
Supersedes: [A09](a09-public-dialogue.md) review mechanism.

## Delivered mechanism

| Deliverable | Implementation | Entry point |
| --- | --- | --- |
| Private ontology as durable state | `workflows/public_speech/ontology.py` writes `OntologyTermRecord` and `OntologyBindingRecord` into the Ghost. | [`observe_ontology`][workflows.public_speech.ontology.observe_ontology] |
| Agentic persona-driven review | One model call returns a judgment, a reason, and advisory findings. | [`review_speech`][workflows.public_speech.speech_review.review_speech] |
| Only a false task result blocks | `is_blocked` reads the explicit flag. Findings never block. | [`is_blocked`][workflows.public_speech.speech_review.is_blocked] |
| Reason per accepted reply | The review reason enters the report and the public Feature. | `review_reason`, `review_judgment` |

The weighted policy file lost its kinds, weights, and thresholds. It now holds
prose only.
The Story regeneration loop was removed.
The pattern-based `ontology.yaml` compatibility gate was removed.
The private ontology is no longer a sentence-by-sentence gate.

## Contracts

[`workflows.public_speech.speech_review.SpeechReview`][] is the new review
contract:

```json
{"judgment": "accept", "reason": "short reason", "false_task_result": false,
 "findings": [{"kind": "invented_ability", "quote": "exact words", "why": "few words"}]}
```

`invented_ability`, `invented_event`, `unsupported_label`, and `private_leak`
are advisory findings.
`false_task_result` is the only hard block.
The review prompt receives the character card, the accepted feelings, the
abilities, the private ontology terms and active bindings, the task results,
the memories, the conversation, and the chosen presentation.
A prior deception intent is described as an intentional action with retained
facts, so a chosen lie is not a false task result.

## Contract and unit evidence

Real MemoryStore, deterministic review, no model endpoint:

| Command | Result |
| --- | --- |
| `uv run --locked python -m unittest discover -s tests/unit -t . -b` | 504 tests pass in 700.542 s |
| `uv run --locked python -m unittest discover -s tests/contract -t . -b` | 63 tests pass in 1.850 s |
| `python -m unittest tests.unit.workflows.public_speech.test_speech_review tests.unit.workflows.public_speech.test_ontology` | 16 tests pass |
| `python -m unittest tests.contract.dialogue_story.test_story_contract tests.unit.workflows.public_speech.test_message_protocol tests.unit.workflows.public_speech.test_publication` | 39 tests pass in 0.576 s |
| `python -m unittest tests.unit.workflows.dialogue_assistant.test_assistant_dialogue tests.unit.workflows.test_companion_wiring tests.unit.workflows.main_cognition.test_cognition_wiring` | 24 tests pass in 316.284 s |

Evidence logs: `.runtime/a09r/foundation-a09r.log` and `.runtime/a09r/contract-a09r.log`.

The new ontology check `tests/unit/workflows/public_speech/test_ontology.py`
verifies:

- the seed terms come from `StateEmotions`, `StateRelationship`, and the fixed
  mental moves,
- a sense that the runtime does not have is not a term,
- a notable feeling binds to the accepted state,
- a weak feeling does not bind,
- a changed state supersedes the old binding,
- a second observation adds no record.

The review check `tests/unit/workflows/public_speech/test_speech_review.py`
verifies:

- a finding does not block and the reason is kept,
- a false task result is the only block,
- one draft produces exactly one review,
- the deception intent reaches the review prompt,
- the review input contains no evidence ids.

The Story contract check verifies that an invented ability is reported as a
finding and the reply is still accepted.

## Documentation

The engineering manual was regenerated with the strict build:

```powershell
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

The build completed with no unresolved links.
The changed guides are [Story speech](../../subsystems/story-speech.md),
[Public dialogue](../../subsystems/public-dialogue.md), and
[the private ontology](../../subsystems/private-ontology.md).
The generated pages `site/subsystems/private-ontology.html` and
`site/work_items/verification/a09r-private-ontology.html` were inspected.
Their API links resolve to the generated reference pages for
`workflows.public_speech.ontology` and
`workflows.public_speech.speech_review`.
The site uses `use_directory_urls: false`, so local navigation points at HTML
files.
Evidence: `.runtime/a09r/mkdocs-strict.log`.

## Live gates

The configured chat endpoint `http://127.0.0.1:8080/v1` was not running in this
session.
The live checks were therefore not run:

- `tests/live/test_speech_review.py` (real model finds planted hallucinations
  and passes an in-character reply),
- `tests/live/test_story_dialogue.py` (real accepted dialogue),
- `dev/work_items/alpha_a09_public_gate.py` (both public paths and the
  connected capability comparison),
- `dev/work_items/alpha_a09_dialogue_gate.py` (real console conversation, C01,
  verified note bytes, and C02).

No live rating is recorded.
The operator still needs to complete M2 with this mechanism: ask about a
capability, a remembered goal, and a private topic, then rate continuity and
the desire to regenerate each reply.

## Verification still required before A09R is complete

The A09R work item lists behavioral cases that need a real conversation:

- An invented ability is reported and does not become an accepted belief.
- An invented event is reported and does not become an accepted memory.
- A false task result is blocked and produces the explicit recovery text.
- A prior deception intent produces a deliberate false statement with retained
  factual state.
- A metaphor, exaggeration, or teasing line is not blocked.
- An unsupported emotion label stays visible and does not create an emotion
  field.
- The reply latency for an ordinary turn stays suitable for conversation.
- A rule-based baseline and the agentic review are compared on the same
  captured replies.

The A09 note case and the C01/C02 regression cases are repeated through the
real console, and the old A09 receipts stay beside the new ones.

## Limits

- The live behavioral acceptance is unverified in this session.
- The term set is seeded from implemented mechanisms only. A17 grows it from
  accepted experience.
- The `unsupported_label` finding is advisory. It does not prove that an
  accepted reply has no other unsupported claim.
- The old A09 console and public gate receipts remain historical evidence.
  They do not verify the A09R mechanism.
