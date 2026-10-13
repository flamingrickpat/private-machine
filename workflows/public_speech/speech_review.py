"""Hallucination review of one public reply.

The review replaces the A09 claim-by-claim fact check. It asks one question:
does the reply state as real something that did not happen, or that the
companion cannot do? Lies, teasing, exaggeration, metaphor, guesses, and
feelings are part of the character and are never violations.

Division of work:

- The model lists violations, each with a kind, an exact quote, and a reason.
- Python weights them with ``speech_review.yaml``. A kind that the policy does
  not name weighs 0, so the model cannot block a reply with an invented kind.
- A quote that does not appear in the reply is dropped. The model cannot
  reject a reply for words it did not contain.

One review is one model call with thinking off. It never rewrites the reply.
The caller decides whether to generate again (``should_regenerate``) and
whether a reply may be published (``is_blocked``).
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from workflows.main_cognition.stage_support import extract_json
from workflows.public_speech.public_context import (
    DialogueTurn, PublicContext, ability_lines, memory_lines, task_lines,
)

REVIEW_FILE = Path(__file__).resolve().parent / "speech_review.yaml"


class Violation(BaseModel):
    """One hallucination that the reviewer found, with its policy weight."""

    model_config = ConfigDict(extra="ignore")
    kind: str
    quote: str = ""
    why: str = ""
    weight: int = 0


class SpeechReview(BaseModel):
    """The weighted result of one review."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "pm.speech-review.v1"
    violations: list[Violation] = Field(default_factory=list)
    ignored: list[Violation] = Field(default_factory=list)
    score: int = 0
    raw: str = ""


def load_policy(path: Path | None = None) -> dict:
    return yaml.safe_load((path or REVIEW_FILE).read_text(encoding="utf-8"))


def build_review_messages(candidate: str, context: PublicContext, policy: dict, *,
                          conversation: list[DialogueTurn] = (), receipts: list[dict] | None = None) -> list[dict]:
    """Return the two review messages. Pure: no model, no store."""
    names = {"companion_name": context.companion_name, "user_name": context.user_name}
    kinds = "\n".join(f"- {name}: {spec['meaning']}" for name, spec in policy["kinds"].items())
    system = policy["system_prompt"].format(kinds=kinds, **names)
    lines = [f'- {turn.speaker} said: "{turn.text}"' for turn in conversation]
    user = policy["user_template"].format(
        abilities="\n".join(ability_lines(context)) or "- Text conversation only.",
        tasks="\n".join(task_lines(context, receipts)) or "- No task or tool ran.",
        memories="\n".join(memory_lines(context)) or "- (none)",
        conversation="\n".join(lines) or "- (none)",
        reply=candidate.replace('"', "'"),
        **names,
    )
    return [{"role": "system", "content": system.strip()}, {"role": "user", "content": user.strip()}]


def weigh(payload: dict, candidate: str, policy: dict, raw: str = "") -> SpeechReview:
    """Turn the model's violation list into a weighted review.

    Unknown kinds and quotes that are not in the reply go to ``ignored`` with
    weight 0. Each distinct ``(kind, quote)`` counts once.
    """
    folded = " ".join(candidate.casefold().split())
    counted, ignored, seen = [], [], set()
    for item in payload.get("violations", []) or []:
        if not isinstance(item, dict):
            continue
        violation = Violation.model_validate(item)
        spec = policy["kinds"].get(violation.kind)
        quote = " ".join(violation.quote.casefold().split()).strip(" \"'")
        key = (violation.kind, quote)
        if spec is None or (quote and quote not in folded) or key in seen:
            ignored.append(violation)
            continue
        seen.add(key)
        violation.weight = int(spec["weight"])
        counted.append(violation)
    return SpeechReview(violations=counted, ignored=ignored, score=sum(item.weight for item in counted), raw=raw)


def should_regenerate(review: SpeechReview, policy: dict) -> bool:
    return review.score >= int(policy["policy"]["regenerate_at"])


def is_blocked(review: SpeechReview, policy: dict) -> bool:
    """Return True when the reply still contains a kind that must never be published."""
    blocking = set(policy["policy"].get("blocking_kinds", []))
    return any(item.kind in blocking for item in review.violations)


def regeneration_note(review: SpeechReview, policy: dict) -> str:
    """Describe the counted problems for the next generation attempt."""
    problems = "\n".join(f'- "{item.quote}": {policy["kinds"][item.kind]["meaning"]}' for item in review.violations)
    return policy["regeneration_note"].format(problems=problems).strip()


async def review_speech(complete, candidate: str, context: PublicContext, *,
                        conversation: list[DialogueTurn] = (), receipts: list[dict] | None = None,
                        policy: dict | None = None) -> SpeechReview:
    """Review one reply with one model call.

    Raises ``StageOutputError`` when the model returns no JSON object.
    Model transport errors propagate from ``complete``.
    """
    policy = policy or load_policy()
    messages = build_review_messages(candidate, context, policy, conversation=conversation, receipts=receipts)
    raw = await complete(messages, temperature=0.0, chat_template_kwargs={"enable_thinking": False})
    return weigh(extract_json(raw), candidate, policy, raw=raw)


__all__ = [
    "SpeechReview",
    "Violation",
    "build_review_messages",
    "is_blocked",
    "load_policy",
    "regeneration_note",
    "review_speech",
    "should_regenerate",
    "weigh",
]
