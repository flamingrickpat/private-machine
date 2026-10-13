"""Agentic review of one public reply.

The review is one model call with a persona-aware prompt. It judges the reply
against the accepted state, the abilities, the private ontology, and the task
results. It returns a short ``judgment`` and a ``reason``.

The review is not a fixed policy. There are no per-kind weights and no score.
An advisory ``finding`` records something that looks unsupported, such as an
invented ability or an invented event. A finding stays in the receipt for
review. It never blocks a reply and never triggers regeneration.

Only ``false_task_result`` is a hard block. A false task result can change the
user's actions on a real object. A persona-driven reply, including a chosen
lie or a metaphor, must not become a false task result. Python decides the
block from the model's explicit ``false_task_result`` flag, so an invented
kind cannot reject a reply.

The caller records ``review.reason`` beside every accepted reply. The one
review never rewrites the reply.
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


class ReviewFinding(BaseModel):
    """One advisory observation about the reply. It never blocks publication."""

    model_config = ConfigDict(extra="ignore")

    kind: str = ""
    quote: str = ""
    why: str = ""


class SpeechReview(BaseModel):
    """The result of one agentic review.

    ``judgment`` is the model's short verdict; ``blocked`` is the runtime's
    decision and depends only on ``false_task_result``. ``findings`` are
    advisory and stay visible for human review.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "pm.speech-review.v2"
    judgment: str = "accept"
    reason: str = ""
    false_task_result: bool = False
    findings: list[ReviewFinding] = Field(default_factory=list)
    raw: str = ""


def load_prompts(path: Path | None = None) -> dict:
    """Load the review prompt file. It holds prose only, never weights."""
    return yaml.safe_load((path or REVIEW_FILE).read_text(encoding="utf-8"))


def _ontology_lines(context: PublicContext) -> list[str]:
    """Render the private ontology terms and their active bindings.

    A term is a stable word for an implemented state. A binding shows the
    state that currently supports the term. The companion may use these words.
    An unsupported label is not in this list, so the review can mark it.
    """
    from workflows.public_speech.ontology import ontology_lines
    return ontology_lines(context)


def build_review_messages(candidate: str, context: PublicContext, prompts: dict, *,
                          conversation: list[DialogueTurn] = (),
                          receipts: list[dict] | None = None) -> list[dict]:
    """Return the two review messages. Pure: no model, no store."""
    names = {"companion_name": context.companion_name, "user_name": context.user_name}
    presentation = context.presentation
    if presentation.mode == "deceive":
        presentation_text = (f"{context.companion_name} chose to deceive on purpose. "
                             f"Purpose: {presentation.purpose or 'unstated'}. "
                             f"Known facts kept: {presentation.known_fact_ids or 'none listed'}. "
                             "The false words are an intentional action, not a false task result.")
    elif presentation.mode == "withhold":
        presentation_text = (f"{context.companion_name} chose to withhold something private. "
                             f"At most they may share: {presentation.shareable_summary or 'nothing'}.")
    else:
        presentation_text = f"{context.companion_name} chose to speak candidly."
    lines = [f'- {turn.speaker} said: "{turn.text}"' for turn in conversation]
    user = prompts["user_template"].format(
        character=context.character.strip() or "(no character card)",
        feelings=context.feelings.strip() or "(calm, nothing strong)",
        abilities="\n".join(ability_lines(context)) or "- Text conversation only.",
        ontology="\n".join(_ontology_lines(context)) or "- (no ontology term recorded yet)",
        tasks="\n".join(task_lines(context, receipts)) or "- No task or tool ran.",
        memories="\n".join(memory_lines(context)) or "- (none)",
        conversation="\n".join(lines) or "- (none)",
        presentation=presentation_text,
        reply=candidate.replace('"', "'"),
        **names,
    )
    return [{"role": "system", "content": prompts["system_prompt"].format(**names).strip()},
            {"role": "user", "content": user.strip()}]


def parse_review(payload: dict, raw: str = "") -> SpeechReview:
    """Read the model payload into a review. Only the explicit flag blocks."""
    if not isinstance(payload, dict):
        raise ValueError("The review model returned JSON that is not an object.")
    findings = []
    for item in payload.get("findings", []) or []:
        if isinstance(item, dict):
            findings.append(ReviewFinding.model_validate(item))
        elif isinstance(item, str) and item.strip():
            findings.append(ReviewFinding(kind=item.strip()))
    false_task_result = bool(payload.get("false_task_result", False))
    judgment = str(payload.get("judgment", "") or "").strip().lower()
    if not judgment:
        judgment = "block" if false_task_result else "accept"
    return SpeechReview(
        judgment=judgment,
        reason=str(payload.get("reason", "") or "").strip(),
        false_task_result=false_task_result,
        findings=findings,
        raw=raw,
    )


def is_blocked(review: SpeechReview) -> bool:
    """Return True only for a false task result. Findings never block."""
    return bool(review.false_task_result)


def block_reason(review: SpeechReview) -> str:
    """Return a short reason for a blocking receipt."""
    if review.reason:
        return review.reason
    for finding in review.findings:
        if finding.kind == "false_task_result" or finding.quote:
            return finding.why or finding.quote or "false task result"
    return "The reply claimed a task result that no receipt supports."


def finding_summary(review: SpeechReview) -> str:
    """Render the advisory findings as one receipt line, or an empty string."""
    return "; ".join(f"{item.kind}: {item.why or item.quote}".strip(": ") for item in review.findings
                     if item.kind or item.quote or item.why)


async def review_speech(complete, candidate: str, context: PublicContext, *,
                        conversation: list[DialogueTurn] = (),
                        receipts: list[dict] | None = None,
                        prompts: dict | None = None) -> SpeechReview:
    """Review one reply with one model call.

    Raises ``StageOutputError`` when the model returns no JSON object.
    Model transport errors propagate from ``complete``.
    """
    prompts = prompts or load_prompts()
    messages = build_review_messages(candidate, context, prompts, conversation=conversation, receipts=receipts)
    raw = await complete(messages, temperature=0.0, chat_template_kwargs={"enable_thinking": False})
    return parse_review(extract_json(raw), raw=raw)


__all__ = [
    "ReviewFinding",
    "SpeechReview",
    "block_reason",
    "build_review_messages",
    "finding_summary",
    "is_blocked",
    "load_prompts",
    "parse_review",
    "review_speech",
]
