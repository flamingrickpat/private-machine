"""Agentic review of one public reply.

The review is one model call with a persona-aware prompt. It judges the reply
against the accepted state, the abilities, the private ontology, and the task
results. It returns a short ``judgment`` and a ``reason``.

The review is not a fixed policy. There are no per-kind weights and no score.
An advisory ``finding`` records something that looks unsupported, such as an
invented ability or an invented event. A finding stays in the receipt for
review. It never blocks a reply and never triggers regeneration.

Only a material ``false_task_result`` is a hard block. A false task result can change the
user's actions on a real object. A persona-driven reply, including a chosen
lie or a metaphor, must not become a false task result. Python decides the
block from the model's explicit ``false_task_result`` flag, so an invented
kind cannot reject a reply.

The caller records ``review.reason`` beside every accepted reply. The review
can select a verbatim speech span when narration surrounds it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal
from pm.constants import Sampler
from pm.context_render.turns import Role, Turn
from pm.llm.completion import AsyncTextCompletion

from pm.utils import yaml_util
from pydantic import BaseModel, ConfigDict, Field

from workflows.main_cognition.model_json import StageOutputError, extract_json
from workflows.public_speech.public_context import (
    DialogueTurn, PublicContext, ability_lines, memory_lines, task_lines,
)

REVIEW_FILE = Path(__file__).resolve().parent / "speech_review.yaml"

# The previous local gate returned empty review text at zero temperature.
# The shared instruct sampler replaces that tested setting.
# Transport repair never regenerates the reply.
REVIEW_TEMPERATURE = Sampler.TempInstruct


class ReviewFinding(BaseModel):
    """One advisory observation about the reply. It never blocks publication."""

    model_config = ConfigDict(extra="ignore")

    kind: str = ""
    quote: str = ""
    why: str = ""


class SpeechReview(BaseModel):
    """The result of one agentic review.

    ``judgment`` distinguishes supported, sufficient, equivalent, and blocked
    replies. Minor discrepancies retain their findings and permit publication.
    ``blocked`` is the runtime's
    decision and depends only on ``false_task_result``. ``findings`` are
    advisory and stay visible for human review.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "pm.speech-review.v2"
    judgment: Literal["accept", "good_enough", "equivalent_or_better", "block"] = "accept"
    reason: str = ""
    false_task_result: bool = False
    spoken_text: str | None = None
    findings: list[ReviewFinding] = Field(default_factory=list)
    raw: str = ""


def select_spoken_text(candidate: str, review: SpeechReview) -> str:
    """Keep a verbatim speech span, or the complete draft when none is selected.

    The review can remove narrator text at the edges. It cannot invent words,
    paraphrase speech, or supply an empty replacement. Invalid extraction raises
    ValueError and keeps both the draft and review available for diagnosis.
    """
    if review.spoken_text is None:
        return candidate
    text = review.spoken_text.strip()
    if not text or text not in candidate:
        raise ValueError("The reviewed speech must be a nonempty verbatim span of the draft.")
    return text


def load_prompts(path: Path | None = None) -> dict:
    """Load the review prompt file. It holds prose only, never weights."""
    return yaml_util.safe_load((path or REVIEW_FILE).read_text(encoding="utf-8"))


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
                          receipts: list[dict] | None = None) -> list[Turn]:
    """Return the two review messages. Pure: no model, no store."""
    names = {"companion_name": context.companion_name, "partner_name": context.partner_name}
    presentation = context.presentation
    if presentation.mode == "deceive":
        presentation_text = (f"{context.companion_name} chose to deceive on purpose. "
                             f"Purpose: {presentation.purpose or 'unstated'}. "
                             f"Known facts kept: {presentation.known_fact_ids or 'none listed'}. "
                             "The chosen presentation does not authorize a false task result.")
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
        reply=candidate,
        **names,
    )
    return [Turn(Role.System, prompts["system_prompt"].format(**names).strip()), Turn(Role.User, user.strip())]


def parse_review(payload: dict, raw: str = "") -> SpeechReview:
    """Read one review with an explicit boolean flag and a nonempty reason.

    Invalid fields or contradictory judgments raise ValueError. In particular,
    a passing label cannot hide a material false task result. The string
    ``false`` cannot become True through Python truth conversion.
    """
    if not isinstance(payload, dict):
        raise ValueError("The review model returned JSON that is not an object.")
    false_task_result = payload.get("false_task_result")
    reason = payload.get("reason")
    if type(false_task_result) is not bool or not isinstance(reason, str) or not reason.strip():
        raise ValueError("A speech review requires a boolean false_task_result and a nonempty reason.")
    findings = []
    for item in payload.get("findings", []) or []:
        if isinstance(item, dict):
            findings.append(ReviewFinding.model_validate(item))
        elif isinstance(item, str) and item.strip():
            findings.append(ReviewFinding(kind=item.strip()))
    judgment = str(payload.get("judgment", "") or "").strip().lower()
    if not judgment:
        judgment = "block" if false_task_result else "accept"
    if (judgment == "block") != false_task_result:
        raise ValueError("A blocking judgment requires a material false task result; passing judgments cannot contain one.")
    return SpeechReview(
        judgment=judgment,
        reason=reason.strip(),
        false_task_result=false_task_result,
        spoken_text=payload.get("spoken_text"),
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


async def review_speech(complete: AsyncTextCompletion, candidate: str, context: PublicContext, *,
                        conversation: list[DialogueTurn] = (),
                        receipts: list[dict] | None = None,
                        prompts: dict | None = None, attempts: int = 2) -> SpeechReview:
    """Review one reply, with one retry for an invalid review object.

    The reply is never changed or generated again. A retry repeats the same
    review when the model returns invalid JSON or invalid required fields.
    Raises ValueError or StageOutputError after the last attempt. Model
    transport errors and cancellation propagate from ``complete``.
    """
    prompts = prompts or load_prompts()
    messages = build_review_messages(candidate, context, prompts, conversation=conversation, receipts=receipts)
    last_error: ValueError | StageOutputError | None = None
    for _ in range(max(1, int(attempts))):
        raw = await complete(messages, temperature=REVIEW_TEMPERATURE,
                             enable_thinking=False)
        try:
            return parse_review(extract_json(raw), raw=raw)
        except (StageOutputError, ValueError) as error:
            last_error = error
    raise last_error


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
