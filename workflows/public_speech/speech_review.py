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

from pm.context_render.correction import render_regeneration_feedback

from pathlib import Path
from dataclasses import dataclass, field
from pm.context_render.prompt import RenderSettings, RenderedPrompt, EvidenceDecision, require_rendered_prompt, RendererCapture, RendererInput
from pm.records.media import RecordImage
from pm.context_render.budget import RequiredContextTooLarge, TokenEstimator
from pm.context_render.images import ImageCatalog, resolve_media
from collections.abc import Sequence
from typing import Literal, TypedDict
from pm.constants import Sampler
from pm.context_render.turns import Role, Turn
from pm.llm.session import ModelSession
from pm.llm.generation import GenerationSettings

from pm.utils import yaml_util
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, with_config, StrictBool

from workflows.public_speech.situational_guidance import render_situational_guidance

from workflows.public_speech.public_context import (
    DialogueTurn, PublicContext, EvidenceItem, ToolEvidence, ability_lines, memory_lines, task_lines,
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
    findings: list[ReviewFinding] = Field(default_factory=list[ReviewFinding])
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


@with_config(ConfigDict(extra="forbid"))
class ReviewPrompts(TypedDict):
    """The two prose templates in the existing review YAML file."""

    system_prompt: str
    user_template: str


def load_prompts(path: Path | None = None) -> ReviewPrompts:
    """Load the review prompt file. It holds prose only, never weights."""
    data = yaml_util.safe_load((path or REVIEW_FILE).read_text(encoding="utf-8"))
    return TypeAdapter(ReviewPrompts).validate_python(data)


def _ontology_lines(context: PublicContext) -> list[str]:
    """Render the private ontology terms and their active bindings.

    A term is a stable word for an implemented state. A binding shows the
    state that currently supports the term. The companion may use these words.
    An unsupported label is not in this list, so the review can mark it.
    """
    from workflows.public_speech.ontology import ontology_lines
    return ontology_lines(context)


def build_review_messages(candidate: str, context: PublicContext, prompts: ReviewPrompts, *,
                          conversation: Sequence[DialogueTurn] = (),
                          receipts: Sequence[ToolEvidence] = ()) -> list[Turn]:
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
    lines = [f'- [{turn.timestamp or "time unavailable"}; source {turn.knoxel_id}] {turn.speaker} said: "{turn.text}"'
             for turn in review_dialogue(context, conversation)]
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
        companion_name=names["companion_name"], partner_name=names["partner_name"],
    )
    grounding = "Selected purpose: " + (context.selected_action or "not supplied")
    if context.situational_guidance is not None:
        grounding += "\n" + render_situational_guidance(context.situational_guidance)
    user = grounding + "\n\n" + user
    return [Turn(Role.System, prompts["system_prompt"].format(companion_name=names["companion_name"], partner_name=names["partner_name"]).strip()), Turn(Role.User, user.strip())]


def review_dialogue(context: PublicContext, conversation: Sequence[DialogueTurn]) -> tuple[DialogueTurn, ...]:
    """Retain attributed dialogue even when the review omits the larger situation projection."""
    if conversation:
        return tuple(conversation)
    guidance = context.situational_guidance
    if guidance is None or guidance.partner is None:
        return ()
    return tuple(DialogueTurn(speaker=source.source, text=source.text, knoxel_id=source.record_id,
                              timestamp=source.occurred_at_utc.isoformat())
                 for source in guidance.partner.recent_dialogue if source.text)


class ReviewPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    false_task_result: StrictBool
    reason: str = Field(min_length=1)
    judgment: Literal["accept", "good_enough", "equivalent_or_better", "block"] | None = None
    spoken_text: str | None = None
    findings: list[ReviewFinding | str] | None = None


def parse_review(raw: str) -> SpeechReview:
    from pm.llm.structured import parse_response
    payload = parse_response(raw, ReviewPayload)
    if not payload.reason.strip():
        raise ValueError("A speech review requires a nonempty reason.")
    findings: list[ReviewFinding] = []
    for item in payload.findings or []:
        if isinstance(item, ReviewFinding):
            findings.append(item)
        elif item.strip():
            findings.append(ReviewFinding(kind=item.strip()))
        else:
            raise ValueError("A review finding cannot be empty.")
    judgment = payload.judgment or ("block" if payload.false_task_result else "accept")
    if (judgment == "block") != payload.false_task_result:
        raise ValueError("A blocking judgment requires a material false task result; passing judgments cannot contain one.")
    return SpeechReview(judgment=judgment, reason=payload.reason.strip(), false_task_result=payload.false_task_result,
        spoken_text=payload.spoken_text, findings=findings, raw=raw)


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


@dataclass(order=True, frozen=True, slots=True)
class RankedReviewEvidence:
    negative_score: float
    record_id: int
    evidence: EvidenceItem = field(compare=False)

    @property
    def score(self) -> float:
        return -self.negative_score


@dataclass(frozen=True, slots=True)
class SpeechReviewRenderInput:
    candidate: str
    context: PublicContext
    prompts: ReviewPrompts
    conversation: tuple[DialogueTurn, ...] = ()
    receipts: tuple[ToolEvidence, ...] = ()
    problem: str = ""
    rejected_answer: str = ""
    images: tuple[RecordImage, ...] = ()


def render_review_turns(input: SpeechReviewRenderInput, context: PublicContext, conversation: tuple[DialogueTurn, ...],
                        feedback: tuple[Turn, ...], images: ImageCatalog) -> tuple[Turn, ...]:
    messages = build_review_messages(input.candidate, context, input.prompts,
                                     conversation=conversation, receipts=input.receipts)
    return (*resolve_media(tuple(messages), images), *feedback)


def render_speech_review(input: SpeechReviewRenderInput, settings: RenderSettings) -> RenderedPrompt:
    from pm.memory.retrieval import distinctive_terms
    feedback_turns: tuple[Turn, ...] = ()
    if input.problem:
        feedback = render_regeneration_feedback("speech review", input.problem, settings)
        feedback_turns = (Turn(Role.User, feedback.text),)
        settings = settings.without_correction_reserve()
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    selected_context = input.context.model_copy()
    selected_context.evidence = []
    selected_context.memories = []
    dialogue = review_dialogue(input.context, input.conversation)
    partner_name = input.context.partner_name or input.context.user_name
    latest_partner = next((turn for turn in reversed(dialogue) if turn.speaker == partner_name), None)
    conversation = dialogue[-4:]
    if latest_partner is not None and latest_partner not in conversation:
        conversation = (latest_partner, *conversation[-3:])
    images = ImageCatalog(input.images)
    turns = render_review_turns(input, selected_context, conversation, feedback_turns, images)
    groups = ["draft and task evidence", "recent dialogue", "character context", "selected action"]
    if selected_context.situational_guidance is not None:
        # The situation projection includes its own dialogue and recent moves.
        # Keep the attributed transcript when that broader context will not fit.
        if estimator.estimate(turns) > settings.input_ceiling:
            selected_context.situational_guidance = None
            turns = render_review_turns(input, selected_context, conversation, feedback_turns, images)
        else:
            groups.append("situational guidance")
    if estimator.estimate(turns) > settings.input_ceiling:
        conversation = (latest_partner,) if latest_partner is not None else conversation[-1:]
        selected_context.character = ""
        selected_context.feelings = ""
        selected_context.ontology = []
        selected_context.capabilities = selected_context.capabilities.model_copy(
            update={"configured": (), "unavailable": (), "notes": ()})
        groups = ["draft and task evidence", "latest dialogue", "selected action"]
        turns = render_review_turns(input, selected_context, conversation, feedback_turns, images)
    if estimator.estimate(turns) > settings.input_ceiling:
        selected_context.selected_action = ""
        groups.remove("selected action")
        turns = render_review_turns(input, selected_context, conversation, feedback_turns, images)
    require_rendered_prompt("speech review", turns, settings)
    terms = set(distinctive_terms(input.candidate))
    ranked: list[RankedReviewEvidence] = []
    for item in input.context.evidence:
        matches = len(terms.intersection(distinctive_terms(item.text)))
        if matches:
            ranked.append(RankedReviewEvidence(-(matches + max(0.0, item.score)), item.knoxel_id, item))
    ranked.sort()
    selected: list[EvidenceItem] = []
    decisions: list[EvidenceDecision] = []
    seen: set[str] = set()
    for ranked_item in ranked:
        score, record_id, item = ranked_item.score, ranked_item.record_id, ranked_item.evidence
        text = " ".join(item.text.casefold().split())
        if text in seen:
            decisions.append(EvidenceDecision(record_id, score, False, "duplicate evidence"))
            continue
        seen.add(text)
        trial_context = selected_context.model_copy()
        trial_context.evidence = [*selected, item]
        trial_turns = render_review_turns(input, trial_context, conversation, feedback_turns, images)
        included = estimator.estimate(trial_turns) <= settings.input_ceiling
        decisions.append(EvidenceDecision(record_id, score, included, "draft relevance" if included else "budget"))
        if included:
            selected.append(item)
            turns = trial_turns
    return require_rendered_prompt("speech review", turns, settings, tuple(decisions), selected_groups=tuple(groups))


async def review_speech(complete: ModelSession, candidate: str, context: PublicContext, *,
                        conversation: Sequence[DialogueTurn] = (),
                        receipts: Sequence[ToolEvidence] = (),
                        prompts: ReviewPrompts | None = None, attempts: int = 2,
                        prepared_messages: Sequence[Turn] | None = None,
                        images: tuple[RecordImage, ...] = ()) -> SpeechReview:
    from pm.llm.structured import JsonOutputError, ModelFormatError
    from pydantic import ValidationError
    selected_prompts = prompts if prompts is not None else load_prompts()
    problem = ""
    rejected_answer = ""
    while True:
        input = SpeechReviewRenderInput(candidate, context, selected_prompts, tuple(conversation),
                                       tuple(receipts), problem, rejected_answer, images)
        try:
            with RendererCapture(complete.capture_settings("speech_review"), RendererInput("speech_review", input, complete.settings)) as capture:
                rendered = render_speech_review(input, complete.settings)
                capture.save_output(rendered)
            raw = await complete.complete_async(rendered.turns, settings=GenerationSettings(temperature=REVIEW_TEMPERATURE, enable_thinking=False))
        except RequiredContextTooLarge as error:
            raise ModelFormatError(f"Speech review could not fit its required evidence. {error}") from error
        try:
            result = parse_review(raw)
            select_spoken_text(candidate, result)
            return result
        except (JsonOutputError, ValidationError, ValueError) as error:
            problem = str(error)
            rejected_answer = raw



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
