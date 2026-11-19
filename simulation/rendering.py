from __future__ import annotations

from pm.context_render.correction import render_regeneration_feedback

from dataclasses import dataclass, field
import json
import re
from typing import TYPE_CHECKING
from pydantic import BaseModel

from pm.context_render.budget import TokenEstimator
from pm.context_render.prompt import EvidenceDecision, RenderedPrompt, RenderSettings, require_rendered_prompt
from pm.context_render.turns import Role, Turn
from simulation.actors import ConversationTurn, DeliveryEvidence, HistoricalTurn
from simulation.plans import Fact, ScenarioPlan, Step

if TYPE_CHECKING:
    from simulation.agents import ActorContext, DraftRequest, PlanRequest, PlanningRequest, ReviewRequest, StageContext
    from simulation.analysis import RatingInput
    from simulation.templates import DayRequest


@dataclass(frozen=True, slots=True)
class PromptEvidence:
    identity: str
    text: str
    priority: float
    order: int
    record_id: int = 0


@dataclass(frozen=True, slots=True, order=True)
class RankedPromptEvidence:
    negative_priority: float
    order: int
    identity: str
    evidence: PromptEvidence = field(compare=False)


@dataclass(frozen=True, slots=True, order=True)
class OrderedPromptEvidence:
    order: int
    identity: str
    evidence: PromptEvidence = field(compare=False)


@dataclass(frozen=True, slots=True)
class SimulationPromptInput:
    operation: str
    instruction: str
    output_schema: str
    required: str
    evidence: tuple[PromptEvidence, ...] = ()
    correction: str = ""
    rejected_answer: str = ""


def render_simulation_prompt(input: SimulationPromptInput, settings: RenderSettings) -> RenderedPrompt:
    system = input.instruction + (
        "\nReturn a JSON data instance matching this schema, not the schema itself:\n" + input.output_schema)
    correction: tuple[Turn, ...] = ()
    if input.correction:
        feedback = render_regeneration_feedback(input.operation, input.correction, settings)
        correction = (Turn(Role.User, feedback.text),)
        settings = settings.without_correction_reserve()
    required_turns = (Turn(Role.System, system), Turn(Role.User, input.required), *correction)
    require_rendered_prompt(input.operation, required_turns, settings)
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    chosen: list[PromptEvidence] = []
    decisions: list[EvidenceDecision] = []
    seen: set[str] = set()
    ranked = sorted(RankedPromptEvidence(-row.priority, row.order, row.identity, row) for row in input.evidence)
    for ranked_row in ranked:
        row = ranked_row.evidence
        if row.identity in seen:
            decisions.append(EvidenceDecision(row.record_id, row.priority, False, "duplicate " + row.identity))
            continue
        seen.add(row.identity)
        if row.priority <= 0:
            decisions.append(EvidenceDecision(row.record_id, row.priority, False, "irrelevant " + row.identity))
            continue
        ordered = sorted(OrderedPromptEvidence(item.order, item.identity, item) for item in [*chosen, row])
        trial = [item.evidence for item in ordered]
        text = input.required + "\n\nSelected evidence:\n" + "\n".join(item.text for item in trial)
        turns = (required_turns[0], Turn(Role.User, text), *correction)
        included = estimator.estimate(turns) <= settings.input_ceiling
        if included:
            chosen = trial
        decisions.append(EvidenceDecision(row.record_id, row.priority, included,
                                          ("selected " if included else "budget excluded ") + row.identity))
    text = input.required
    if chosen:
        text += "\n\nSelected evidence:\n" + "\n".join(item.text for item in chosen)
    return require_rendered_prompt(input.operation, (required_turns[0], Turn(Role.User, text), *correction),
                                   settings, tuple(decisions))


def output_schema(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False, separators=(",", ":"))


def fact_text(fact: Fact) -> str:
    at = fact.at.isoformat(timespec="minutes") if fact.at is not None else "unscheduled"
    return f"[{fact.id}; {fact.kind.value}; {at}] {fact.text}"


def scene_text(scene: Step) -> str:
    return (f"[{scene.id}; {scene.at.isoformat(timespec='minutes')}; target {scene.target}; speaker {scene.speaker}] "
            f"{scene.instruction}\nPrivate intent: {scene.actor_intent}\n"
            f"Required facts: {', '.join(scene.fact_ids)}\nCompletion: {scene.completion}")


def historical_text(turn: HistoricalTurn) -> str:
    return f"[memory {turn.record_id}; {turn.at.isoformat(timespec='minutes')}] {turn.speaker}: {turn.text}"


def conversation_text(turn: ConversationTurn) -> str:
    return f"[{turn.at}] {turn.speaker}: {turn.text}"


def delivery_text(turn: DeliveryEvidence) -> str:
    return (f"[{turn.delivery_id or 'no actor delivery ID'}; {turn.timestamp.isoformat(timespec='seconds')}; "
            f"{turn.kind}; target {turn.target or 'public'}] {turn.speaker}: {turn.content}")


def relevance(text: str, focus: str) -> float:
    stop = {"the", "and", "that", "this", "with", "from", "have", "your", "will", "their", "about"}
    words = set(re.findall(r"[\w']{4,}", text.casefold())) - stop
    query = set(re.findall(r"[\w']{4,}", focus.casefold())) - stop
    return len(words & query) / max(1, len(query))


def actor_required(context: ActorContext) -> str:
    return (f"Speaker: {context.identity.speaker_name}\nAI companion: {context.identity.companion_name}\n"
            f"Relationship: {context.identity.relationship}\nCurrent time: {context.at.isoformat(timespec='minutes')}\n"
            f"Current scene: {context.scene}\nPrivate intent: {context.intent}\nVoice guidance: {context.guidance}\n"
            "Current scenario constraints:\n" + "\n".join(fact_text(fact) for fact in context.current_facts))


def actor_evidence(context: ActorContext, *, draft: str = "") -> tuple[PromptEvidence, ...]:
    focus = "\n".join((context.scene, context.intent, draft))
    evidence: list[PromptEvidence] = []
    for index, turn in enumerate(context.conversation[:-2]):
        priority = 3 + index / max(1, len(context.conversation)) + relevance(turn.text, focus)
        evidence.append(PromptEvidence(f"conversation:{index}", conversation_text(turn), priority, index))
    base = len(context.conversation)
    for index, turn in enumerate(context.history):
        priority = 1 + context.history_priority.get(turn.record_id, 0) + relevance(turn.text, focus)
        evidence.append(PromptEvidence(f"memory:{turn.record_id}", historical_text(turn), priority,
                                       base + index, turn.record_id))
    return tuple(evidence)


def actor_draft_input(request: DraftRequest, instruction: str, schema: str) -> SimulationPromptInput:
    context = request.context
    required = actor_required(context)
    required += "\nLatest exchange:\n" + "\n".join(conversation_text(turn) for turn in context.conversation[-2:])
    latest = request.rejected_drafts[-1] if request.rejected_drafts else None
    return SimulationPromptInput("simulation actor", instruction, schema, required, actor_evidence(context),
        correction=latest.reason if latest is not None else "",
        rejected_answer=latest.draft.content if latest is not None else "")


def actor_review_input(request: ReviewRequest, instruction: str, schema: str) -> SimulationPromptInput:
    context = request.context
    required = actor_required(context) + "\nDraft to review:\n" + request.draft.content
    required += "\nLatest exchange:\n" + "\n".join(conversation_text(turn) for turn in context.conversation[-2:])
    evidence = list(actor_evidence(context, draft=request.draft.content))
    for index, fact in enumerate(request.future_facts):
        score = relevance(fact.text, request.draft.content)
        evidence.append(PromptEvidence(f"future:{fact.id}", "Future constraint: " + fact_text(fact),
                                       score * 10, 100000 + index))
    return SimulationPromptInput("simulation actor review", instruction, schema, required, tuple(evidence))


def facts_input(request: PlanRequest, instruction: str, schema: str) -> SimulationPromptInput:
    required = (f"User: {request.speaker_name}\nCompanion: {request.companion_name}\n"
                f"Start time: {request.start_time.isoformat()}\nBrief:\n{request.brief}")
    return SimulationPromptInput("simulation fact extraction", instruction, schema, required, correction=request.correction)


def planning_input(request: PlanningRequest, instruction: str, schema: str) -> SimulationPromptInput:
    base = facts_input(request, instruction, schema)
    required = base.required + "\nExtracted facts:\n" + "\n".join(fact_text(fact) for fact in request.extracted.facts)
    required += "\nChronology:\n" + "\n".join(f"{event.at.isoformat()}: {event.event}" for event in request.extracted.chronology)
    return SimulationPromptInput("simulation scene planning", instruction, schema, required, correction=base.correction)


def plan_review_input(plan: ScenarioPlan, instruction: str, schema: str) -> SimulationPromptInput:
    required = (f"Title: {plan.title}\nBrief: {plan.brief}\nStart time: {plan.start_time.isoformat()}\nFacts:\n"
                + "\n".join(fact_text(fact) for fact in plan.facts) + "\nScenes:\n"
                + "\n\n".join(scene_text(scene) for scene in plan.steps))
    return SimulationPromptInput("simulation plan review", instruction, schema, required)


def day_input(request: DayRequest, instruction: str, schema: str) -> SimulationPromptInput:
    director = request.director
    required = (f"Title: {request.title}\nBrief: {request.brief}\nDay number: {request.day_number}\n"
                f"Target: {request.target}\nSpeaker: {request.speaker}\nCompanion: {request.companion}\n"
                f"Scene times: {', '.join(time.isoformat() for time in request.scene_times)}\n"
                f"Daily rhythm: {director.daily_rhythm}\nStory threads: {director.story_threads}\n"
                f"Scheduled events: {director.scheduled_events}\nGuidance: {director.guidance}\n"
                f"Actor guidance: {director.actor_guidance}")
    evidence = tuple(PromptEvidence(scene.id, "Previous scene: " + scene_text(scene),
                                   1 + index / max(1, len(request.previous_scenes)), index)
                     for index, scene in enumerate(request.previous_scenes))
    return SimulationPromptInput("simulation daily planning", instruction, schema, required, evidence)


def stage_input(context: StageContext, instruction: str, schema: str) -> SimulationPromptInput:
    eligible = set(context.stage.fact_ids) | set(context.delivered_fact_ids)
    required = f"Companion: {context.companion_name}\nCurrent scene:\n{scene_text(context.stage)}\nRelevant facts:\n"
    required += "\n".join(fact_text(fact) for fact in context.scenario.facts if fact.id in eligible)
    required += "\nAlready delivered facts: " + ", ".join(context.delivered_fact_ids)
    required += "\nCurrent-stage deliveries:\n" + "\n".join(
        delivery_text(turn) for turn in context.conversation if turn.timestamp >= context.stage.at)
    earlier = tuple(PromptEvidence(f"delivery:{index}", delivery_text(turn),
                                  1 + index / max(1, len(context.conversation)), index)
                    for index, turn in enumerate(context.conversation) if turn.timestamp < context.stage.at)
    return SimulationPromptInput("simulation stage judgment", instruction, schema, required, earlier)


def rating_input(request: RatingInput, instruction: str, schema: str) -> SimulationPromptInput:
    required = (f"Criterion: {request.criterion.name}\nInstructions: {request.criterion.instructions}\n"
                f"Relationship context: {request.context}\nParticipants: {', '.join(request.participants)}\n"
                "Section to rate:\n")
    for row in request.section:
        turn = row.delivery
        required += f"[{row.index}; {turn.timestamp.isoformat(timespec='seconds')}] {turn.speaker}: {turn.content}\n"
    evidence: list[PromptEvidence] = []
    for row in request.preceding_context:
        turn = row.delivery
        evidence.append(PromptEvidence(f"preceding:{row.index}",
            f"Preceding [{row.index}] {turn.speaker}: {turn.content}", 3, row.index))
    for index, turn in enumerate(request.historical_reference):
        evidence.append(PromptEvidence(f"memory:{turn.record_id}", historical_text(turn), 1, -1000 + index, turn.record_id))
    return SimulationPromptInput("simulation conversation rating", instruction, schema, required, tuple(evidence))
