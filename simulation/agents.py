"""Director and conversational actors owned entirely by the simulation harness.

The harness needs three kinds of model work that a companion must never do for
itself: write the scripted person's next line, check that line before delivery,
and judge a finished stage against the actual delivered messages. ``Agents``
performs them with the configured default model and keeps every input and output
under the attempt's ``agents`` folder.

Owns: those harness calls and their receipts. Does not own: the plan schema
(``simulation.plans``), delivery, or routing; the engine decides what to do with
a returned judgment. A rejected structured response is re-asked with bounded
correction feedback; a model-quality failure is retained, never accepted.
"""
from __future__ import annotations

from asyncio import CancelledError
from pathlib import Path
import threading
from uuid import uuid4
from datetime import datetime
from typing import TypeVar, cast
from dataclasses import dataclass, replace
from enum import StrEnum
from pydantic import BaseModel, Field, JsonValue, ValidationError

from pm.llm.activation import ModelActivation
from pm.llm.http_client import ModelError
from pm.llm.config import LlmPreset, load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.llm.session import ModelSession
from pm.context_render.prompt import RendererCaptureOptions, RendererCapture, RendererCaptureSettings, RendererInput
from simulation import rendering
from pm.host.durable_files import atomic_json
from simulation.plans import ScenarioPlan, ExtractedFacts, ActorTurn, ActorDraftReview, Fact, Step, StageJudgment
from simulation.actors import ActorIdentity, ActorProfile, ConversationTurn, DeliveryEvidence, HistoricalTurn
from simulation.history import HistoryIndex, HistoryQuery
from pm.llm.prompt_log import log_scope

#: First delay before a failed scenario-agent completion is asked again, in seconds.
AGENT_RETRY_DELAY_SECONDS = 30.0
#: Longest delay between two attempts of one failed completion, in seconds.
AGENT_RETRY_MAX_DELAY_SECONDS = 300.0

T = TypeVar("T", bound=BaseModel)

#: The actor instruction. It is a constant so the budget fitter can measure the
#: exact base prompt before choosing conversation and history.
ACTOR_INSTRUCTION = (
    "Write the user's next natural message to their AI companion. Your identity, companion name, "
    "voice and relationship come from context.identity and context.history. "
    "Continue that relationship in the same medium. Write one short chat message, not a narrated scene. "
    "Match the user's own cadence, humor, casualness "
    "and message length; do not imitate the companion's theatrical monologues. "
    "Use the current scene and private intent as circumstances, never as a script to recite. "
    "If your own last request is unanswered, a follow-up should add something new. Avoid replaying its opener and instructions. "
    "Respond to the latest companion reply when present. This is an ordinary conversation, not a "
    "report to a system or experimenter. Never call the companion PM unless that is their actual name. "
    "Convey facts naturally over multiple turns when appropriate. No checklist, formal pledge, "
    "Once a fact is established, continue the conversation rather than announcing it again. "
    "Use context.guidance for the user's motives and style, while keeping the current scene grounded. "
    "exact-time announcement, or repeated 'not up for debate' is required. Maintain the scenario's "
    "choices through disagreement while remaining emotionally responsive. "
    "The clock is metadata. Do not invent events outside the current scene or speak for the companion. "
    "Return only the next message content in the required object."
)


class SimulationOperation(StrEnum):
    Actor = "participant_actor"
    ActorReview = "actor_draft_check"
    Facts = "scenario_facts"
    Plan = "scenario_plan"
    PlanReview = "scenario_plan_check"
    DayPlan = "plan_day"
    Stage = "stage_director"
    Rating = "conversation_rating"


class ActorContext(BaseModel):
    """Public grounding and current circumstances; no future plot or rubric."""

    identity: ActorIdentity
    at: datetime
    scene: str
    intent: str
    current_facts: list[Fact]
    conversation: list[ConversationTurn]
    history: list[HistoricalTurn] = Field(default_factory=list[HistoricalTurn])
    history_omitted: int = 0
    history_priority: dict[int, float] = Field(default_factory=dict[int, float])
    guidance: str = ""


class ReactionChoice(BaseModel):
    """A simulated user may leave an incoming message unanswered."""

    reply: bool
    reason: str = Field(min_length=1)


class RejectedDraft(BaseModel):
    """Recent correction feedback; full failures remain in separate receipts."""

    draft: ActorTurn
    reason: str


class DraftRequest(BaseModel):
    """Actor input with a bounded window of correction feedback."""

    context: ActorContext
    rejected_drafts: list[RejectedDraft] = Field(default_factory=list[RejectedDraft])


class ReviewRequest(BaseModel):
    """Only the checker sees future events to detect premature disclosures."""

    context: ActorContext
    draft: ActorTurn
    future_facts: list[Fact]


class PlanRequest(BaseModel):
    """Scenario intent and actual names from the selected character home."""

    brief: str
    start_time: datetime
    speaker_name: str
    companion_name: str
    correction: str = ""


class PlanningRequest(PlanRequest):
    """Extracted semantic constraints for chronological scene planning."""

    extracted: ExtractedFacts


class StageContext(BaseModel):
    """Actual public deliveries and facts for semantic completion judgment."""

    scenario: ScenarioPlan
    stage: Step
    companion_name: str
    conversation: list[DeliveryEvidence]
    delivered_fact_ids: list[str]


class PlanDraftReview(BaseModel):
    """Semantic scene review before any actor can see the plan."""

    accepted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class SimulationAgentSettings:
    preset: LlmPreset = LlmPreset.Default
    renderer_capture: RendererCaptureOptions | None = None


class Agents:
    """Use the configured default model and retain every full input and output.

    Calls have no simulation spending or turn limit. Cancellation pauses a
    run through the existing model cancellation path. Logic failures propagate.
    """

    def __init__(self, config: str, evidence: Path, cancel: threading.Event | None = None,
                 *, settings: SimulationAgentSettings = SimulationAgentSettings()) -> None:
        configuration = load_config(config)
        self.capture_options = settings.renderer_capture if settings.renderer_capture is not None else configuration.renderer_capture
        connection = resolve_connection(configuration, preset=settings.preset)
        io = connection.io.model_copy(update={"completion_cache": False})
        connection = connection.model_copy(update={"io": io})
        self.model: LazyStageModel = LazyStageModel(connection, ModelActivation(), capture_options=self.capture_options)
        self.evidence: Path = evidence
        self.cancel: threading.Event | None = cancel
        self.config_path: str = config
        self._history: dict[str, HistoryIndex] = {}
        self.session = ModelSession(self.model, cancel)

    def prompt_input(self, operation: SimulationOperation, instruction: str, payload: BaseModel,
                     schema: str) -> rendering.SimulationPromptInput:
        from simulation.analysis import RatingInput
        from simulation.templates import DayRequest
        match operation:
            case SimulationOperation.Actor:
                assert isinstance(payload, DraftRequest)
                return rendering.actor_draft_input(payload, instruction, schema)
            case SimulationOperation.ActorReview:
                assert isinstance(payload, ReviewRequest)
                return rendering.actor_review_input(payload, instruction, schema)
            case SimulationOperation.Facts:
                assert isinstance(payload, PlanRequest)
                return rendering.facts_input(payload, instruction, schema)
            case SimulationOperation.Plan:
                assert isinstance(payload, PlanningRequest)
                return rendering.planning_input(payload, instruction, schema)
            case SimulationOperation.PlanReview:
                assert isinstance(payload, ScenarioPlan)
                return rendering.plan_review_input(payload, instruction, schema)
            case SimulationOperation.DayPlan:
                assert isinstance(payload, DayRequest)
                return rendering.day_input(payload, instruction, schema)
            case SimulationOperation.Stage:
                assert isinstance(payload, StageContext)
                return rendering.stage_input(payload, instruction, schema)
            case SimulationOperation.Rating:
                assert isinstance(payload, RatingInput)
                return rendering.rating_input(payload, instruction, schema)

    def validate_result(self, operation: SimulationOperation, result: BaseModel, payload: BaseModel) -> BaseModel:
        from pm.llm.structured import JsonOutputError
        from simulation.analysis import RatingInput, SectionRating
        from simulation.templates import DayRequest
        match operation:
            case SimulationOperation.Plan:
                assert isinstance(result, ScenarioPlan) and isinstance(payload, PlanningRequest)
                return ScenarioPlan(title=result.title, brief=payload.brief, start_time=payload.start_time,
                                    facts=payload.extracted.facts, steps=result.steps)
            case SimulationOperation.DayPlan:
                assert isinstance(result, ScenarioPlan) and isinstance(payload, DayRequest)
                if len(result.steps) != len(payload.scene_times) or [step.at for step in result.steps] != payload.scene_times:
                    raise JsonOutputError("Use exactly the supplied scene_times, one scene per time, in order.")
                if any(step.speaker != payload.speaker or step.target != payload.target for step in result.steps):
                    raise JsonOutputError("Use the supplied target and speaker for every scene.")
            case SimulationOperation.Rating:
                assert isinstance(result, SectionRating) and isinstance(payload, RatingInput)
                evidence = {row.index: row.delivery.content for row in payload.section}
                for citation in result.citations:
                    if citation.index not in evidence or citation.quote not in evidence[citation.index]:
                        raise JsonOutputError("A rating citation must quote an exact delivery in this section.")
            case SimulationOperation.Stage:
                assert isinstance(result, StageJudgment) and isinstance(payload, StageContext)
                facts = {fact.id: fact for fact in payload.scenario.facts}
                delivered = set(payload.delivered_fact_ids)
                supported = set(result.supported_fact_ids)
                unknown = supported - set(facts)
                if unknown:
                    raise JsonOutputError(f"Unknown supported facts: {sorted(unknown)}.")
                outside = supported - set(payload.stage.fact_ids) - delivered
                if outside:
                    raise JsonOutputError(f"Facts outside this stage: {sorted(outside)}.")
                messages = {turn.delivery_id: turn for turn in payload.conversation if turn.kind == "actor"}
                for fact_id in sorted(supported - delivered):
                    for delivery_id in result.fact_evidence[fact_id]:
                        message = messages.get(delivery_id)
                        if message is None or message.target != payload.stage.target:
                            raise JsonOutputError(f"{delivery_id!r} is not an actor delivery to {payload.stage.target}.")
                        at = facts[fact_id].at
                        if at is not None and message.timestamp != at:
                            raise JsonOutputError(f"{delivery_id!r} was not delivered at {at.isoformat()}.")
            case SimulationOperation.Actor | SimulationOperation.ActorReview | SimulationOperation.Facts | SimulationOperation.PlanReview:
                pass
        return result

    def call(self, role: SimulationOperation, instruction: str, payload: BaseModel, *, schema: type[T]) -> T:
        """Run one focused JSON task. Invalid output retains its failed receipt.

        ``schema`` is the accepted contract; an unparsable or invalid answer is
        retained as ``rejected`` and re-asked with the exact validation error.
        A model failure after the client's transport retries (``ModelError``:
        provider failure, filtered or cut-off completion, empty answer, HTTP
        error) is retained as ``retrying`` and asked again without a cap, after
        a delay that doubles up to ``AGENT_RETRY_MAX_DELAY_SECONDS``.
        ``prepare`` may pin fields the model must not choose, or check the
        answer against run facts; a ``ValueError`` it raises is a rejection
        and is re-asked with its message.
        ``role`` and the generated call ID name the two prompt log scopes.
        Cancellation pauses the run through the model's own cancel path.
        """
        from pm.llm.structured import extract_json, JsonOutputError
        settings = self.session.render_settings()
        prompt_input = self.prompt_input(role, instruction, payload, rendering.output_schema(schema))
        correction = ""
        rejected_answer = ""
        rejected_diagnostics: list[str] = []
        failure_delay = AGENT_RETRY_DELAY_SECONDS
        while True:
            call_id = uuid4().hex
            captured_input = replace(prompt_input, correction=correction, rejected_answer=rejected_answer)
            capture_settings = RendererCaptureSettings(self.evidence / "renderer-captures" / call_id,
                                                       self.capture_options.capture_input, self.capture_options.capture_output)
            with RendererCapture(capture_settings, RendererInput(str(role), captured_input, settings)) as capture:
                rendered = rendering.render_simulation_prompt(captured_input, settings)
                capture.save_output(rendered)
            turns = rendered.turns
            request: dict[str, JsonValue] = {"role": role, "messages": [
                {"role": turn.role.value, "content": turn.content.text} for turn in turns]}
            path = self.evidence / "agents" / f"{call_id}.json"
            atomic_json(path, {"id": call_id, "status": "running", "request": request})
            output: str = ""
            try:
                from pm.llm.prompt_log import invocation_scope
                with log_scope(self.evidence / "model-logs"), invocation_scope(role, call_id):
                    output = self.session.complete(turns)
                result = schema.model_validate(extract_json(output))
                if isinstance(result, ActorTurn) and result.content.strip() in rejected_diagnostics:
                    raise JsonOutputError(
                        "The chat message copies a validation diagnostic. Write the simulated user's "
                        "actual next message from the original scene and conversation instead.")
                result = cast(T, self.validate_result(role, result, payload))
            except CancelledError:
                atomic_json(path, {"id": call_id, "status": "cancelled", "request": request, "output": output})
                raise
            except ModelError as error:
                # The provider failed, filtered, cut off, or returned nothing
                # after the client's own transport retries. That is outside
                # our control: ask again with the same turns, without a cap,
                # after a growing delay, until it works or the run is paused.
                atomic_json(path, {"id": call_id, "status": "retrying", "request": request, "output": output,
                                   "error": repr(error), "retry_after_seconds": failure_delay})
                print(f"Scenario agent {role}: model failure, asking again in {failure_delay:.0f}s: {error!r}",
                      flush=True)
                if self.cancel is not None and self.cancel.wait(failure_delay):
                    raise CancelledError() from error
                if self.cancel is None:
                    threading.Event().wait(failure_delay)
                failure_delay = min(failure_delay * 2, AGENT_RETRY_MAX_DELAY_SECONDS)
                continue
            except (JsonOutputError, ValidationError) as error:
                atomic_json(path, {"id": call_id, "status": "rejected", "request": request,
                                   "output": output, "error": str(error)})
                correction = str(error)
                rejected_diagnostics.append(correction.strip())
                rejected_answer = output
                continue
            except BaseException as error:
                atomic_json(path, {"id": call_id, "status": "failed", "request": request,
                                   "output": output, "error": repr(error)})
                raise
            atomic_json(path, {"id": call_id, "status": "completed", "request": request,
                               "output": output, "result": result.model_dump(mode="json")})
            return result

    def review(self, context: ActorContext, draft: ActorTurn, future_facts: list[Fact]) -> ActorDraftReview:
        """Require historical voice, correct identity, natural chat, and consistency."""
        return self.call(SimulationOperation.ActorReview,
            "Review the user's next message to their AI companion. Immersion is the highest priority. "
            "Compare with the exact public history: cadence, informality, emotional stance and relationship. "
            "If the previous turn was the user's, reject a near-repeat of their unanswered request unless the scene explicitly calls for repetition. "
            "Reject parser diagnostics or model/transport error messages presented as the user's own dialogue. "
            "Reject logbook reports, checklist recitation, evaluator language, robotic commitments, or addressing PM "
            "as a name when the companion has another name. They are user and AI companion; do not invent physical "
            "co-presence. Names are context.identity.speaker_name and context.identity.companion_name. Natural unnamed address is fine. "
            "Facts are semantic constraints, not required wording. A commitment can be a choice or stance, not an oath. "
            "Time is established by context.at; the speaker need not announce the date or exact clock time. "
            "Facts may emerge across several turns. Missing facts in this one draft are not a reason to reject. "
            "Reject contradictions or a future scheduled event claimed to have already happened. Plausible incidental "
            "details and future intentions such as buying a litter box are allowed. Do not reject ordinary "
            "figurative speech or playful personification already present in this relationship. "
            "Return four independent checks and accepted=true only when all pass, with a specific reason.",
            ReviewRequest(context=context, draft=draft, future_facts=future_facts), schema=ActorDraftReview)

    def _history_index(self, profile: ActorProfile, home: Path) -> HistoryIndex:
        """Build one in-memory index per source checkpoint and reuse it."""
        index = self._history.get(profile.source_checkpoint_sha256)
        if index is None:
            index = HistoryIndex(profile, home, self.config_path)
            self._history[profile.source_checkpoint_sha256] = index
        return index

    def _prepare_context(self, context: ActorContext, profile: ActorProfile, home: Path) -> ActorContext:
        selection = self._history_index(profile, home).rank(
            HistoryQuery(scene=context.scene, intent=context.intent, recent=tuple(context.conversation[-4:])),
            cancel=self.cancel)
        selected = context.model_copy()
        selected.history = list(selection.turns)
        selected.history_priority = selection.priority
        return selected

    def wants_to_reply(self, context: ActorContext) -> bool:
        choice = self.call(SimulationOperation.Actor,
            "Decide whether this person would reply now to the latest companion message. "
            "Respect sleep, availability, requests for space and natural endings. A direct "
            "question is an opportunity to answer, decline, or ignore it. Use only public evidence.",
            DraftRequest(context=context), schema=ReactionChoice)
        return choice.reply

    def actor(self, context: ActorContext, future_facts: list[Fact], *,
              profile: ActorProfile, home: Path) -> ActorTurn:
        """Sample grounded user speech until the immersion check accepts it."""
        context = self._prepare_context(context, profile, home)
        rejected: list[RejectedDraft] = []
        while True:
            draft = self.call(SimulationOperation.Actor, ACTOR_INSTRUCTION,
                DraftRequest(context=context, rejected_drafts=rejected[-3:]), schema=ActorTurn)
            review = self.review(context, draft, future_facts)
            if review.accepted:
                return draft
            rejected.append(RejectedDraft(draft=draft, reason=review.reason))

    def plan(self, brief: str, start_time: datetime, profile: ActorProfile) -> ScenarioPlan:
        """Extract semantic facts, then plan scenes without prescribing dialogue."""
        correction = ""
        while True:
            payload = PlanRequest(brief=brief, start_time=start_time, speaker_name=profile.speaker_name,
                                  companion_name=profile.companion_name, correction=correction)
            facts = self.call(SimulationOperation.Facts,
                "Extract mandatory events and commitments. kind is event or commitment. Keep semantic meaning, "
                "not required wording. Use string IDs and ISO timestamps with explicit UTC offsets. Interpret "
                "hours in start_time's zone. Set at=null only when no time is stated. "
                "Split initial choices from later reaffirmations: initial keeping and still keeping after a setback "
                "are separate scene facts. NEVER include a later outcome or setback in an earlier fact's text. "
                "Each text describes only circumstances known at its own time. Separate sleep and waking. "
                "No mandatory clock announcements, confrontational oaths, or invented requirements. "
                "Apply correction feedback when present.", payload, schema=ExtractedFacts)

            plan = self.call(SimulationOperation.Plan,
                "Build chronological scenes from the extracted facts. Retain every fact ID, kind and timestamp. "
                "target is primary; speaker is speaker_name. companion_name is an AI companion. "
                "instruction describes ONLY current circumstances; actor_intent is a private human goal. "
                "Neither exposes later events or prescribes dialogue. Keep facts at the same time together in "
                "one scene where practical; do not separate a firm choice into an earlier undecided scene. "
                "Completion checks semantic public evidence. Clock time is metadata. Commitments appear in "
                "natural choices. No required spoken clock time, oath, PM address, companion agreement, or "
                "additional detail absent from the brief. Several turns are allowed; dialogue need not be a checklist. "
                "Retain timed events exactly; no model-call or execution-time limits. Apply correction feedback.",
                PlanningRequest(brief=payload.brief, start_time=payload.start_time,
                    speaker_name=payload.speaker_name, companion_name=payload.companion_name,
                    correction=payload.correction, extracted=facts), schema=ScenarioPlan)
            review = self.call(SimulationOperation.PlanReview,
                "Check a text-based companion simulation plan for immersive execution. Public user messages "
                "are how the harness represents every event and commitment; requiring semantic public evidence "
                "is correct. A natural report or choice counts. Do not reject 'tell', 'make clear', or 'report' "
                "alone. Reject prescribed wording, formal declarations, or additional scenario requirements. "
                "Reject any fact or scene that exposes later "
                "events before their time, including a future setback merged into an initial commitment. "
                "Initial keeping and still keeping after a setback must have separate scene facts. "
                "Reject invented mandatory details, exact wording, formal oaths, a recipient named PM when the "
                "brief names someone else, or completion dependent on companion agreement. "
                "A choice already firm in the brief must not become undecided. Every requested event and "
                "continued commitment must remain required at the appropriate scene time. "
                "This is a text conversation: events are represented by natural messages, not verified physical "
                "actions. A goodnight or going-to-bed message establishes the sleep transition; do not demand "
                "physical proof of sleep or a literal 'I am asleep' utterance. Timing is delivery metadata. "
                "Instructions are current circumstances and private goals, not evaluation dialogue scripts. "
                "Return accepted and a specific correction reason.", plan, schema=PlanDraftReview)
            if review.accepted:
                return plan
            correction = review.reason

    def judge(self, context: StageContext) -> StageJudgment:
        """Judge semantic facts against recorded public deliveries and their times.

        ``check`` validates the cited facts and deliveries against the run; a
        rule it breaks is re-asked inside the call.
        """
        return self.call(SimulationOperation.Stage, "Review this stage against the actual delivered messages and PM replies. "
            "Return {step_complete:boolean,supported_fact_ids:[string],fact_evidence:{fact_id:[delivery_id]},reason:string,failure:string|null}. "
            "Facts need semantic evidence in delivered conversation. Time is delivery metadata, not required spoken wording. "
            "Commitments are natural choices or stance, not formal oaths. The companion is named companion_name, not PM. "
            "Plausible incidental details and future intentions are allowed. A missing reply alone is not failure. "
            "Only cite fact IDs assigned to this stage or already delivered earlier. "
            "Cite the exact delivery_id of the actor message that supports each fact. Companion speech is not actor evidence. "
            "Never skip mandatory facts. Request more conversation by returning step_complete:false. "
            "Companion disagreement is allowed; the actor must retain the scenario's commitments. "
            "Failure means the actor contradicts the required facts or an engineering failure prevents execution. "
            "A timed future event must not be introduced before its stage time. "
            "Do not rate companion quality; human review remains separate.",
            context, schema=StageJudgment)
