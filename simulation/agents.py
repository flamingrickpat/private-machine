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
from collections.abc import Callable
import json
from pathlib import Path
import threading
from uuid import uuid4
from datetime import datetime
from typing import TypeVar
from pydantic import BaseModel, Field, JsonValue, ValidationError

from pm.llm.activation import ModelActivation
from pm.llm.config import LlmPreset, load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.host.durable_files import atomic_json
from simulation.plans import ScenarioPlan, ExtractedFacts, ActorTurn, ActorDraftReview, Fact, Step, StageJudgment
from simulation.actors import ActorProfile, ConversationTurn, DeliveryEvidence
from pm.llm.prompt_log import log_scope

T = TypeVar("T", bound=BaseModel)


class ActorContext(BaseModel):
    """Public grounding and current circumstances; no future plot or rubric."""

    profile: ActorProfile
    at: datetime
    scene: str
    intent: str
    current_facts: list[Fact]
    conversation: list[ConversationTurn]
    guidance: str = ""


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


class Agents:
    """Use the configured default model and retain every full input and output.

    Calls have no simulation spending or turn limit. Cancellation pauses a
    run through the existing model cancellation path. Logic failures propagate.
    """

    def __init__(self, config: str, evidence: Path, cancel: threading.Event | None = None,
                 *, preset: LlmPreset = LlmPreset.Default) -> None:
        self.model: LazyStageModel = LazyStageModel(resolve_connection(load_config(config), preset=preset), ModelActivation())
        self.evidence: Path = evidence
        self.cancel: threading.Event | None = cancel

    def call(self, role: str, instruction: str, payload: BaseModel, *, schema: type[T],
             prepare: Callable[[T], T] | None = None) -> T:
        """Run one focused JSON task. Invalid output retains its failed receipt.

        ``schema`` is the accepted contract; an unparsable or invalid answer is
        retained as ``rejected`` and re-asked with the exact validation error.
        ``prepare`` may pin fields the model must not choose.
        ``role`` and the generated call ID name the two prompt log scopes.
        Cancellation pauses the run through the model's own cancel path.
        """
        from pm.llm.structured import extract_json, JsonOutputError
        required = " Required JSON schema: " + json.dumps(schema.model_json_schema())
        original: list[dict[str, str]] = [{"role": "system", "content": instruction +
            " Return a JSON data instance matching the schema. Do not return the schema itself." + required},
                                     {"role": "user", "content": payload.model_dump_json()}]
        turns = original
        while True:
            call_id = uuid4().hex
            request: dict[str, JsonValue] = {"role": role, "messages": [
                {"role": turn["role"], "content": turn["content"]} for turn in turns]}
            path = self.evidence / "agents" / f"{call_id}.json"
            atomic_json(path, {"id": call_id, "status": "running", "request": request})
            output: str = ""
            try:
                from pm.llm.prompt_log import invocation_scope
                with log_scope(self.evidence / "model-logs"), invocation_scope(role, call_id):
                    measured = self.model.measure_context(turns, cancel=self.cancel)
                    if measured["input_tokens"] > measured["input_budget_tokens"]:
                        raise ValueError("The scenario task exceeds the serving context. Its evidence was retained.")
                    output = self.model.complete(turns, cancel=self.cancel)
                result = schema.model_validate(extract_json(output))
                if prepare is not None:
                    result = prepare(result)
            except CancelledError:
                atomic_json(path, {"id": call_id, "status": "cancelled", "request": request, "output": output})
                raise
            except (JsonOutputError, ValidationError) as error:
                atomic_json(path, {"id": call_id, "status": "rejected", "request": request,
                                   "output": output, "error": str(error)})
                turns = [*original, {"role": "assistant", "content": output},
                         {"role": "user", "content": "Correct only this response to satisfy the schema: " + str(error)}]
                measured = self.model.measure_context(turns, cancel=self.cancel)
                if measured["input_tokens"] > measured["input_budget_tokens"]:
                    raise ValueError("Scenario repair filled the model context. Edit the retained plan or input.") from error
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
        return self.call("actor_draft_check",
            "Review the user's next message to their AI companion. Immersion is the highest priority. "
            "Compare with the exact public history: cadence, informality, emotional stance and relationship. "
            "Reject logbook reports, checklist recitation, evaluator language, robotic commitments, or addressing PM "
            "as a name when the companion has another name. They are user and AI companion; do not invent physical "
            "co-presence. Names are profile.speaker_name and profile.companion_name. Natural unnamed address is fine. "
            "Facts are semantic constraints, not required wording. A commitment can be a choice or stance, not an oath. "
            "Time is established by context.at; the speaker need not announce the date or exact clock time. "
            "Facts may emerge across several turns. Missing facts in this one draft are not a reason to reject. "
            "Reject contradictions or a future scheduled event claimed to have already happened. Plausible incidental "
            "details and future intentions such as buying a litter box are allowed. Do not reject ordinary "
            "figurative speech or playful personification already present in this relationship. "
            "Return four independent checks and accepted=true only when all pass, with a specific reason.",
            ReviewRequest(context=context, draft=draft, future_facts=future_facts), schema=ActorDraftReview)

    def actor(self, context: ActorContext, future_facts: list[Fact]) -> ActorTurn:
        """Sample grounded user speech until the immersion check accepts it."""
        rejected: list[RejectedDraft] = []
        while True:
            draft = self.call("participant_actor",
                "Write the user's next natural message to their AI companion. Your identity, companion name, "
                "voice and relationship come from context.profile and its complete public history. "
                "Continue that relationship in the same medium. Write one short chat message, not a narrated scene. "
                "Match the user's own cadence, humor, casualness "
                "and message length; do not imitate the companion's theatrical monologues. "
                "Use the current scene and private intent as circumstances, never as a script to recite. "
                "Respond to the latest companion reply when present. This is an ordinary conversation, not a "
                "report to a system or experimenter. Never call the companion PM unless that is their actual name. "
                "Convey facts naturally over multiple turns when appropriate. No checklist, formal pledge, "
                "Once a fact is established, continue the conversation rather than announcing it again. "
                "Use context.guidance for the user's motives and style, while keeping the current scene grounded. "
                "exact-time announcement, or repeated 'not up for debate' is required. Maintain the scenario's "
                "choices through disagreement while remaining emotionally responsive. "
                "The clock is metadata. Do not invent events outside the current scene or speak for the companion. "
                "Return only the next message content in the required object.",
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
            facts = self.call("scenario_facts",
                "Extract mandatory events and commitments. kind is event or commitment. Keep semantic meaning, "
                "not required wording. Use string IDs and ISO timestamps with explicit UTC offsets. Interpret "
                "hours in start_time's zone. Set at=null only when no time is stated. "
                "Split initial choices from later reaffirmations: initial keeping and still keeping after a setback "
                "are separate scene facts. NEVER include a later outcome or setback in an earlier fact's text. "
                "Each text describes only circumstances known at its own time. Separate sleep and waking. "
                "No mandatory clock announcements, confrontational oaths, or invented requirements. "
                "Apply correction feedback when present.", payload, schema=ExtractedFacts)

            def pin_facts(value: ScenarioPlan) -> ScenarioPlan:
                return ScenarioPlan(title=value.title, brief=brief, start_time=start_time,
                                    facts=facts.facts, steps=value.steps)

            plan = self.call("scenario_plan",
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
                    correction=payload.correction, extracted=facts), schema=ScenarioPlan, prepare=pin_facts)
            review = self.call("scenario_plan_check",
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
        """Judge semantic facts against recorded public deliveries and their times."""
        return self.call("stage_director", "Review this stage against the actual delivered messages and PM replies. "
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
