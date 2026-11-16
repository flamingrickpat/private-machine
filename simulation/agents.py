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
from pm.llm.http_client import ModelError
from pm.llm.config import LlmPreset, load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.llm.context_window import estimate_tokens
from pm.host.durable_files import atomic_json
from simulation.plans import ScenarioPlan, ExtractedFacts, ActorTurn, ActorDraftReview, Fact, Step, StageJudgment
from simulation.actors import ActorIdentity, ActorProfile, ConversationTurn, DeliveryEvidence, HistoricalTurn
from simulation.history import HistoryIndex, HistoryQuery, bound_conversation, conversation_tokens
from pm.llm.prompt_log import log_scope
from pm.context_render import render_context
from pm.context_render.budget import context_recency, fit_turns, optional_content
from pm.context_render.turns import Content, ContextTag, Role, Turn
from pm.llm.prompt_fitting import prompt_budget

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


def _turns(instruction: str, payload: BaseModel, schema: type[BaseModel]) -> list[Turn]:
    """Build the exact system and user turns that the provider receives."""
    required = " Required JSON schema: " + json.dumps(schema.model_json_schema())
    return [Turn(Role.System, instruction +
                 " Return a JSON data instance matching the schema. Do not return the schema itself." + required),
            Turn(Role.User, payload_content(payload))]


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
    history_salience: dict[int, float] = Field(default_factory=dict[int, float], description="Optional embedding scores for selected historical turns.")
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


def payload_content(payload: BaseModel) -> Content:
    """Keep task constraints required; select whole historical conversation turns."""
    if isinstance(payload, (DraftRequest, ReviewRequest)):
        content = Content(payload.model_dump_json(exclude={"context": {"conversation", "history", "history_salience"}}))
        content += "\ncontext.history (original public utterances):\n"
        for position, row in enumerate(payload.context.history):
            content += "\n" + optional_content(render_context(row),
                ContextTag(f"history:{row.record_id}", payload.context.history_salience.get(row.record_id, 0),
                           record_id=row.record_id, recency=context_recency(row.at)))
        content += "\ncontext.conversation (current run, chronological):\n"
        for position, row in enumerate(payload.context.conversation):
            content += "\n" + optional_content(render_context(row),
                ContextTag(f"conversation:{position}", 2,
                           required=position >= len(payload.context.conversation) - 4, recency=context_recency(row.at)))
        return content
    if isinstance(payload, StageContext):
        content = Content(payload.model_dump_json(exclude={"conversation"}))
        content += "\nconversation (actual public deliveries, chronological):\n"
        for position, row in enumerate(payload.conversation):
            content += "\n" + optional_content(render_context(row),
                ContextTag(f"delivery:{position}", 0,
                           required=row.timestamp >= payload.stage.at, recency=context_recency(row.timestamp)))
        return content
    return render_context(payload)


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
        self.config_path: str = config
        self._history: dict[str, HistoryIndex] = {}
        self._budget: int | None = None

    def call(self, role: str, instruction: str, payload: BaseModel, *, schema: type[T],
             prepare: Callable[[T], T] | None = None) -> T:
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
        original = _turns(instruction, payload, schema)
        turns = original
        failure_delay = AGENT_RETRY_DELAY_SECONDS
        while True:
            call_id = uuid4().hex
            request: dict[str, JsonValue] = {"role": role, "messages": [
                {"role": turn.role.value, "content": turn.content.text} for turn in turns]}
            path = self.evidence / "agents" / f"{call_id}.json"
            atomic_json(path, {"id": call_id, "status": "running", "request": request})
            output: str = ""
            try:
                from pm.llm.prompt_log import invocation_scope
                with log_scope(self.evidence / "model-logs"), invocation_scope(role, call_id):
                    capacity = self.model.measure_context([], cancel=self.cancel)
                    selection = fit_turns(turns, prompt_budget(capacity, internal=False))
                    request["messages"] = [{"role": turn.role.value, "content": turn.content.text}
                                           for turn in selection.turns]
                    atomic_json(path, {"id": call_id, "status": "running", "request": request,
                                       "selection": selection.receipt().model_dump(mode="json")})
                    output = self.model.complete(selection.turns, cancel=self.cancel)
                result = schema.model_validate(extract_json(output))
                if prepare is not None:
                    try:
                        result = prepare(result)
                    except ValueError as error:
                        # ``prepare`` checks the answer against facts the schema
                        # cannot hold (known fact IDs, recorded deliveries). A
                        # broken rule is a model mistake and is asked again.
                        raise JsonOutputError(str(error)) from error
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
                turns = [*original, Turn(Role.Assistant, output),
                         Turn(Role.User, "Correct only this response to satisfy the schema: " + str(error))]
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
            "co-presence. Names are context.identity.speaker_name and context.identity.companion_name. Natural unnamed address is fine. "
            "Facts are semantic constraints, not required wording. A commitment can be a choice or stance, not an oath. "
            "Time is established by context.at; the speaker need not announce the date or exact clock time. "
            "Facts may emerge across several turns. Missing facts in this one draft are not a reason to reject. "
            "Reject contradictions or a future scheduled event claimed to have already happened. Plausible incidental "
            "details and future intentions such as buying a litter box are allowed. Do not reject ordinary "
            "figurative speech or playful personification already present in this relationship. "
            "Return four independent checks and accepted=true only when all pass, with a specific reason.",
            ReviewRequest(context=context, draft=draft, future_facts=future_facts), schema=ActorDraftReview)

    def _input_budget(self) -> int:
        """Resolve the configured input ceiling once from the serving context."""
        if self._budget is None:
            measurement = self.model.measure_context([{"role": "user", "content": "x"}], cancel=self.cancel)
            self._budget = measurement["input_budget_tokens"]
        return self._budget

    def _history_index(self, profile: ActorProfile, home: Path) -> HistoryIndex:
        """Build one in-memory index per source checkpoint and reuse it."""
        index = self._history.get(profile.source_checkpoint_sha256)
        if index is None:
            index = HistoryIndex(profile, home, self.config_path)
            self._history[profile.source_checkpoint_sha256] = index
        return index

    def _fit_context(self, context: ActorContext, profile: ActorProfile, home: Path) -> ActorContext:
        """Fill the configured input budget with recent dialogue and retrieved history."""
        budget = self._input_budget()
        empty = context.model_copy(update={"conversation": [], "history": [], "history_omitted": 0})
        base = estimate_tokens(self.model.connection.model,
                               _turns(ACTOR_INSTRUCTION, DraftRequest(context=empty), ActorTurn))[0]
        reserve = max(1024, budget // 20)
        available = budget - base - reserve
        if available <= 0:
            raise ValueError("The scenario actor prompt leaves no room for conversation or history.")
        conversation_budget = available // 2
        conversation = bound_conversation(context.conversation, conversation_budget)
        # The conversation may use less than its half; history takes the rest.
        history_budget = available - conversation_tokens(conversation)
        selection = self._history_index(profile, home).select(
            HistoryQuery(scene=context.scene, intent=context.intent, recent=tuple(conversation)),
            history_budget, cancel=self.cancel)
        return context.model_copy(update={"conversation": conversation,
                                          "history": list(selection.turns),
                                          "history_omitted": selection.omitted_turns,
                                          "history_salience": selection.salience})

    def actor(self, context: ActorContext, future_facts: list[Fact], *,
              profile: ActorProfile, home: Path) -> ActorTurn:
        """Sample grounded user speech until the immersion check accepts it."""
        context = self._fit_context(context, profile, home)
        rejected: list[RejectedDraft] = []
        while True:
            draft = self.call("participant_actor", ACTOR_INSTRUCTION,
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

    def judge(self, context: StageContext, *,
              check: Callable[[StageJudgment], StageJudgment] | None = None) -> StageJudgment:
        """Judge semantic facts against recorded public deliveries and their times.

        ``check`` validates the cited facts and deliveries against the run; a
        rule it breaks is re-asked inside the call.
        """
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
            context, schema=StageJudgment, prepare=check)
